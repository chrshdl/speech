"""
Measures how fast a config's model trains on this machine, and estimates
the hours per epoch for each batch size and precision, before starting a
long run. Each batch size accumulates its gradients over as many batches
as keep the config's effective batch (batch_size times accumulate), so
the runs it compares learn the same way.

It times the training step on synthetic batches as long as the clips at
percentiles of the training set's lengths, from the shortest to the
longest, and sums the times of the real batches, which the loader forms
from clips of about the same length. It also reports the peak memory
against what the device can hold and how fast the data loader prepares
augmented clips, since an epoch takes the slower of the model and the
loader. Run it while nothing else trains.
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn

import speech
import train
from speech import loader, models
from speech.features import STEP_MS

FRAMES_PER_SECOND = 1000 // STEP_MS
# The clip lengths to time, as percentiles of the training clips' lengths.
PERCENTILES = (0, 20, 40, 60, 80, 95, 100)
GB = 1024**3


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_limit(device):
    """The memory the device can hold, or None on the CPU."""
    if device.type == "mps":
        return torch.mps.recommended_max_memory()
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory
    return None


def clip_lengths(durations):
    """The clip lengths to time, in seconds."""
    return sorted({round(float(s), 1) for s in np.percentile(durations, PERCENTILES)})


def batch_lengths(durations, batch_size):
    """
    The longest clip of each training batch, which are consecutive runs
    of the dataset's order, as loader.BatchRandomSampler makes them.
    """
    ends = range(batch_size, len(durations) + 1, batch_size)
    return np.array([max(durations[e - batch_size : e]) for e in ends])


def synthetic_batch(preproc, batch_size, secs, chars_per_sec):
    """Random features and labels of clips about secs long."""
    frames = max(int(secs * FRAMES_PER_SECOND), 32)
    inputs = [
        np.random.randn(frames - random.randint(0, 10), preproc.input_dim).astype(
            np.float32
        )
        for _ in range(batch_size)
    ]
    n = max(1, round(secs * chars_per_sec))
    labels = [
        [random.randrange(preproc.vocab_size) for _ in range(n)]
        for _ in range(batch_size)
    ]
    return inputs, labels


def time_step(model, batch, precision, accumulate, steps, device):
    """
    Returns the median seconds of a forward and backward pass after a
    warmup, as train.py runs them, and the peak memory, or None on the
    CPU.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak = None
    times = []
    for i in range(steps + 1):
        start = time.time()
        with precision.autocast():
            loss = model.loss(batch)
        precision.scaler.scale(loss / accumulate).backward()
        loss.item()
        synchronize(device)
        if device.type == "mps":
            # Freed blocks stay cached until empty_cache, so this is the peak.
            peak = max(peak or 0, torch.mps.driver_allocated_memory())
            torch.mps.empty_cache()
        if i > 0:
            times.append(time.time() - start)
    if device.type == "cuda":
        peak = torch.cuda.max_memory_reserved(device)
    return float(np.median(times)), peak


def time_optimizer_step(model, optimizer, grad_clip, device, steps=5):
    """The median seconds of clipping the gradients and an Adam step."""
    times = []
    for _ in range(steps):
        start = time.time()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        optimizer.step()
        synchronize(device)
        times.append(time.time() - start)
    return float(np.median(times))


def model_hours(durations, batch_size, accumulate, step_seconds, optimizer_seconds):
    """
    The hours of an epoch's training steps, from the seconds per batch at
    each timed clip length, a dict that has None for lengths that ran out
    of memory. Returns None if the longest batch does not fit.
    """
    fits = sorted((s, t) for s, t in step_seconds.items() if t is not None)
    lengths = batch_lengths(durations, batch_size)
    if not fits or lengths.max() > fits[-1][0]:
        return None
    xs, ys = zip(*fits)
    seconds = np.interp(lengths, xs, ys).sum()
    seconds += np.ceil(len(lengths) / accumulate) * optimizer_seconds
    return seconds / 3600


def loader_rate(data_json, preproc, augment, batch_size, workers, clips):
    """
    Returns the seconds of audio per second the loader's workers load,
    featurize and augment, with every augmentation of the config active.
    """
    ldr = loader.make_loader(data_json, preproc, batch_size, workers, augment=augment)
    configs = [c for v in augment.values() for c in (v if isinstance(v, list) else [v])]
    ldr.dataset.set_epoch(max((c.get("from_epoch", 0) for c in configs), default=0))
    audio, n, start = 0.0, 0, None
    for i, (inputs, _) in enumerate(ldr):
        # Starting the workers and filling their prefetch queue is not steady.
        if i == 2 * workers:
            start = time.time()
        if start is None:
            continue
        audio += sum(len(x) for x in inputs) / FRAMES_PER_SECOND
        n += len(inputs)
        if n >= clips:
            break
    return audio / (time.time() - start) if n else None


def other_training():
    """The processes running train.py, which skew the timings."""
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "train.py"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return []
    return out.stdout.splitlines()


def gb(n):
    return "-" if n is None else f"{n / GB:.1f} GB"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="A training config json.")
    parser.add_argument(
        "--data",
        help="A dataset json whose clip lengths to time and estimate the epoch "
        "of, by default the config's training set.",
    )
    parser.add_argument(
        "--batch-sizes", default=None, help="Comma separated, by default the config's."
    )
    parser.add_argument(
        "--precisions",
        default=None,
        help="Comma separated from none, bf16 and fp16, by default the config's.",
    )
    parser.add_argument(
        "--steps", type=int, default=2, help="Steps to time at each clip length."
    )
    parser.add_argument(
        "--epoch-hours",
        type=float,
        help="How long an epoch takes with the config's batch size and "
        "precision, to scale the estimates to.",
    )
    parser.add_argument(
        "--loader-clips",
        type=int,
        default=3000,
        help="Clips to time the data loader on, or 0 to skip it.",
    )
    parser.add_argument("--device", type=torch.device, default=speech.best_device())
    args = parser.parse_args()

    others = other_training()
    if others:
        print("WARNING: training seems to be running, which skews the timings:")
        print("\n".join("  " + line for line in others))

    with open(args.config) as fid:
        config = json.load(fid)
    opt_cfg, data_cfg = config["optimizer"], config["data"]
    data_json = args.data or data_cfg["train_set"]
    effective = opt_cfg["batch_size"] * opt_cfg.get("accumulate", 1)
    batch_sizes = [opt_cfg["batch_size"]]
    if args.batch_sizes:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    for b in batch_sizes:
        if effective % b:
            parser.error(f"{b} does not divide the config's batches of {effective}.")
    precisions = [config.get("mixed_precision")]
    if args.precisions:
        precisions = [None if p == "none" else p for p in args.precisions.split(",")]
    device = args.device

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    preproc = loader.Preprocessor(data_json, start_and_end=data_cfg["start_and_end"])
    data = loader.AudioDataset(data_json, preproc, batch_sizes[0]).data
    durations = [d["duration"] for d in data]
    total_hours = sum(durations) / 3600
    chars_per_sec = float(np.median([len(d["text"]) / d["duration"] for d in data]))
    grid = clip_lengths(durations)
    print(
        f"{len(data):,} clips, {total_hours:,.0f} hours, {chars_per_sec:.1f} "
        f"characters per second. Timing clips of {', '.join(map(str, grid))} s."
    )
    limit = memory_limit(device)
    if limit:
        print(f"The {device.type} device holds {gb(limit)}.")
    print(flush=True)

    model_class = getattr(models, config["model"]["class"])
    model = model_class(preproc.input_dim, preproc.vocab_size, config["model"])
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["learning_rate"])

    # Seconds per batch at each clip length, None where it ran out of
    # memory, and the peak memory, for each batch size and precision.
    step_seconds, peaks = {}, {}
    for b, p in itertools.product(batch_sizes, precisions):
        precision = train.MixedPrecision(p, device)
        step_seconds[b, p], peaks[b, p] = {}, None
        oom = False
        for secs in grid:
            # Longer clips than ones that ran out of memory do too.
            if oom:
                step_seconds[b, p][secs] = None
                continue
            name = f"batch {b:>4}, {p or 'float32':>7}, {secs:>5} s clips"
            batch = synthetic_batch(preproc, b, secs, chars_per_sec)
            try:
                s, peak = time_step(
                    model, batch, precision, effective // b, args.steps, device
                )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"{name}: out of memory", flush=True)
                step_seconds[b, p][secs] = None
                oom = True
                model.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            step_seconds[b, p][secs] = s
            if peak is not None:
                peaks[b, p] = max(peaks[b, p] or 0, peak)
            over = "  (beyond what the device holds)" if limit and peak > limit else ""
            print(
                f"{name}: {s:6.2f} s/batch, {s / b * 1000:6.1f} ms/clip, "
                f"{gb(peak)}{over}",
                flush=True,
            )
    optimizer_seconds = time_optimizer_step(
        model, optimizer, opt_cfg["grad_clip"], device
    )
    print(f"optimizer step: {optimizer_seconds * 1000:.0f} ms\n", flush=True)
    del model, optimizer

    workers = data_cfg.get("num_workers", 4)
    loader_hours = None
    if args.loader_clips:
        augment = {n: data_cfg[n] for n in loader.AUGMENTATIONS if n in data_cfg}
        for w in sorted({workers, max(workers, (os.cpu_count() or 1) - 2)}):
            rate = loader_rate(
                data_json,
                preproc,
                augment,
                opt_cfg["batch_size"],
                w,
                args.loader_clips,
            )
            if rate is None:
                print(f"loader, {w:>2} workers: too few clips to time")
                continue
            print(
                f"loader, {w:>2} workers: {rate:6.0f} s of audio per s, "
                f"{total_hours / rate:5.1f} hours per epoch",
                flush=True,
            )
            if w == workers:
                loader_hours = total_hours / rate

    # The loader's workers prepare the next batches while the model trains.
    rows = {}
    for (b, p), seconds in step_seconds.items():
        hours = model_hours(durations, b, effective // b, seconds, optimizer_seconds)
        epoch = None if hours is None else max(hours, loader_hours or 0)
        rows[b, p] = (hours, epoch)
    base = rows.get((opt_cfg["batch_size"], config.get("mixed_precision")))
    base = base[1] if base else None
    scale = args.epoch_hours / base if args.epoch_hours and base else None

    print(f"\nEpoch estimates with the config's {workers} loader workers:\n")
    head = f"{'batch':>5} {'accum':>5} {'precision':>9} {'model h':>8} {'epoch h':>8}"
    head += f" {'peak memory':>12}"
    if scale:
        head += f" {f'scaled to {args.epoch_hours:g} h':>16}"
    print(head)
    for (b, p), (hours, epoch) in rows.items():
        line = f"{b:>5} {effective // b:>5} {p or 'float32':>9} "
        if hours is None:
            print(line + "out of memory on the longest batch")
            continue
        line += f"{hours:8.1f} {epoch:8.1f} {gb(peaks[b, p]):>12}"
        if scale:
            line += f" {epoch * scale:16.1f}"
        if limit and peaks[b, p] and peaks[b, p] > limit:
            line += "  beyond the device's memory"
        print(line)
    print(
        "\nmodel h: the training steps alone. epoch h: the slower of them and "
        "the loader."
    )
    if scale:
        print(
            f"Scaled: --epoch-hours times each estimate's ratio to the config's, "
            f"which is {base:.1f} h. A large gap between the two means the "
            "estimate misses a cost, so trust the scaled column less."
        )
    elif args.epoch_hours:
        print(
            "The config's batch size and precision were not timed, so nothing is scaled."
        )
    fastest = [h for h, _ in rows.values() if h is not None]
    if loader_hours and fastest and loader_hours >= min(fastest):
        print(
            f"The loader limits the epoch with {workers} workers. More workers, "
            "as timed above, help more than another batch size."
        )


if __name__ == "__main__":
    main()
