import argparse
import itertools
import json
import time

import torch

import speech
import train
from speech import loader, models


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(model_cfg, data_json, batch_size, precision, steps, device, warmup=3):
    """
    Times training steps of a model on real batches, without data loading
    or augmentation. Returns the seconds of audio trained on per second.
    """
    preproc = loader.Preprocessor(data_json, start_and_end=False)
    torch.manual_seed(0)
    model_class = getattr(models, model_cfg["class"])
    model = model_class(preproc.input_dim, preproc.vocab_size, model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mixed = train.MixedPrecision(precision, device)
    ldr = loader.make_loader(data_json, preproc, batch_size, num_workers=0)
    batches = list(itertools.islice(iter(ldr), warmup + steps))
    if len(batches) < warmup + steps:
        raise ValueError(f"{data_json} has too few batches of {batch_size}")

    audio = 0.0
    for i, batch in enumerate(batches):
        if i == warmup:
            synchronize(device)
            start = time.time()
            audio = 0.0
        optimizer.zero_grad()
        with mixed.autocast():
            loss = model.loss(batch)
        mixed.scaler.scale(loss).backward()
        mixed.scaler.step(optimizer)
        mixed.scaler.update()
        if device.type == "mps":
            torch.mps.empty_cache()
        # The features have 100 frames per second.
        audio += sum(len(x) for x in batch[0]) / 100
    synchronize(device)
    return audio / (time.time() - start)


def training_hours(data_cfg):
    """The hours of audio in a config's training set, if it is prepared."""
    try:
        data = loader.read_data_json(data_cfg["train_set"])
    except FileNotFoundError:
        return None
    return sum(d["duration"] for d in data) / 3600


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure how fast a config's model trains on this machine, "
        "and estimate the time per epoch, before starting a long run."
    )
    parser.add_argument("config", help="A training config json.")
    parser.add_argument(
        "--data",
        help="A dataset json with batches to time, by default the config's dev set.",
    )
    parser.add_argument(
        "--batch-sizes", default=None, help="Comma separated, by default the config's."
    )
    parser.add_argument(
        "--precisions",
        default=None,
        help="Comma separated from none, bf16 and fp16, by default the config's.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Steps to time.")
    parser.add_argument(
        "--hours",
        type=float,
        help="Hours of training audio for the estimate, by default the config's "
        "training set if it is prepared.",
    )
    parser.add_argument("--device", type=torch.device, default=speech.best_device())
    args = parser.parse_args()

    with open(args.config) as fid:
        config = json.load(fid)
    data_json = args.data or config["data"]["dev_set"]
    if not isinstance(data_json, str):
        data_json = data_json[0]
    batch_sizes = [config["optimizer"]["batch_size"]]
    if args.batch_sizes:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    precisions = [config.get("mixed_precision")]
    if args.precisions:
        precisions = [None if p == "none" else p for p in args.precisions.split(",")]
    hours = args.hours or training_hours(config["data"])

    for batch_size, precision in itertools.product(batch_sizes, precisions):
        speed = benchmark(
            config["model"], data_json, batch_size, precision, args.steps, args.device
        )
        line = (
            f"{args.device.type}, batch {batch_size}, {precision or 'float32'}: "
            f"{speed:.0f} s of audio per second"
        )
        if hours:
            line += f", {hours / speed:.1f} h per epoch of {hours:,.0f} h"
        print(line, flush=True)
