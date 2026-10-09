"""
Trains several setups of the 3,750 hour model for a few hundred steps each
and reports, every 100 steps, the training loss per character, the greedy
CER on fixed Common Voice and LibriSpeech dev clips and how many GRU gates
are saturated. Ends with a summary table. Run it from the repository root
on the machine with the data.
"""

import argparse
import copy
import json
import os
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn

import speech
from speech import loader
from speech.models import CTC

CONFIG = "examples/commonvoice/ctc_streaming_3750h_config.json"
SMALL = "examples/librispeech/ctc_streaming_100h_config.json"
CV_TRAIN = "examples/commonvoice/data/trial_train_set.json"
CV_DEV = "examples/commonvoice/data/trial_dev_set.json"
LS_TRAIN = "examples/librispeech/data/LibriSpeech/train-clean-100.json"
LS_DEV = "examples/librispeech/data/LibriSpeech/dev-clean.json"
# The augmentations of the full run's first epoch.
EPOCH0_AUGMENT = ["volume", "pitch", "tempo", "spec_augment"]


def setups(base, small):
    big = base["model"]
    small_model = copy.deepcopy(small["model"])
    small_model["encoder"]["batch_norm"] = True
    small_model["encoder"]["relu_clip"] = 20
    return {
        "A": {"desc": "2e-4, SortaGrad (the stalled run)", "model": big, "lr": 2e-4},
        "B": {
            "desc": "2e-4, random order",
            "model": big,
            "lr": 2e-4,
            "sortagrad": False,
        },
        "C": {"desc": "1e-4, SortaGrad", "model": big, "lr": 1e-4},
        "D": {
            "desc": "2e-4, SortaGrad, LibriSpeech 100 h",
            "model": big,
            "lr": 2e-4,
            "data": "ls",
        },
        "E": {"desc": "small 4x512, 5e-4, SortaGrad", "model": small_model, "lr": 5e-4},
        "F": {
            "desc": "2e-4, SortaGrad, no augmentation",
            "model": big,
            "lr": 2e-4,
            "augment": False,
        },
    }


def sample_json(lines, n, seed, keep=lambda d: True):
    """Writes n random examples to a temporary dataset file."""
    rng = random.Random(seed)
    chosen = [line for line in lines if keep(json.loads(line))]
    chosen = rng.sample(chosen, min(n, len(chosen)))
    if not chosen:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fid:
        fid.writelines(chosen)
    return fid.name


def autocast(device, mixed_precision):
    """The forward pass's precision, as in train.py."""
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mixed_precision)
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)


@torch.no_grad()
def dev_cer(model, ldr, preproc, precision):
    model.set_eval()
    results = []
    with precision():
        for batch in ldr:
            preds = model.infer(batch)
            results.extend(
                (preproc.decode(lab), preproc.decode(p))
                for lab, p in zip(batch[1], preds)
            )
    model.set_train()
    return speech.compute_cer(results)


@torch.no_grad()
def saturation(model, x):
    """The largest share of saturated update gates in any GRU layer."""
    model.set_eval()
    h = model.flatten_conv(model.conv_forward(x.unsqueeze(1))).float()
    worst = 0.0
    for norm, gru in zip(model.rnn.norms, model.rnn.layers):
        xin = model.rnn.frames(h, norm)
        out, _ = gru(xin)
        prev = torch.cat([torch.zeros_like(out[:, :1]), out[:, :-1]], dim=1)
        zi = (xin @ gru.weight_ih_l0.T + gru.bias_ih_l0).chunk(3, -1)[1]
        zh = (prev @ gru.weight_hh_l0.T + gru.bias_hh_l0).chunk(3, -1)[1]
        z = torch.sigmoid(zi + zh)
        worst = max(worst, ((z > 0.99) | (z < 0.01)).float().mean().item())
        h = out
    model.set_train()
    return worst


def run(name, setup, base, args, devs, device):
    torch.manual_seed(0)
    random.seed(0)
    data_cfg = base["data"]
    train_json = LS_TRAIN if setup.get("data") == "ls" else args.train
    preproc = loader.Preprocessor(train_json, start_and_end=False)
    augment = {}
    if setup.get("augment", True):
        augment = {
            k: {kk: v for kk, v in data_cfg[k].items() if kk != "from_epoch"}
            for k in EPOCH0_AUGMENT
        }
    ldr = loader.make_loader(
        train_json,
        preproc,
        base["optimizer"]["batch_size"],
        args.workers,
        augment=augment,
        sortagrad=setup.get("sortagrad", True),
    )
    dev_ldrs = {k: loader.make_loader(p, preproc, 16, 0) for k, p in devs.items()}
    with open(devs["ls"]) as fid:
        probe_audio = json.loads(fid.readline())["audio"]
    probe = torch.from_numpy(preproc.preprocess(probe_audio, "")[0])
    probe = probe.unsqueeze(0).to(device)

    model = CTC(preproc.input_dim, preproc.vocab_size, setup["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=setup["lr"])

    def precision():
        return autocast(device, base.get("mixed_precision"))

    model.set_train()
    per_char, rows = [], []
    start = time.time()
    print(f"[{name}] {setup['desc']}", flush=True)
    for step, batch in enumerate(ldr):
        if step == args.steps:
            break
        optimizer.zero_grad()
        with precision():
            loss = model.loss(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 200)
        optimizer.step()
        if device.type == "mps":
            torch.mps.empty_cache()
        per_char.append(loss.item() / np.mean([len(lab) for lab in batch[1]]))
        if (step + 1) % args.every == 0:
            cers = {
                k: dev_cer(model, d, preproc, precision) for k, d in dev_ldrs.items()
            }
            row = dict(
                step=step + 1,
                loss=float(np.median(per_char[-100:])),
                sat=saturation(model, probe),
                **cers,
            )
            rows.append(row)
            cer_text = " ".join(f"{k} CER {v:.3f}" for k, v in cers.items())
            print(
                f"[{name}] step {row['step']:4d} loss/char {row['loss']:.2f} {cer_text}"
                f" | worst gate saturation {row['sat']:.0%} | {time.time() - start:4.0f}s",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--setups", default="A,B,C,D,E,F")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=100, help="Steps between reports.")
    parser.add_argument("--train", default=CV_TRAIN, help="The mixed training sample.")
    parser.add_argument(
        "--dev", default=CV_DEV, help="Dev clips to take Common Voice clips from."
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(CONFIG) as fid:
        base = json.load(fid)
    with open(SMALL) as fid:
        small = json.load(fid)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open(args.dev) as fid:
        dev_lines = fid.readlines()
    with open(LS_DEV) as fid:
        ls_lines = fid.readlines()
    devs = {
        "cv": sample_json(dev_lines, 64, 0, lambda d: "commonvoice" in d["audio"]),
        "ls": sample_json(ls_lines, 64, 0),
    }
    devs = {k: v for k, v in devs.items() if v is not None}

    results = {}
    all_setups = setups(base, small)
    for name in args.setups.split(","):
        results[name] = run(name, all_setups[name], base, args, devs, device)

    print("\nSummary at the last step:")
    print(
        f"{'setup':40s} {'loss/char':>9s}"
        + "".join(f" {k + ' CER':>7s}" for k in devs)
        + f" {'gate sat':>8s}"
    )
    for name, rows in results.items():
        last = rows[-1]
        print(
            f"{name}: {all_setups[name]['desc']:37s} {last['loss']:9.2f}"
            + "".join(f" {last[k]:7.3f}" for k in devs)
            + f" {last['sat']:8.0%}"
        )
    for path in devs.values():
        os.unlink(path)


if __name__ == "__main__":
    main()
