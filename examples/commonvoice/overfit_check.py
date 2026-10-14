"""
Trains each model and precision on one fixed batch of 8 LibriSpeech clips
for 300 steps. A setup that can learn drives the loss from about 330 close
to zero; one that cannot stays near where it started. Run it from the
repository root on the machine to check.
"""

import json
import platform
import time

import torch
from torch import nn

from speech import loader
from speech.models import CTC

STEPS = 300
TRAIN = "examples/librispeech/data/LibriSpeech/train-clean-100.json"
BIG = "examples/commonvoice/ctc_streaming_3750h_config.json"
SMALL = "examples/librispeech/ctc_streaming_100h_config.json"


def model_config(path):
    with open(path) as fid:
        return json.load(fid)["model"]


def main():
    torch.manual_seed(0)
    with open(TRAIN) as fid:
        data = [json.loads(line) for line in fid]
    data = sorted((d for d in data if 4 < d["duration"] < 6), key=lambda d: d["audio"])
    preproc = loader.Preprocessor(TRAIN, start_and_end=False)
    batch = list(zip(*[preproc.preprocess(d["audio"], d["text"]) for d in data[:8]]))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{platform.platform()}, torch {torch.__version__}, {device}")

    picks = [0, 25, 50, 100, 150, 200, 250, STEPS - 1]
    print(f"{'loss at step':24s}" + "".join(f"{i:>7d}" for i in picks))
    for name, path, dtype in [
        ("3750h model, float32", BIG, None),
        ("3750h model, bf16", BIG, torch.bfloat16),
        ("100h model, float32", SMALL, None),
        ("100h model, bf16", SMALL, torch.bfloat16),
    ]:
        torch.manual_seed(0)
        model = CTC(preproc.input_dim, preproc.vocab_size, model_config(path))
        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        start = time.time()
        losses = []
        for _ in range(STEPS):
            optimizer.zero_grad()
            with torch.autocast(
                device.type, dtype=dtype or torch.float32, enabled=dtype is not None
            ):
                loss = model.loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer.step()
            losses.append(loss.item())
        print(
            f"{name:24s}"
            + "".join(f"{losses[i]:7.1f}" for i in picks if i < len(losses))
            + f"  {time.time() - start:5.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
