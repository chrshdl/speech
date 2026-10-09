"""
Compares the clips a SortaGrad epoch starts with, the shortest ones,
against a random sample of the training set. For each group it decodes a
sample with a trained model and measures the audio: the peak and loudness,
how much of it is digital silence and how long the speech lasts. Ends with
the worst of the short clips, to listen to. Run it from the repository
root.
"""

import argparse
import json
import random

import editdistance
import numpy as np
import torch

import speech
from speech import loader
from speech.utils import wave

CONFIG = "examples/commonvoice/ctc_streaming_3750h_config.json"
MODEL = "examples/librispeech/models/ctc_streaming_100h"


def db(x):
    return 20 * np.log10(max(x, 1e-10))


def audio_stats(path):
    audio, rate = wave.array_from_wave(path)
    audio = audio.astype(np.float32) / 32768
    peak = np.abs(audio).max()
    # Speech: 20 ms frames within 30 dB of the loudest frame.
    frame = rate // 50
    n = len(audio) // frame
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    voiced = (20 * np.log10(rms / rms.max()) > -30).sum() * 0.02 if n else 0.0
    return {
        "peak dBFS": db(peak),
        "rms dBFS": db(np.sqrt((audio**2).mean())),
        "zero %": 100 * (audio == 0).mean(),
        "speech s": voiced,
    }


@torch.no_grad()
def score(model, preproc, data):
    rows = []
    int_to_char = preproc.int_to_char
    for d in data:
        inputs, labels = preproc.preprocess(d["audio"], d["text"])
        loss = model.loss(([inputs], [labels])).item()
        probs = model.forward_impl(torch.from_numpy(inputs).unsqueeze(0), softmax=True)[
            0
        ]
        best = model.max_decode(probs.argmax(1).tolist(), model.blank)
        hyp = "".join(int_to_char[i] for i in best)
        row = {
            "audio": d["audio"],
            "text": d["text"],
            "hyp": hyp,
            "duration": d["duration"],
            "chars/s": len(d["text"]) / d["duration"],
            "CER": editdistance.eval(hyp, d["text"]) / len(d["text"]),
            "loss/char": loss / len(d["text"]),
        }
        row.update(audio_stats(d["audio"]))
        rows.append(row)
    return rows


def summary(name, rows):
    keys = [
        "duration",
        "chars/s",
        "CER",
        "loss/char",
        "peak dBFS",
        "rms dBFS",
        "zero %",
        "speech s",
    ]
    print(f"\n{name}: {len(rows)} clips, medians")
    print("  " + "  ".join(f"{k} {np.median([r[k] for r in rows]):.2f}" for k in keys))
    cer = np.array([r["CER"] for r in rows])
    speech_s = np.array([r["speech s"] for r in rows])
    peak = np.array([r["peak dBFS"] for r in rows])
    zero = np.array([r["zero %"] for r in rows])
    print(
        f"  CER over 0.8: {(cer > 0.8).mean():.0%}, over 0.95: {(cer > 0.95).mean():.0%} | "
        f"speech under 0.5 s: {(speech_s < 0.5).mean():.0%} | "
        f"peak under -30 dBFS: {(peak < -30).mean():.0%} | "
        f"over 10% digital silence: {(zero > 10).mean():.0%}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument(
        "--model", default=MODEL, help="A trained model to decode with."
    )
    parser.add_argument("--clips", type=int, default=400, help="Clips per group.")
    parser.add_argument(
        "--batches",
        type=int,
        default=400,
        help="SortaGrad batches to count as the start.",
    )
    args = parser.parse_args()

    with open(args.config) as fid:
        config = json.load(fid)
    batch_size = config["optimizer"]["batch_size"]
    dataset = loader.AudioDataset(config["data"]["train_set"], None, batch_size)
    order = list(loader.BatchRandomSampler(dataset, batch_size, sortagrad=True))
    first = [dataset.data[i] for i in order[: args.batches * batch_size]]
    rng = random.Random(0)
    groups = {
        f"First {args.batches} SortaGrad batches": rng.sample(
            first, min(args.clips, len(first))
        ),
        "Random clips": rng.sample(dataset.data, args.clips),
    }

    model, preproc = speech.load(args.model, tag="best")
    model.set_eval()
    results = {name: score(model, preproc, data) for name, data in groups.items()}
    for name, rows in results.items():
        summary(name, rows)

    worst = sorted(next(iter(results.values())), key=lambda r: -r["CER"])[:12]
    print("\nWorst of the first clips:")
    for r in worst:
        print(
            f"  CER {r['CER']:.2f} | {r['duration']:.1f} s, speech {r['speech s']:.1f} s, "
            f"peak {r['peak dBFS']:.0f} dBFS | {r['audio']}\n"
            f"    text: {r['text']}\n    heard: {r['hyp']}"
        )


if __name__ == "__main__":
    main()
