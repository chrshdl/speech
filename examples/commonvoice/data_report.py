"""
Shows how a training run does on each kind of speech, and how well it
learned each source of its training data. Trains nothing. Run it from the
repository root on the machine with the data.

eval: the greedy CER and WER of each epoch on each dev set, the config's
and any others, such as SLURP's requests to a voice assistant from
examples/slurp/prepare.py, which are what the device hears:

    uv run examples/commonvoice/data_report.py eval \\
        examples/commonvoice/models/ctc_streaming_3750h_stride3 --epochs 0,1 \\
        --sets examples/slurp/data/slurp-test-headset.json \\
        examples/slurp/data/slurp-test-far.json

score: a random sample of each training source, decoded by a trained model:
per clip its CTC loss per character, CER, speaking rate and loudness (see
check_clips.py), a table comparing the sources, the clips whose transcript
cannot fit the model's output frames, whose CTC loss counts as 0 in
training, and report.html with the worst clips of each source to listen
to, which tells mislabeled data from merely hard speech:

    uv run examples/commonvoice/data_report.py score \\
        examples/commonvoice/models/ctc_streaming_3750h_stride3 --epoch 1

It writes every scored clip to scores.jsonl in --out, to filter on later.
"""

import argparse
import html
import itertools
import json
import os
import random

import numpy as np
import torch
from check_clips import score

import speech
from speech.utils.dataset import read_data_json

CONFIG = "examples/commonvoice/ctc_streaming_3750h_stride3_config.json"
BATCH = 16


def load(model_dir, epoch):
    """The model of the epoch, or with "best" or "last" that one, on the GPU."""
    if epoch in ("best", "last"):
        model, preproc = speech.load(model_dir, tag="best" if epoch == "best" else None)
    else:
        model, preproc = speech.load(model_dir, tag=None, epoch=int(epoch))
    model.to(speech.best_device())
    model.set_eval()
    return model, preproc


@torch.no_grad()
def decode(model, preproc, data):
    """(label, prediction) pairs of character lists of every clip, greedy."""
    results = []
    # Similar lengths together pad less.
    data = sorted(data, key=lambda d: d["duration"])
    for i in range(0, len(data), BATCH):
        batch = [preproc.preprocess(d["audio"], d["text"]) for d in data[i : i + BATCH]]
        inputs, labels = [x for x, _ in batch], [y for _, y in batch]
        preds = model.infer((inputs, labels))
        results += [
            (preproc.decode(y), preproc.decode(p)) for y, p in zip(labels, preds)
        ]
    return results


def run_eval(args, config):
    sets = list(config["data"]["dev_set"]) + list(args.sets or [])
    sets = [s for s in sets if os.path.exists(s) or print(f"Skipping {s}, not found.")]
    epochs = args.epochs.split(",")
    table = {}
    for epoch in epochs:
        model, preproc = load(args.model, epoch)
        for path in sets:
            results = decode(model, preproc, read_data_json(path))
            table[(path, epoch)] = (
                speech.compute_cer(results),
                speech.compute_wer(results),
            )
            print(
                f"epoch {epoch} {path}: CER {table[path, epoch][0]:.3f} WER {table[path, epoch][1]:.3f}"
            )
    width = max(len(os.path.basename(s)) for s in sets)
    print(
        f"\n{'CER / WER':{width}}  " + "  ".join(f"{'epoch ' + e:>13}" for e in epochs)
    )
    for path in sets:
        cells = [f"{table[path, e][0]:.3f} / {table[path, e][1]:.3f}" for e in epochs]
        print(
            f"{os.path.basename(path):{width}}  " + "  ".join(f"{c:>13}" for c in cells)
        )


def fits(model, inputs, labels):
    """Whether CTC can align the labels to the model's output frames."""
    frames = int(model.encoded_lengths(torch.tensor([inputs.shape[0]]))[0])
    repeats = sum(a == b for a, b in itertools.pairwise(labels))
    return frames >= len(labels) + repeats


def percentile(rows, key, q):
    return float(np.percentile([r[key] for r in rows], q)) if rows else float("nan")


def run_score(args, config):
    sources = list(args.sources or config["data"]["train_set"])
    model, preproc = load(args.model, args.epoch)
    rng = random.Random(0)
    os.makedirs(args.out, exist_ok=True)
    by_source = {}
    with open(os.path.join(args.out, "scores.jsonl"), "w") as out:
        for source in sources:
            data = read_data_json(source)
            sample = rng.sample(data, min(args.clips, len(data)))
            rows = score(model, preproc, sample)
            for row, d in zip(rows, sample):
                inputs, labels = preproc.preprocess(d["audio"], d["text"])
                row["fits"] = fits(model, inputs, labels)
                row["source"] = source
                # audio_stats() measures with NumPy, in float32.
                out.write(json.dumps(row, default=float) + "\n")
            by_source[source] = rows
            print(f"{source}: {len(rows)} of {len(data)} clips scored", flush=True)

    print(
        f"\nTraining sources scored by {args.model} epoch {args.epoch}, medians and shares:"
    )
    head = "source                     clips  dur s  chars/s  loss/char p50 p90  CER p50  CER>0.8  no fit  silent>10%"
    print(head)
    for source, rows in by_source.items():
        name = os.path.relpath(source, "examples")[:25]
        fit = [r for r in rows if r["fits"]]
        print(
            f"{name:25}  {len(rows):6}  {percentile(rows, 'duration', 50):5.1f}"
            f"  {percentile(rows, 'chars/s', 50):7.1f}"
            f"  {percentile(fit, 'loss/char', 50):9.2f} {percentile(fit, 'loss/char', 90):4.2f}"
            f"  {percentile(rows, 'CER', 50):7.2f}"
            f"  {np.mean([r['CER'] > 0.8 for r in rows]):7.1%}"
            f"  {np.mean([not r['fits'] for r in rows]):6.1%}"
            f"  {np.mean([r['zero %'] > 10 for r in rows]):10.1%}"
        )
    print("\nDurations, share of clips (the device hears requests of 2 to 4 s):")
    edges = [0, 2, 4, 8, 15, 30, float("inf")]
    labels = ["<2 s", "2-4", "4-8", "8-15", "15-30", ">30"]
    print(f"{'source':25}  " + "  ".join(f"{l:>6}" for l in labels))
    for source, rows in by_source.items():
        counts, _ = np.histogram([r["duration"] for r in rows], bins=edges)
        shares = counts / max(len(rows), 1)
        print(
            f"{os.path.relpath(source, 'examples')[:25]:25}  "
            + "  ".join(f"{s:6.1%}" for s in shares)
        )

    page = os.path.join(args.out, "report.html")
    with open(page, "w") as f:
        f.write(
            "<!doctype html><meta charset='utf-8'><title>Worst training clips</title>"
        )
        f.write(
            "<style>body{font:14px system-ui;margin:24px;max-width:1100px}"
            "td{padding:4px 8px;vertical-align:top;border-bottom:1px solid #ddd}"
            "audio{width:260px}.m{color:#666;white-space:nowrap}</style>"
        )
        f.write(
            f"<h1>Worst clips of each training source</h1><p>{html.escape(args.model)}, "
            f"epoch {html.escape(str(args.epoch))}, by loss per character. "
            "Listen for transcripts that do not match the audio, cuts in the middle of "
            "words, other languages and silence: those are data to drop. Hard but right "
            "clips are data to keep.</p>"
        )
        for source, rows in by_source.items():
            worst = sorted(
                (r for r in rows if r["fits"]), key=lambda r: -r["loss/char"]
            )
            worst = (
                worst[: args.worst] + [r for r in rows if not r["fits"]][: args.worst]
            )
            f.write(f"<h2>{html.escape(source)}</h2><table>")
            for r in worst:
                note = "" if r["fits"] else " <b>does not fit</b>"
                f.write(
                    f"<tr><td><audio controls preload='none' src='file://{html.escape(os.path.abspath(r['audio']))}'></audio></td>"
                    f"<td>{html.escape(r['text'])}<br><i>heard: {html.escape(r['hyp'])}</i></td>"
                    f"<td class='m'>loss/char {r['loss/char']:.2f}<br>CER {r['CER']:.2f}{note}<br>"
                    f"{r['duration']:.1f} s, {r['chars/s']:.0f} chars/s<br>peak {r['peak dBFS']:.0f} dBFS</td></tr>"
                )
            f.write("</table>")
    print(f"\nWrote {os.path.join(args.out, 'scores.jsonl')} and {page}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=CONFIG, help="The run's training config.")
    commands = parser.add_subparsers(dest="command", required=True)
    ev = commands.add_parser("eval", help="CER and WER of each epoch on each dev set.")
    ev.add_argument("model", help="The run's model directory.")
    ev.add_argument(
        "--epochs", default="best", help="Epochs from 0, or best or last, such as 0,1."
    )
    ev.add_argument("--sets", nargs="*", help="Dev sets besides the config's.")
    sc = commands.add_parser("score", help="Scores a sample of each training source.")
    sc.add_argument("model", help="The run's model directory.")
    sc.add_argument("--epoch", default="best", help="Epoch from 0, or best or last.")
    sc.add_argument(
        "--sources", nargs="*", help="Training sets instead of the config's."
    )
    sc.add_argument(
        "--clips", type=int, default=2000, help="Clips to score per source."
    )
    sc.add_argument(
        "--worst", type=int, default=30, help="Worst clips per source to list."
    )
    sc.add_argument(
        "--out", default="examples/commonvoice/data_report", help="Where to write."
    )
    args = parser.parse_args()
    with open(args.config) as fid:
        config = json.load(fid)
    (run_eval if args.command == "eval" else run_score)(args, config)


if __name__ == "__main__":
    main()
