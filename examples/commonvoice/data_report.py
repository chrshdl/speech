"""
Shows how a training run does on each kind of speech, and how well it
learned each source of its training data, and cleans a source of the
clips whose transcript does not match the audio. Run it from the
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

clean: scores every clip of a dataset, in batches, and writes the scores
next to it, then a page of clips from each band of the highest losses to
listen to, to find the loss above which the transcripts stop matching the
audio, such as VoxPopuli's, whose transcripts are the Parliament's edited
record rather than what was said. With that loss as --max-loss, it writes
the dataset without those clips and the ones that cannot fit:

    uv run examples/commonvoice/data_report.py clean \\
        examples/commonvoice/models/ctc_streaming_3750h_stride3 --epoch 2 \\
        examples/voxpopuli/data/train.json
    uv run examples/commonvoice/data_report.py clean \\
        examples/commonvoice/models/ctc_streaming_3750h_stride3 --epoch 2 \\
        examples/voxpopuli/data/train.json --max-loss 2.5

which writes examples/voxpopuli/data/train-clean.json, for the config's
train_set. The scores are kept, so only the first run scores.
"""

import argparse
import html
import itertools
import json
import os
import random

import numpy as np
import torch
import tqdm
from check_clips import score
from rapidfuzz.distance import Levenshtein
from torch.nn import functional as F

import speech
from speech.utils.dataset import read_data_json

CONFIG = "examples/commonvoice/ctc_streaming_3750h_stride3_config.json"
BATCH = 16
# The bands of the highest losses per character that clean shows clips of,
# in percentiles of the clips that fit.
BANDS = [(99, 100), (98, 99), (97, 98), (95, 97), (90, 95)]


def load(model_dir, epoch):
    """The model of the epoch, or with "best" or "last" that one, on the GPU."""
    if epoch in ("best", "last"):
        model, preproc = speech.load(model_dir, tag="best" if epoch == "best" else None)
    else:
        model, preproc = speech.load(model_dir, tag=None, epoch=int(epoch))
    model.to(speech.best_device())
    model.set_eval()
    return model, preproc


def short(source):
    """A dataset's name, such as librispeech/train-clean-100."""
    parts = os.path.normpath(source).split(os.sep)
    dataset = parts[1] if parts[0] == "examples" and len(parts) > 2 else parts[-2]
    return f"{dataset}/{os.path.splitext(parts[-1])[0]}"


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
    width = max(len(short(s)) for s in sets)
    print(
        f"\n{'CER / WER':{width}}  " + "  ".join(f"{'epoch ' + e:>13}" for e in epochs)
    )
    for path in sets:
        cells = [f"{table[path, e][0]:.3f} / {table[path, e][1]:.3f}" for e in epochs]
        print(f"{short(path):{width}}  " + "  ".join(f"{c:>13}" for c in cells))


def can_fit(frames, labels):
    """Whether CTC can align the labels to the frames, repeats need a blank."""
    repeats = sum(a == b for a, b in itertools.pairwise(labels))
    return frames >= len(labels) + repeats


def fits(model, inputs, labels):
    """Whether CTC can align the labels to the model's output frames."""
    frames = int(model.encoded_lengths(torch.tensor([inputs.shape[0]]))[0])
    return can_fit(frames, labels)


def percentile(rows, key, q):
    return float(np.percentile([r[key] for r in rows], q)) if rows else float("nan")


def clip_row(r):
    """A row of a listening page: the clip, its transcript and what was heard."""
    note = "" if r["fits"] else " <b>does not fit</b>"
    extra = f"<br>peak {r['peak dBFS']:.0f} dBFS" if "peak dBFS" in r else ""
    return (
        f"<tr><td><audio controls preload='none' src='file://{html.escape(os.path.abspath(r['audio']))}'></audio></td>"
        f"<td>{html.escape(r['text'])}<br><i>heard: {html.escape(r['hyp'])}</i></td>"
        f"<td class='m'>loss/char {r['loss/char']:.2f}<br>CER {r['CER']:.2f}{note}<br>"
        f"{r['duration']:.1f} s, {len(r['text']) / r['duration']:.0f} chars/s{extra}</td></tr>"
    )


def write_page(path, title, intro, sections):
    """A listening page of sections, each a heading and its clips."""
    with open(path, "w") as f:
        f.write(
            f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>"
        )
        f.write(
            "<style>body{font:14px system-ui;margin:24px;max-width:1100px}"
            "td{padding:4px 8px;vertical-align:top;border-bottom:1px solid #ddd}"
            "audio{width:260px}.m{color:#666;white-space:nowrap}</style>"
        )
        f.write(f"<h1>{html.escape(title)}</h1><p>{intro}</p>")
        for heading, rows in sections:
            f.write(f"<h2>{html.escape(heading)}</h2><table>")
            f.write("".join(clip_row(r) for r in rows))
            f.write("</table>")


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
    head = "source                           clips  dur s  chars/s  loss/char p50 p90  CER p50  CER>0.8  no fit  silent>10%"
    print(head)
    for source, rows in by_source.items():
        fit = [r for r in rows if r["fits"]]
        print(
            f"{short(source)[:31]:31}  {len(rows):6}  {percentile(rows, 'duration', 50):5.1f}"
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
    print(f"{'source':31}  " + "  ".join(f"{l:>6}" for l in labels))
    for source, rows in by_source.items():
        counts, _ = np.histogram([r["duration"] for r in rows], bins=edges)
        shares = counts / max(len(rows), 1)
        print(f"{short(source)[:31]:31}  " + "  ".join(f"{s:6.1%}" for s in shares))

    page = os.path.join(args.out, "report.html")
    sections = []
    for source, rows in by_source.items():
        worst = sorted((r for r in rows if r["fits"]), key=lambda r: -r["loss/char"])
        no_fit = [r for r in rows if not r["fits"]]
        sections.append((source, worst[: args.worst] + no_fit[: args.worst]))
    write_page(
        page,
        "Worst clips of each training source",
        f"{html.escape(args.model)}, epoch {html.escape(str(args.epoch))}, by loss per "
        "character. Listen for transcripts that do not match the audio, cuts in the "
        "middle of words, other languages and silence: those are data to drop. Hard "
        "but right clips are data to keep.",
        sections,
    )
    print(f"\nWrote {os.path.join(args.out, 'scores.jsonl')} and {page}")


class Clips(torch.utils.data.Dataset):
    """A dataset's clips as model inputs and labels, read in parallel."""

    def __init__(self, data, preproc):
        self.data, self.preproc = data, preproc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        inputs, labels = self.preproc.preprocess(d["audio"], d["text"])
        return i, inputs, labels


def transpose(items):
    return list(zip(*items))


@torch.no_grad()
def score_all(model, preproc, data, workers, batch=32):
    """
    Each clip's CTC loss per character, its greedy transcript and CER, and
    whether it fits, in batches of clips of similar lengths.
    """
    order = sorted(range(len(data)), key=lambda i: data[i]["duration"])
    batches = [order[i : i + batch] for i in range(0, len(order), batch)]
    loader = torch.utils.data.DataLoader(
        Clips(data, preproc),
        batch_sampler=batches,
        num_workers=workers,
        collate_fn=transpose,
    )
    rows = [None] * len(data)
    for ids, inputs, labels in tqdm.tqdm(loader, total=len(batches)):
        x, y, x_lens, y_lens = model.collate(list(inputs), list(labels))
        out = model.forward_impl(x, x_lens)
        log_probs = F.log_softmax(out.float(), dim=2)
        lens = model.encoded_lengths(x_lens)
        losses = F.ctc_loss(
            log_probs.transpose(0, 1),
            y.to(out.device),
            lens,
            y_lens,
            blank=model.blank,
            reduction="none",
            zero_infinity=True,
        ).cpu()
        best = log_probs.argmax(2).cpu().numpy()
        for k, i in enumerate(ids):
            text, frames = data[i]["text"], int(lens[k])
            path = model.max_decode(best[k][:frames].tolist(), model.blank)
            hyp = "".join(preproc.int_to_char[c] for c in path)
            rows[i] = {
                "audio": data[i]["audio"],
                "text": text,
                "hyp": hyp,
                "duration": data[i]["duration"],
                "loss/char": float(losses[k]) / len(text),
                "CER": Levenshtein.distance(hyp, text) / len(text),
                "fits": can_fit(frames, labels[k]),
            }
    return rows


def run_clean(args, config):
    data = read_data_json(args.dataset)
    stem = os.path.splitext(args.dataset)[0]
    scores = f"{stem}-scores.jsonl"
    if os.path.exists(scores):
        with open(scores) as fid:
            rows = [json.loads(line) for line in fid]
        if [r["audio"] for r in rows] != [d["audio"] for d in data]:
            raise SystemExit(f"{scores} is of other clips, delete it to score again.")
        print(f"Read the scores of {len(rows)} clips from {scores}")
    else:
        model, preproc = load(args.model, args.epoch)
        rows = score_all(model, preproc, data, args.workers)
        with open(scores + ".tmp", "w") as fid:
            fid.writelines(json.dumps(r) + "\n" for r in rows)
        os.replace(scores + ".tmp", scores)
        print(f"Wrote {scores}")

    fit = sorted((r for r in rows if r["fits"]), key=lambda r: r["loss/char"])
    no_fit = [r for r in rows if not r["fits"]]
    hours = sum(r["duration"] for r in rows) / 3600
    print(
        f"\n{len(rows)} clips, {hours:.1f} hours. Loss per character of the clips that fit:"
    )
    print("percentile  loss/char  clips above  hours above")
    for q in (50, 90, 95, 97, 98, 99, 99.5):
        value = percentile(fit, "loss/char", q)
        above = [r for r in fit if r["loss/char"] > value]
        print(
            f"{q:10}  {value:9.2f}  {len(above):11}  {sum(r['duration'] for r in above) / 3600:11.1f}"
        )
    print(
        f"Cannot fit: {len(no_fit)} clips, {sum(r['duration'] for r in no_fit) / 3600:.1f} hours"
    )

    rng = random.Random(0)
    sections = []
    for lo, hi in BANDS:
        band = fit[int(len(fit) * lo / 100) : int(len(fit) * hi / 100)]
        if not band:
            continue
        sample = sorted(
            rng.sample(band, min(args.listen, len(band))), key=lambda r: -r["loss/char"]
        )
        heading = (
            f"Percentiles {lo} to {hi}: loss/char {band[0]['loss/char']:.2f} "
            f"to {band[-1]['loss/char']:.2f}"
        )
        sections.append((heading, sample))
    if no_fit:
        sections.append(("Cannot fit", no_fit[: args.listen]))
    page = f"{stem}-listen.html"
    write_page(
        page,
        f"Clips of {args.dataset} by loss",
        "Random clips of each band of the highest losses per character, highest "
        "first. Find the band where the transcripts start to match the audio, "
        "and give its lowest loss as --max-loss.",
        sections,
    )
    print(f"Wrote {page}")

    if args.max_loss is not None:
        keep = [
            d
            for d, r in zip(data, rows)
            if r["fits"] and r["loss/char"] <= args.max_loss
        ]
        out = args.write or f"{stem}-clean.json"
        with open(out, "w") as fid:
            fid.writelines(json.dumps(d) + "\n" for d in keep)
        dropped = len(data) - len(keep)
        dropped_hours = hours - sum(d["duration"] for d in keep) / 3600
        print(
            f"\nWrote {out}: {len(keep)} clips, dropped {dropped} "
            f"({dropped / len(data):.1%}) of {dropped_hours:.1f} hours "
            f"with loss/char over {args.max_loss} or that cannot fit"
        )


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
    cl = commands.add_parser(
        "clean", help="Scores every clip of a dataset and drops the mislabeled."
    )
    cl.add_argument("model", help="The run's model directory.")
    cl.add_argument("dataset", help="The dataset json to clean.")
    cl.add_argument("--epoch", default="best", help="Epoch from 0, or best or last.")
    cl.add_argument(
        "--max-loss",
        type=float,
        help="Drop the clips with a higher loss per character, and write the rest.",
    )
    cl.add_argument(
        "--write", help="The cleaned dataset json, by default <dataset>-clean.json."
    )
    cl.add_argument(
        "--listen", type=int, default=12, help="Clips to list from each band."
    )
    cl.add_argument(
        "--workers", type=int, default=8, help="Processes reading the clips."
    )
    args = parser.parse_args()
    with open(args.config) as fid:
        config = json.load(fid)
    {"eval": run_eval, "score": run_score, "clean": run_clean}[args.command](
        args, config
    )


if __name__ == "__main__":
    main()
