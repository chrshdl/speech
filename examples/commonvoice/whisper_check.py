"""
Checks a dataset's transcripts with Whisper, a speech recognizer trained on
other data than these models, for a person to review the clips in which
Whisper does not hear the transcript: mislabeled, cut off, or hard to
understand even for a person, such as many Speech Commands clips whose word
is cut off. Run it from the repository root on the machine with the data.

check: Whisper's transcript of each clip and its character error rate
against the clip's, written to <dataset>-whisper.jsonl, a run started again
continuing where it stopped, and how many clips fall in each band of CER.
Whisper's transcript is normalized like the datasets', its numbers spelled
out. Whisper sometimes makes up a short phrase for silence or noise, such
as "thank you", which counts as not hearing the transcript:

    uv run --with transformers --with accelerate \\
        examples/commonvoice/whisper_check.py check \\
        examples/speech_commands/data/train.json examples/voxpopuli/data/train.json

review: serves the clips whose CER is over --min-cer on 127.0.0.1, one at a
time, to listen to and mark OK, Whisper misheard it and the clip stays, or
NOK, Whisper is right and the clip is left out of training. The verdicts
are saved at once to <dataset>-review.json. train.py does the same after
each epoch, with "review" in its config's data, see speech/utils/review.py:

    uv run examples/commonvoice/whisper_check.py review \\
        examples/speech_commands/data/train.json

write: the dataset without the clips reviewed NOK, and with --unreviewed
drop also without those not reviewed yet, to <dataset>-reviewed.json.
"""

import argparse
import json
import os

import numpy as np
import torch
import tqdm
from rapidfuzz.distance import Levenshtein

from speech.utils import review, wave
from speech.utils.dataset import read_data_json
from speech.utils.prepare import ALLOWED, normalize

MODEL = "openai/whisper-large-v3-turbo"
BANDS = [(0.5, None), (0.25, 0.5), (0.15, 0.25), (0.0, 0.15)]
ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = [
    "_",
    "_",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def spell(n):
    """A number from 0 to 999,999 in words, such as twenty one."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        rest = "" if n % 100 == 0 else " " + spell(n % 100)
        return ONES[n // 100] + " hundred" + rest
    rest = "" if n % 1000 == 0 else " " + spell(n % 1000)
    return spell(n // 1000) + " thousand" + rest


def to_text(transcript):
    """Whisper's transcript in the datasets' characters, numbers spelled out."""
    words = []
    for word in transcript.replace(",", "").split():
        digits = word.strip(".?!%")
        if digits.isdigit() and len(digits) <= 6:
            word = word.replace(digits, " " + spell(int(digits)) + " ")
        words.append(word)
    text = " ".join(words)
    return normalize(text) or " ".join(" ".join(ALLOWED.findall(text.lower())).split())


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def check(dataset, model, batch_size):
    """Appends Whisper's transcript of each clip not checked yet."""
    from transformers import pipeline

    data = read_data_json(dataset)
    out_path = review.files(dataset)[0]
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fid:
            done = {json.loads(line)["audio"] for line in fid}
    todo = [d for d in data if d["audio"] not in done]
    if todo:
        where = device()
        recognizer = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=where,
            dtype=torch.float32 if where == "cpu" else torch.float16,
        )
        with open(out_path, "a") as out:
            for i in tqdm.tqdm(range(0, len(todo), batch_size), desc=dataset):
                batch = todo[i : i + batch_size]
                inputs = []
                for d in batch:
                    audio, rate = wave.array_from_wave(d["audio"])
                    inputs.append(
                        {"raw": audio.astype(np.float32) / 32768, "sampling_rate": rate}
                    )
                results = recognizer(
                    inputs,
                    batch_size=batch_size,
                    generate_kwargs={"language": "english", "task": "transcribe"},
                )
                for d, result in zip(batch, results):
                    heard = to_text(result["text"])
                    row = {
                        "audio": d["audio"],
                        "text": d["text"],
                        "duration": d["duration"],
                        "whisper": result["text"].strip(),
                        "hyp": heard,
                        "CER": Levenshtein.distance(heard, d["text"]) / len(d["text"]),
                    }
                    out.write(json.dumps(row) + "\n")
                out.flush()

    with open(out_path) as fid:
        rows = list(map(json.loads, fid))
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"\n{dataset}: {len(rows)} clips, {hours:.1f} hours. Whisper's CER:")
    print("CER          clips  share  hours")
    for lo, hi in BANDS:
        if hi is None:
            band, name = [r for r in rows if r["CER"] > lo], f"over {lo}"
        elif lo == 0.0:
            band, name = [r for r in rows if r["CER"] <= hi], f"up to {hi}"
        else:
            band, name = [r for r in rows if lo < r["CER"] <= hi], f"{lo} to {hi}"
        print(
            f"{name:11}  {len(band):5}  {len(band) / len(rows):5.1%}"
            f"  {sum(r['duration'] for r in band) / 3600:5.1f}"
        )
    print(f"of those, no speech heard: {sum(not r['hyp'] for r in rows)} clips")


def write(dataset, unreviewed, min_cer):
    """The dataset without the clips reviewed NOK, and maybe the unreviewed."""
    data = read_data_json(dataset)
    drop = review.excluded([dataset])
    pending = [r["audio"] for r in review.queue([dataset], min_cer) if not r["verdict"]]
    if unreviewed == "drop":
        drop |= set(pending)
    keep = [d for d in data if d["audio"] not in drop]
    out = f"{os.path.splitext(dataset)[0]}-reviewed.json"
    with open(out, "w") as fid:
        fid.writelines(json.dumps(d) + "\n" for d in keep)
    print(
        f"Wrote {out}: {len(keep)} of {len(data)} clips, dropped {len(data) - len(keep)}; "
        f"{len(pending)} to review were not reviewed yet and are "
        f"{'dropped' if unreviewed == 'drop' else 'kept'}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    ch = commands.add_parser("check", help="Transcribes each clip with Whisper.")
    ch.add_argument("datasets", nargs="+", help="The dataset jsons to check.")
    ch.add_argument("--model", default=MODEL, help="The Whisper model.")
    ch.add_argument("--batch-size", type=int, default=16)
    rv = commands.add_parser("review", help="Serves the review of the mismatches.")
    rv.add_argument("datasets", nargs="+", help="The checked dataset jsons.")
    rv.add_argument("--min-cer", type=float, default=review.MIN_CER)
    rv.add_argument("--port", type=int, default=8765)
    wr = commands.add_parser("write", help="Writes the dataset without NOK clips.")
    wr.add_argument("dataset", help="The reviewed dataset json.")
    wr.add_argument("--unreviewed", choices=["keep", "drop"], default="keep")
    wr.add_argument("--min-cer", type=float, default=review.MIN_CER)
    args = parser.parse_args()
    if args.command == "check":
        for dataset in args.datasets:
            check(dataset, args.model, args.batch_size)
    elif args.command == "review":
        review.serve(args.datasets, args.port, args.min_cer)
    else:
        write(args.dataset, args.unreviewed, args.min_cer)


if __name__ == "__main__":
    main()
