"""
Prepares Mozilla Common Voice for training. Normalizes the transcripts to
the characters of the models, lowercase letters, the apostrophe and the
space, converts the MP3 clips to 16 kHz FLAC and writes a dataset json
file for each split.
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import re
import unicodedata

import numpy as np
import soundfile
import tqdm
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
ALLOWED = re.compile(r"[a-z' ]+")
APOSTROPHES = re.compile(r"[’‘`´ʼ]")
DASHES = re.compile(r"[-‐‑‒–—]")
PUNCTUATION = re.compile(r"[.,!?;:\"“”„«»()\[\]{}…¿¡/*&]")

# Common Voice's TSV files hold sentences with quotes in them.
csv.field_size_limit(2**31 - 1)


def normalize(sentence):
    """
    Returns the sentence as lowercase letters, apostrophes and single
    spaces, or None if it has other characters, such as digits, that
    would need a guess to write out.
    """
    # Split accented letters and ligatures, then drop the accents.
    text = unicodedata.normalize("NFKD", sentence)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = APOSTROPHES.sub("'", text)
    text = DASHES.sub(" ", text)
    text = PUNCTUATION.sub(" ", text)
    words = []
    for word in text.split():
        # Keep apostrophes inside words and after a plural s, as in
        # "don't" and "the students' books", and drop quoting ones.
        word = word.lstrip("'")
        if not word.endswith("s'"):
            word = word.rstrip("'")
        if word:
            words.append(word)
    text = " ".join(words)
    if not text or not ALLOWED.fullmatch(text):
        return None
    return text


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as fid:
        return list(csv.DictReader(fid, delimiter="\t", quoting=csv.QUOTE_NONE))


def convert(job):
    """
    Converts a clip to 16 kHz mono FLAC and returns its duration in
    seconds, or None if it cannot be read. Converted clips are kept, so
    an interrupted run can continue.
    """
    source, target = job
    try:
        if os.path.exists(target):
            info = soundfile.info(target)
            return info.frames / info.samplerate
        audio, rate = soundfile.read(source, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            g = math.gcd(rate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, rate // g)
        audio = np.clip(audio, -1, 1)
        tmp = target + ".tmp"
        soundfile.write(tmp, audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        os.replace(tmp, target)
        return len(audio) / SAMPLE_RATE
    except (RuntimeError, soundfile.LibsndfileError):
        return None


def split_rows(cv_dir, train_from):
    """
    Returns the rows of each split. Training on all validated clips gives
    much more data than the official train split. The clips of dev and
    test speakers, and the dev and test sentences, are left out of it.
    """
    dev = read_tsv(os.path.join(cv_dir, "dev.tsv"))
    test = read_tsv(os.path.join(cv_dir, "test.tsv"))
    if train_from == "train":
        train = read_tsv(os.path.join(cv_dir, "train.tsv"))
    else:
        held_out = dev + test
        speakers = {r["client_id"] for r in held_out}
        sentences = {normalize(r["sentence"]) for r in held_out}
        paths = {r["path"] for r in held_out}
        train = [
            r
            for r in read_tsv(os.path.join(cv_dir, "validated.tsv"))
            if r["client_id"] not in speakers
            and r["path"] not in paths
            and normalize(r["sentence"]) not in sentences
        ]
    return {"train": train, "dev": dev, "test": test}


def prepare(cv_dir, out_dir, train_from="validated", workers=None):
    """
    Converts the clips of each split and writes <out_dir>/<split>.json.
    Returns a summary per split.
    """
    clip_dir = os.path.join(out_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    summary = {}
    with multiprocessing.Pool(workers) as pool:
        for split, rows in split_rows(cv_dir, train_from).items():
            texts, jobs = [], []
            for r in rows:
                text = normalize(r["sentence"])
                if text is None:
                    continue
                name = os.path.splitext(r["path"])[0] + ".flac"
                texts.append(text)
                jobs.append(
                    (
                        os.path.join(cv_dir, "clips", r["path"]),
                        os.path.join(clip_dir, name),
                    )
                )
            durations = list(
                tqdm.tqdm(
                    pool.imap(convert, jobs, chunksize=16), total=len(jobs), desc=split
                )
            )
            kept = 0
            seconds = 0.0
            with open(os.path.join(out_dir, f"{split}.json"), "w") as fid:
                for text, (_, audio), duration in zip(texts, jobs, durations):
                    if duration is None:
                        continue
                    datum = {"text": text, "duration": duration, "audio": audio}
                    json.dump(datum, fid)
                    fid.write("\n")
                    kept += 1
                    seconds += duration
            summary[split] = {
                "clips": kept,
                "hours": seconds / 3600,
                "dropped_text": len(rows) - len(jobs),
                "dropped_audio": len(jobs) - kept,
            }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Common Voice: normalize the text, convert the clips "
        "to 16 kHz FLAC and write train, dev and test dataset json files."
    )
    parser.add_argument(
        "cv_dir",
        help="The extracted Common Voice language directory, which holds clips/ "
        "and the .tsv files.",
    )
    parser.add_argument(
        "out_dir", help="Where to write the FLAC clips and the json files."
    )
    parser.add_argument(
        "--train-from",
        choices=["validated", "train"],
        default="validated",
        help="Train on all validated clips without the dev and test speakers and "
        "sentences, the default, or only on the official train split.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Processes converting audio, by default one per CPU core.",
    )
    args = parser.parse_args()

    for split, s in prepare(
        args.cv_dir, args.out_dir, args.train_from, args.workers
    ).items():
        print(
            f"{split}: {s['clips']} clips, {s['hours']:.1f} h. Dropped "
            f"{s['dropped_text']} for their text, {s['dropped_audio']} for their audio."
        )
