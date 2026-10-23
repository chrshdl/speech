"""
Prepares the Olivia wake word dataset of github.com/mindorii/kws, from
its zip file: recordings of people saying "olivia", most of them once,
at 44.1 kHz. Writes each clip at 16 kHz with where it says the word, as
found by a streaming CTC model, and train, dev and test sets.

Some clips are silent, cut off or say something else. A clip is kept
only when the CTC model hears the word in it, at an alignment score of
at least --min-score.
"""

import argparse
import collections
import io
import json
import math
import os
import zipfile

import numpy as np
import scipy.signal
import soundfile
import tqdm

from speech import streaming, wake_word
from speech.features import log_specgram

SAMPLE_RATE = 16000
WORD = "olivia"
# The zip's folders of each set. Its olivia_train_noise holds the
# training clips with noise added, which train.py adds itself.
SETS = {"train": "olivia_train/", "test": "olivia_test/"}
# The lowest mean log probability per label of the word's alignment,
# relative to the model's best path. Lower are mostly silent or broken
# clips, which the model transcribes as nothing.
MIN_SCORE = -2.5
# The training clips longer than this often say the word several times,
# whose other repetitions train.py would take for other speech.
MAX_TRAIN_SECONDS = 3.0
# The share of the training clips, the last by number, held out as dev.
# The clips have no speaker ids, and neighboring numbers likely share a
# speaker, so a block is held out rather than a random sample.
DEV_SHARE = 0.1


def read(data):
    """The clip's 16 bit samples at SAMPLE_RATE, mixed to mono."""
    audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE // g, rate // g)
    return np.clip(np.round(audio * 32768), -32768, 32767).astype(np.int16)


def align(model, labels, audio):
    """
    Returns the alignment score of the word in the audio, and the seconds
    where it starts and ends, or None if the audio is too short.
    """
    aligned = wake_word.align(
        model.probs(log_specgram(audio, SAMPLE_RATE)), labels, model.blank
    )
    if aligned is None:
        return None
    score, start, end = aligned
    return score, start * model.frame_seconds, end * model.frame_seconds


def main(args):
    model = streaming.load(args.asr_model)
    labels = np.array([model.char_to_int[c] for c in WORD])
    clips_dir = os.path.join(args.out_dir, "clips")
    sets = collections.defaultdict(list)
    dropped = collections.Counter()
    with zipfile.ZipFile(args.zip) as archive:
        names = [
            (name, split)
            for name in sorted(archive.namelist())
            for split, folder in SETS.items()
            if name.startswith(folder) and name.endswith(".wav")
        ]
        for name, split in tqdm.tqdm(names, unit=" clips"):
            audio = read(archive.read(name))
            duration = len(audio) / SAMPLE_RATE
            aligned = align(model, labels, audio) if duration >= 0.1 else None
            if aligned is None:
                dropped[split, "empty"] += 1
                continue
            score, start, end = aligned
            if score < args.min_score:
                dropped[split, "not heard"] += 1
                continue
            if end - start > wake_word.CLIP_SECONDS:
                dropped[split, "longer than a window"] += 1
                continue
            if split == "train" and duration > MAX_TRAIN_SECONDS:
                dropped[split, "long"] += 1
                continue
            path = os.path.join(clips_dir, split, os.path.basename(name))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            soundfile.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
            sets[split].append(
                {
                    "audio": path,
                    "duration": round(duration, 3),
                    "text": WORD,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "score": round(score, 3),
                }
            )

    train = sorted(sets["train"], key=lambda d: int(os.path.basename(d["audio"])[:-4]))
    held_out = round(len(train) * DEV_SHARE)
    sets["train"], sets["dev"] = train[:-held_out], train[-held_out:]
    for split in ("train", "dev", "test"):
        with open(os.path.join(args.out_dir, f"{split}.json"), "w") as fid:
            fid.writelines(json.dumps(datum) + "\n" for datum in sets[split])
        hours = sum(d["duration"] for d in sets[split]) / 3600
        print(f"{split}: {len(sets[split])} clips, {hours:.2f} hours")
    for (split, reason), count in sorted(dropped.items()):
        print(f"Dropped from {split}: {count} {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", help="The dataset's zip file, olivia.zip.")
    parser.add_argument(
        "asr_model",
        help="A streaming CTC model of characters, which finds the word in each "
        "clip, such as model-int8.onnx of the 3,750 hour model.",
    )
    parser.add_argument("--out-dir", default="examples/olivia/data")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    main(parser.parse_args())
