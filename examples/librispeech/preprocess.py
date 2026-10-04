import argparse
import glob
import json
import os

import tqdm

from speech.utils import wave

SETS = ["train-clean-100", "dev-clean"]


def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.trans.txt")
    files = glob.glob(pattern)
    data = {}
    for f in files:
        with open(f) as fid:
            lines = (l.strip().split() for l in fid)
            lines = ((l[0], " ".join(l[1:])) for l in lines)
            data.update(lines)
    return data


def path_from_key(key, prefix, ext):
    dirs = key.split("-")
    dirs[-1] = key
    path = os.path.join(prefix, *dirs)
    return path + os.path.extsep + ext


def clean_text(text):
    return text.strip().lower()


def build_json(path):
    transcripts = load_transcripts(path)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path) + os.path.extsep + "json"
    with open(os.path.join(dirname, basename), "w") as fid:
        for k, t in tqdm.tqdm(transcripts.items()):
            audio_file = path_from_key(k, path, ext="flac")
            dur = wave.wav_duration(audio_file)
            t = clean_text(t)
            datum = {"text": t, "duration": dur, "audio": audio_file}
            json.dump(datum, fid)
            fid.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess librispeech dataset.")

    parser.add_argument(
        "output_directory",
        help="The dataset is saved in <output_directory>/LibriSpeech.",
    )
    parser.add_argument(
        "--sets", nargs="+", default=SETS, help="The subsets to preprocess."
    )
    args = parser.parse_args()

    path = os.path.join(args.output_directory, "LibriSpeech")
    for d in args.sets:
        print(f"Preprocessing {d}")
        build_json(os.path.join(path, d))
