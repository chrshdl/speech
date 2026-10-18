"""
Writes evaluation sets of SLURP, English requests to a voice assistant,
such as "turn off the dining room" or "what's the weather in london",
recorded by 177 speakers each with a headset and with distant microphones
in a room. Its requests are short, as asked of the device, while the
training sets are read books and speeches.

For each of SLURP's devel and test sets it writes a random sample of the
sentences, each with one recording of a microphone, its close-talk headset
or a distant one, as 16 kHz FLAC with a dataset json, and the text of all
the set's sentences, for the perplexity of a language model:

    examples/slurp/data/slurp-test-headset.json   400 clips
    examples/slurp/data/slurp-test-far.json       400 clips
    examples/slurp/data/slurp-devel-headset.json  300 clips
    examples/slurp/data/slurp-devel-far.json      300 clips
    examples/slurp/data/slurp-{devel,test}-text.json

The sentences come from SLURP's repository (CC BY 4.0), the recordings
from the copy of its audio on Hugging Face, marcel-gohsen/slurp, about
1.2 GB for these two sets (CC BY-NC 4.0: for evaluation, not to train a
model released for others). Sentences that normalize() cannot write out,
such as with digits or e-mail addresses, are left out.
"""

import argparse
import collections
import json
import os
import random
import urllib.request

import pyarrow.parquet as pq
import soundfile

from speech.utils.prepare import normalize

TEXT = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/{}.jsonl"
AUDIO = "https://huggingface.co/datasets/marcel-gohsen/slurp/resolve/main/data/{}"
FILES = {
    "devel": ["devel-00000-of-00001.parquet"],
    "test": ["test-00000-of-00002.parquet", "test-00001-of-00002.parquet"],
}
SIZES = {"devel": 300, "test": 400}


def download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def sentences(split, data_dir):
    """Each sentence's normalized text, and its sentence of each recording."""
    texts, by_file = {}, {}
    path = download(TEXT.format(split), os.path.join(data_dir, f"{split}.jsonl"))
    with open(path) as lines:
        for line in lines:
            row = json.loads(line)
            text = normalize(row["sentence"])
            if text is None:
                continue
            texts[row["slurp_id"]] = text
            for recording in row["recordings"]:
                by_file[recording["file"]] = row["slurp_id"]
    return texts, by_file


def prepare(data_dir, split, mic, texts, by_file):
    """Writes the sample of one recording per sentence from the microphone."""
    recordings = collections.defaultdict(list)
    for name in FILES[split]:
        parquet = download(AUDIO.format(name), os.path.join(data_dir, name))
        for audio in (
            pq.read_table(parquet, columns=["audio"]).column("audio").to_pylist()
        ):
            path = audio["path"]
            if ("headset" in path) == (mic == "headset") and path in by_file:
                recordings[by_file[path]].append(audio)
    rng = random.Random(0)
    ids = sorted(recordings)
    rng.shuffle(ids)
    clip_dir = os.path.join(data_dir, f"{split}-{mic}")
    os.makedirs(clip_dir, exist_ok=True)
    with open(os.path.join(data_dir, f"slurp-{split}-{mic}.json"), "w") as out:
        for slurp_id in ids[: SIZES[split]]:
            audio = rng.choice(recordings[slurp_id])
            target = os.path.join(clip_dir, audio["path"])
            with open(target, "wb") as f:
                f.write(audio["bytes"])
            info = soundfile.info(target)
            row = {
                "text": texts[slurp_id],
                "duration": round(info.frames / info.samplerate, 3),
                "audio": target,
            }
            out.write(json.dumps(row) + "\n")
    print(f"{split} {mic}: {min(SIZES[split], len(ids))} of {len(ids)} sentences")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", default="examples/slurp/data", help="Where to write the sets."
    )
    args = parser.parse_args()
    os.makedirs(args.data, exist_ok=True)
    for split in FILES:
        texts, by_file = sentences(split, args.data)
        for mic in ("headset", "far"):
            prepare(args.data, split, mic, texts, by_file)
        with open(os.path.join(args.data, f"slurp-{split}-text.json"), "w") as out:
            out.writelines(json.dumps({"text": t}) + "\n" for t in texts.values())


if __name__ == "__main__":
    main()
