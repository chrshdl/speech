"""
Downloads the Speech Commands dataset by Pete Warden, version 0.02, and
writes its train, dev and test sets: about 106,000 one second clips of 35
words, such as "left", "right", "yes" and "stop", each said alone by one
of 2,618 speakers. It is released under CC BY 4.0.
"""

import argparse
import collections
import json
import os
import tarfile
import urllib.request

import tqdm

from speech.utils import wave

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
# The noise for augmentation, see examples/noise.
NOISE = "_background_noise_"
# The official splits list the clips of the dev and test sets, so that no
# speaker is in two of them. The rest are for training.
SPLITS = {"dev": "validation_list.txt", "test": "testing_list.txt"}


def extract(archive, out_dir):
    """
    Extracts the clips and the split lists from the archive, a local file
    or else streamed from URL, so it is never stored.
    """
    os.makedirs(out_dir, exist_ok=True)
    with (
        open(archive, "rb") if archive else urllib.request.urlopen(URL) as source,
        tarfile.open(fileobj=source, mode="r|gz") as tf,
        tqdm.tqdm(unit=" files") as progress,
    ):
        for member in tf:
            parts = os.path.normpath(member.name).split(os.sep)
            clip = member.name.endswith(".wav") and NOISE not in parts
            if clip or parts[-1] in SPLITS.values():
                tf.extract(member, out_dir, filter="data")
                progress.update()


def write_sets(out_dir):
    """
    Writes train.json, dev.json and test.json to out_dir, one clip per line
    with its word as the text, and returns the clips and hours of each, and
    the clips of each word.
    """
    split_of = {}
    for split, listing in SPLITS.items():
        with open(os.path.join(out_dir, listing)) as fid:
            split_of.update((line.strip(), split) for line in fid if line.strip())

    clips = collections.defaultdict(list)
    words = collections.Counter()
    for word in sorted(os.listdir(out_dir)):
        folder = os.path.join(out_dir, word)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".wav"):
                continue
            path = os.path.join(folder, name)
            split = split_of.get(f"{word}/{name}", "train")
            clips[split].append(
                {"text": word, "duration": wave.wav_duration(path), "audio": path}
            )
            words[word] += 1

    summary = {}
    for split in ("train", "dev", "test"):
        with open(os.path.join(out_dir, f"{split}.json"), "w") as fid:
            fid.writelines(json.dumps(datum) + "\n" for datum in clips[split])
        hours = sum(d["duration"] for d in clips[split]) / 3600
        summary[split] = (len(clips[split]), hours)
    return summary, words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "out_dir",
        help="Where to save the clips and the sets, such as "
        "examples/speech_commands/data.",
    )
    parser.add_argument(
        "--archive",
        help="The downloaded speech_commands_v0.02.tar.gz, instead of streaming it.",
    )
    args = parser.parse_args()
    extract(args.archive, args.out_dir)
    summary, words = write_sets(args.out_dir)
    for split, (count, hours) in summary.items():
        print(f"{split}: {count} clips, {hours:.1f} hours")
    print("clips of each word:", dict(words.most_common()))
