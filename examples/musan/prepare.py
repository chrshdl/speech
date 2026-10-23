"""
Extracts MUSAN (Snyder, Chen and Povey, 2015), a corpus of music, speech
and noise at 16 kHz, and writes a train and a test set of each kind, a
tenth of the recordings, by name, held out for testing. It is released
under CC BY 4.0.
"""

import argparse
import json
import os
import tarfile
import urllib.request
import zlib

import soundfile
import tqdm

URL = "https://www.openslr.org/resources/17/musan.tar.gz"
KINDS = ("music", "speech", "noise")
# One recording in TEST_MODULO, by a hash of its name, is for testing.
TEST_MODULO = 10


def extract(archive, out_dir):
    """
    Extracts the recordings from the archive, a local file or else
    streamed from URL, so it is never stored.
    """
    os.makedirs(out_dir, exist_ok=True)
    with (
        open(archive, "rb") if archive else urllib.request.urlopen(URL) as source,
        tarfile.open(fileobj=source, mode="r|gz") as tf,
        tqdm.tqdm(unit=" files") as progress,
    ):
        for member in tf:
            if member.name.endswith(".wav"):
                tf.extract(member, out_dir, filter="data")
                progress.update()


def write_sets(out_dir):
    """
    Writes <kind>-train.json and <kind>-test.json to out_dir, one
    recording per line, and returns the recordings and hours of each.
    """
    counts = {}
    for kind in KINDS:
        sets = {"train": [], "test": []}
        root = os.path.join(out_dir, "musan", kind)
        for folder, _, files in sorted(os.walk(root)):
            for name in sorted(files):
                if not name.endswith(".wav"):
                    continue
                path = os.path.join(folder, name)
                info = soundfile.info(path)
                split = (
                    "test" if zlib.crc32(name.encode()) % TEST_MODULO == 0 else "train"
                )
                sets[split].append(
                    {"audio": path, "duration": round(info.duration, 3), "text": ""}
                )
        for split, data in sets.items():
            with open(os.path.join(out_dir, f"{kind}-{split}.json"), "w") as fid:
                fid.writelines(json.dumps(d) + "\n" for d in data)
            hours = sum(d["duration"] for d in data) / 3600
            counts[kind, split] = (len(data), hours)
    return counts


def main(args):
    if not args.skip_extract:
        extract(args.archive, args.out_dir)
    for (kind, split), (count, hours) in write_sets(args.out_dir).items():
        print(f"{kind}-{split}: {count} recordings, {hours:.1f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", default="examples/musan/data")
    parser.add_argument(
        "--archive", help="A downloaded musan.tar.gz, by default streamed from URL."
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only write the sets, of recordings extracted before.",
    )
    main(parser.parse_args())
