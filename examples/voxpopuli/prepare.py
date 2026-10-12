"""
Downloads the transcribed English speech of VoxPopuli, 543 hours of
European Parliament speeches by 1,313 speakers with many accents, released
under CC0, and writes its train, dev and test sets with the clips as
16 kHz FLAC.

The clips and transcripts come from the Parquet files of the copy on
Hugging Face, facebook/voxpopuli, about 99 GB, which are downloaded one at
a time and deleted once their clips are written. An interrupted run
continues where it stopped.
"""

import argparse
import collections
import io
import json
import os
import time
import urllib.error
import urllib.request

import pyarrow.parquet as pq
import soundfile
import tqdm

from speech.utils.prepare import normalize, to_flac

URL = "https://huggingface.co/datasets/facebook/voxpopuli/resolve/main/en/"
# The Parquet files of each set, whose official splits keep each speaker
# in one of them.
FILES = {
    "train": [f"train-{i:05d}-of-00030.parquet" for i in range(30)],
    "dev": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}
COLUMNS = ["audio_id", "audio", "normalized_text", "is_gold_transcript"]
# Longer clips are dropped, 1% of them and 3% of the hours. A few last over
# a minute.
MAX_SECONDS = 30.0
CHUNK = 1 << 20


def download(name, path, retries=10):
    """
    Downloads the file to path, through a partial file that a later call
    continues from, retrying after a dropped connection.
    """
    part = path + ".part"
    for attempt in range(retries):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        request = urllib.request.Request(
            URL + name, headers={"Range": f"bytes={done}-"}
        )
        try:
            with urllib.request.urlopen(request) as response, open(part, "ab") as fid:
                # A server that ignores the range sends the whole file.
                if done and response.status != 206:
                    fid.truncate(0)
                total = done + int(response.headers.get("Content-Length", 0))
                with tqdm.tqdm(
                    total=total, initial=done, unit="B", unit_scale=True, desc=name
                ) as progress:
                    while chunk := response.read(CHUNK):
                        fid.write(chunk)
                        progress.update(len(chunk))
            os.replace(part, path)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            if attempt == retries - 1:
                raise
            print(f"{name}: {error}, retrying")
            time.sleep(min(2**attempt, 60))


def extract(parquet, clip_dir, gold_only=False, max_seconds=MAX_SECONDS):
    """
    Writes the clips of a Parquet file to clip_dir as FLAC, and returns
    their data, one dict per clip, and the counts of the clips left out by
    reason. Clips written before are kept.
    """
    data = []
    dropped = collections.Counter()
    table = pq.ParquetFile(parquet)
    for group in range(table.metadata.num_row_groups):
        for row in table.read_row_group(group, columns=COLUMNS).to_pylist():
            if gold_only and not row["is_gold_transcript"]:
                dropped["not gold"] += 1
                continue
            text = normalize(row["normalized_text"] or "")
            if text is None:
                dropped["text"] += 1
                continue
            path = os.path.join(clip_dir, row["audio_id"] + ".flac")
            try:
                if os.path.exists(path):
                    info = soundfile.info(path)
                    duration = info.frames / info.samplerate
                else:
                    audio, rate = soundfile.read(
                        io.BytesIO(row["audio"]["bytes"]), dtype="float32"
                    )
                    duration = len(audio) / rate
                    if duration <= max_seconds:
                        duration = to_flac(audio, rate, path)
            except (RuntimeError, soundfile.LibsndfileError):
                dropped["audio"] += 1
                continue
            if duration > max_seconds:
                dropped["too long"] += 1
                continue
            data.append({"text": text, "duration": duration, "audio": path})
    return data, dropped


def prepare(
    out_dir, gold_only=False, train_files=None, keep=False, max_seconds=MAX_SECONDS
):
    """
    Downloads and extracts each Parquet file, or the first train_files of
    the train set, and writes <out_dir>/<set>.json. Returns the clips,
    hours and dropped clips of each set.
    """
    clip_dir = os.path.join(out_dir, "clips")
    done_dir = os.path.join(out_dir, "done")
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    summary = {}
    for split, names in FILES.items():
        if split == "train" and train_files is not None:
            names = names[:train_files]
        data, dropped = [], collections.Counter()
        for name in names:
            # Each finished file leaves its clips' data, so a later run
            # neither downloads nor extracts it again.
            done = os.path.join(done_dir, name + ".json")
            if not os.path.exists(done):
                parquet = os.path.join(out_dir, name)
                if not os.path.exists(parquet):
                    download(name, parquet)
                file_data, file_dropped = extract(
                    parquet, clip_dir, gold_only, max_seconds
                )
                with open(done + ".tmp", "w") as fid:
                    json.dump({"data": file_data, "dropped": file_dropped}, fid)
                os.replace(done + ".tmp", done)
                if not keep:
                    os.remove(parquet)
            with open(done) as fid:
                finished = json.load(fid)
            data += finished["data"]
            dropped.update(finished["dropped"])
        with open(os.path.join(out_dir, f"{split}.json"), "w") as fid:
            fid.writelines(json.dumps(datum) + "\n" for datum in data)
        hours = sum(d["duration"] for d in data) / 3600
        summary[split] = {"clips": len(data), "hours": hours, "dropped": dict(dropped)}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "out_dir",
        help="Where to save the clips and the sets, such as examples/voxpopuli/data.",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Leave out the clips whose transcript VoxPopuli does not mark as gold.",
    )
    parser.add_argument(
        "--train-files",
        type=int,
        help="Only the first of the 30 train files, about 18 hours each, such as "
        "1 to try it.",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=MAX_SECONDS, help="Drop longer clips."
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep the Parquet files instead of deleting each once extracted.",
    )
    args = parser.parse_args()
    summary = prepare(
        args.out_dir,
        args.gold_only,
        args.train_files,
        args.keep_downloads,
        args.max_seconds,
    )
    for split, s in summary.items():
        print(
            f"{split}: {s['clips']} clips, {s['hours']:.1f} hours, dropped {s['dropped']}"
        )
