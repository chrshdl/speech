"""
Downloads the transcripts of The People's Speech, English speech of
public talks, hearings and other recordings, and writes them as text for a
word LM, normalized like the acoustic model's transcripts, one segment per
line, without the sentences of the --exclude sets:

    examples/peoples_speech/data/ps-clean.txt.gz

Only the text column of each Parquet file of the copy on Hugging Face,
MLCommons/peoples_speech, is read, by ranged requests, not its audio. Each
file's text is kept in <data>/<subset>/, so an interrupted run continues
where it stopped. The subsets are those under CC BY 4.0: clean, 804
files, and dirty, 3,140 files of less accurate transcripts.

Hugging Face allows about 200 requests to huggingface.co per 5 minutes,
so each file makes one, for its address on the storage servers, paced
across the workers, and its ranged reads go to that address.
"""

import argparse
import concurrent.futures
import gzip
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pyarrow.parquet as pq
import tqdm

from speech.utils.prepare import normalize

URL = (
    "https://huggingface.co/datasets/MLCommons/peoples_speech/resolve/~parquet/"
    "{}/train/{:04d}.parquet"
)
FILES = {"clean": 804, "dirty": 3140}
# At most one request to huggingface.co per PACE seconds, 187 per 5 minutes.
PACE = 1.6
RETRIES = 6
TIMEOUT = 60


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


NO_REDIRECT = urllib.request.build_opener(NoRedirect)


class Pacer:
    def __init__(self, seconds):
        """Spaces calls of wait() from any thread at least seconds apart."""
        self.seconds = seconds
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self, extra=0.0):
        with self.lock:
            pause = self.last + self.seconds - time.time()
            if pause > 0:
                time.sleep(pause)
            self.last = time.time() + extra


class RemoteFile(io.RawIOBase):
    def __init__(self, url):
        """A file on a server, read by ranged requests."""
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        start = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = start[whence] + offset
        return self.position

    def readinto(self, buffer):
        end = min(self.position + len(buffer), self.size)
        if end <= self.position:
            return 0
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end - 1}"}
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def storage_url(url, pacer):
    """Where huggingface.co redirects the url to, waiting when rate limited."""
    while True:
        pacer.wait()
        try:
            NO_REDIRECT.open(
                urllib.request.Request(url, method="HEAD"), timeout=TIMEOUT
            )
        except urllib.error.HTTPError as error:
            if error.code in (301, 302, 307, 308):
                return error.headers["Location"]
            if error.code != 429:
                raise
            retry = int(error.headers.get("Retry-After", 60)) + 5
            print(f"Rate limited, waiting {retry} s")
            pacer.wait(extra=retry)
            continue
        raise RuntimeError(f"{url} was not redirected")


def fetch(data_dir, subset, number, pacer):
    """
    Writes the normalized text of the subset's file number to
    <data_dir>/<subset>/<number>.txt.gz, unless it was written before.
    """
    target = os.path.join(data_dir, subset, f"{number:04d}.txt.gz")
    if os.path.exists(target):
        return target
    for attempt in range(RETRIES):
        try:
            remote = RemoteFile(storage_url(URL.format(subset, number), pacer))
            with io.BufferedReader(remote, buffer_size=1 << 20) as f:
                table = pq.ParquetFile(f).read(columns=["text"])
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            if attempt == RETRIES - 1:
                raise
            print(f"{subset}/{number:04d}: {error}, retrying")
            time.sleep(30 * (attempt + 1))
    lines = [t for t in map(normalize, table.column("text").to_pylist()) if t]
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with gzip.open(target + ".tmp", "wt") as f:
        f.writelines(line + "\n" for line in lines)
    os.replace(target + ".tmp", target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data",
        default="examples/peoples_speech/data",
        help="Where to write the text.",
    )
    parser.add_argument(
        "--subsets",
        default="clean",
        help="The subsets, comma separated, clean or clean,dirty.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Dataset json files whose sentences to leave out, such as the dev "
        "and test sets the LM is measured on.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="The files to read at once."
    )
    args = parser.parse_args()
    subsets = args.subsets.split(",")
    held_out = set()
    for path in args.exclude:
        with open(path) as lines:
            held_out.update(json.loads(line)["text"] for line in lines)

    pacer = Pacer(PACE)
    jobs = [(s, n) for s in subsets for n in range(FILES[s])]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        parts = list(
            tqdm.tqdm(
                pool.map(lambda job: fetch(args.data, *job, pacer), jobs),
                total=len(jobs),
                unit="file",
            )
        )

    out = os.path.join(args.data, f"ps-{'-'.join(subsets)}.txt.gz")
    kept = dropped = words = 0
    with gzip.open(out + ".tmp", "wt") as f:
        for part in parts:
            with gzip.open(part, "rt") as lines:
                for line in lines:
                    if line.rstrip("\n") in held_out:
                        dropped += 1
                        continue
                    f.write(line)
                    kept += 1
                    words += len(line.split())
    os.replace(out + ".tmp", out)
    print(
        f"{out}: {kept:,} lines, {words / 1e6:.1f} million words, "
        f"{dropped} held-out sentences left out"
    )


if __name__ == "__main__":
    main()
