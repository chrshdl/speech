import argparse
import tarfile
import urllib.request

EXT = ".tar.gz"
FILES = ["raw-metadata", "train-clean-100", "dev-clean"]
BASE_URL = "https://www.openslr.org/resources/12/"


def download_and_extract(in_file, out_dir):
    # Extract while downloading so the archive is never stored.
    file_url = BASE_URL + in_file + EXT
    with (
        urllib.request.urlopen(file_url) as response,
        tarfile.open(fileobj=response, mode="r|gz") as tf,
    ):
        tf.extractall(path=out_dir, filter="data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download librispeech dataset.")

    parser.add_argument(
        "output_directory",
        help="The dataset is saved in <output_directory>/LibriSpeech.",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=FILES,
        help="The subsets to download, for example dev-clean test-clean.",
    )
    args = parser.parse_args()

    for f in args.sets:
        print(f"Downloading {f}")
        download_and_extract(f, args.output_directory)
