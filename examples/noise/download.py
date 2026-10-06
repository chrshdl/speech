"""
Downloads the background noise recordings of the Speech Commands dataset by
Pete Warden, released under CC BY 4.0, for noise augmentation.
"""

import argparse
import os
import shutil
import tarfile
import urllib.request

URL = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
FOLDER = "_background_noise_"


def download(out_dir):
    """
    Streams the archive and extracts only its noise folder, stopping once
    the archive has moved past it. Returns the extracted file names.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = []
    with (
        urllib.request.urlopen(URL) as response,
        tarfile.open(fileobj=response, mode="r|gz") as tf,
    ):
        for member in tf:
            parts = os.path.normpath(member.name).split(os.sep)
            if FOLDER not in parts:
                if names:
                    break
                continue
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            with open(os.path.join(out_dir, name), "wb") as fid:
                shutil.copyfileobj(tf.extractfile(member), fid)
            names.append(name)
    return names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the background noise of Speech Commands."
    )
    parser.add_argument("out_dir", help="Where to save the noise recordings.")
    args = parser.parse_args()
    for name in download(args.out_dir):
        print(name)
