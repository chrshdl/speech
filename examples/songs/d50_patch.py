"""
Makes a patch file for speech/backend/d50.py from a Roland D-50 bank dump,
the .syx file of its 64 patches:

    python examples/songs/d50_patch.py bank.syx 11 \\
        --source "Roland D-50 factory patch I-11, PN-D50-00" \\
        > speech/backend/patches/fantasia.json

The patch is its number on the D-50, 11 to 88, its bank and number. Without
a patch, it lists the bank's patches, and whether the client can play
them, see speech/backend/d50.py.
"""

import argparse
import json
import sys

from speech.backend import d50


def number(text):
    """A patch's index in a bank, from its number on the D-50, such as 11."""
    if len(text) != 2 or not all("1" <= c <= "8" for c in text):
        raise argparse.ArgumentTypeError(f"{text!r} is no patch, 11 to 88")
    return 8 * (int(text[0]) - 1) + int(text[1]) - 1


def main(args):
    with open(args.bank, "rb") as f:
        patches = d50.bank(f.read())
    if args.patch is None:
        for n, patch in enumerate(patches):
            try:
                d50.check(patch)
                playable = ""
            except ValueError as error:
                playable = f"  (not playable: {error})"
            print(f"{n // 8 + 1}{n % 8 + 1} {patch['name']}{playable}")
        return
    patch = patches[args.patch]
    d50.check(patch)
    patch = {"name": patch["name"], "source": args.source} | patch
    json.dump(patch, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bank", help="The bank dump, a .syx file.")
    parser.add_argument("patch", nargs="?", type=number, help="Its number, 11 to 88.")
    parser.add_argument("--source", default="", help="Where the patch is from.")
    main(parser.parse_args())
