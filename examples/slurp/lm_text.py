"""
Writes the text of requests to a voice assistant, for a word LM of what
the device hears: SLURP's training sentences, recorded and synthesized,
and the samples of the backend's intents with each value of their slots,
such as "what's the weather in hamburg", one per line, normalized like
the acoustic model's transcripts. Sentences of SLURP's devel and test
sets are left out, so that they measure the LM fairly:

    examples/slurp/data/slurp-train-text.txt
    examples/slurp/data/intents-text.txt

The sentences come from SLURP's repository (CC BY 4.0), as for
prepare.py.
"""

import argparse
import itertools
import os

from prepare import sentences

from speech.backend import brightness, game, keyboard, songs, weather
from speech.intents import SLOT
from speech.utils.prepare import normalize


def phrases(intent):
    """Each sample of the intent, with each value of its slots."""
    for sample in intent.samples:
        names = SLOT.findall(sample)
        for values in itertools.product(*(intent.slots[name] for name in names)):
            text = sample
            for name, value in zip(names, values):
                text = text.replace("{" + name + "}", value, 1)
            yield text


def intents(songs_dir):
    """The intents and commands backend.py recognizes."""
    found = [weather.INTENT, game.INTENT, *game.COMMANDS, keyboard.INTENT]
    found.append(brightness.INTENT)
    jukebox = songs.Songs(songs.load(songs_dir))
    if jukebox.intent is not None:
        found.append(jukebox.intent)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", default="examples/slurp/data", help="Where to write the text."
    )
    parser.add_argument(
        "--songs",
        default="examples/songs",
        help="The songs whose names the intent of playing one has, as backend.py's "
        "--songs.",
    )
    args = parser.parse_args()
    os.makedirs(args.data, exist_ok=True)
    held_out = {
        text
        for split in ("devel", "test")
        for text in sentences(split, args.data)[0].values()
    }

    train = [
        text
        for split in ("train", "train_synthetic")
        for text in sentences(split, args.data)[0].values()
        if text not in held_out
    ]
    with open(os.path.join(args.data, "slurp-train-text.txt"), "w") as out:
        out.writelines(text + "\n" for text in train)

    texts = {normalize(p) for intent in intents(args.songs) for p in phrases(intent)}
    texts = sorted(texts - held_out - {None})
    with open(os.path.join(args.data, "intents-text.txt"), "w") as out:
        out.writelines(text + "\n" for text in texts)
    print(f"{len(train)} training sentences and {len(texts)} intent phrases")


if __name__ == "__main__":
    main()
