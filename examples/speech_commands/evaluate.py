"""
Measures how a streaming CTC model recognizes the words of Speech
Commands' test set, each said alone: which of the 35 words each clip is
closest to, whether its transcript is the word, and how the game's
commands, "left" and "right", are recognized among the other words.
"""

import argparse
import collections
import json

import numpy as np

from speech import streaming
from speech.backend import game
from speech.features import log_specgram
from speech.intents import THRESHOLD, Intent, IntentRecognizer
from speech.utils import wave

# Silence around each clip, in seconds. Within a stream, as in the game, a
# word comes after other audio, which the model needs to settle.
PADDING = 0.5


def evaluate(model, data, threshold):
    words = sorted({d["text"] for d in data})
    vocabulary = IntentRecognizer(
        [Intent(word, (word,)) for word in words],
        model.char_to_int,
        model.blank,
        model.frame_seconds,
    )
    commands = IntentRecognizer(
        game.COMMANDS,
        model.char_to_int,
        model.blank,
        model.frame_seconds,
        threshold,
        spaced=True,
    )
    counts = collections.Counter()
    for datum in data:
        audio, sample_rate = wave.array_from_wave(datum["audio"])
        padding = np.zeros(int(PADDING * sample_rate), audio.dtype)
        audio = np.concatenate([padding, audio, padding])
        probs = model.probs(log_specgram(audio, sample_rate))
        word = datum["text"]

        closest = vocabulary.closest(probs)
        counts["closest"] += closest is not None and closest.intent.name == word
        best = probs.argmax(axis=1)
        labels = [
            label
            for i, label in enumerate(best)
            if label != model.blank and (i == 0 or label != best[i - 1])
        ]
        counts["transcript"] += model.decode(labels).strip() == word

        match = commands.recognize(probs)
        command = match.intent.name.lower() if match is not None else None
        if word in ("left", "right"):
            counts["commands"] += 1
            if command == word:
                counts["recognized"] += 1
            elif command is not None:
                counts["wrong"] += 1
        else:
            counts["others taken"] += command is not None
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "model", help="A .onnx file from export.py, or a model directory."
    )
    parser.add_argument(
        "--data",
        default="examples/speech_commands/data/test.json",
        help="The clips, by default the test set.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="The lowest score of a command, as backend.py's --intent-threshold.",
    )
    args = parser.parse_args()
    with open(args.data) as fid:
        data = [json.loads(line) for line in fid]
    counts = evaluate(streaming.load(args.model), data, args.threshold)
    n, words = len(data), counts["commands"]
    others = n - words
    print(f"Closest of the words: {counts['closest'] / n:.3f} of {n} clips")
    print(f"Transcript is the word: {counts['transcript'] / n:.3f}")
    print(
        f"Commands recognized: {counts['recognized']} of {words}, "
        f"in the wrong direction: {counts['wrong']}"
    )
    print(f"Other words taken for a command: {counts['others taken']} of {others}")
