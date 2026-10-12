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
from speech.intents import Intent, IntentRecognizer
from speech.utils import wave

# Silence around each clip, in seconds. Within a stream, as in the game, a
# word comes after other audio, which the model needs to settle.
PADDING = 0.5


def evaluate(model, data):
    """
    Returns the counts of the clips closest to their word and transcribed
    as it, and for each clip its word and the command closest to it, with
    its score, or None, so the commands can be counted for any threshold.
    """
    words = sorted({d["text"] for d in data})
    vocabulary = IntentRecognizer(
        [Intent(word, (word,)) for word in words],
        model.char_to_int,
        model.blank,
        model.frame_seconds,
    )
    commands = game.commands(model)
    counts = collections.Counter()
    heard = []
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

        match = commands.closest(probs)
        if match is None:
            heard.append((word, None, None))
        else:
            heard.append((word, match.intent.name.lower(), match.score))
    return counts, heard


def command_counts(heard, threshold):
    """
    The "left" and "right" clips, how many of them the commands recognize
    at the threshold, and in the wrong direction, and the other words taken
    for a command.
    """
    counts = collections.Counter()
    for word, command, score in heard:
        if command is not None and score < threshold:
            command = None
        if word in ("left", "right"):
            counts["commands"] += 1
            if command == word:
                counts["recognized"] += 1
            elif command is not None:
                counts["wrong"] += 1
        else:
            counts["others"] += 1
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
        default=str(game.THRESHOLD),
        help="The lowest score of a command, as backend.py's --command-threshold, "
        "or several, comma separated, to compare them from one pass over the clips.",
    )
    args = parser.parse_args()
    with open(args.data) as fid:
        data = [json.loads(line) for line in fid]
    counts, heard = evaluate(streaming.load(args.model), data)
    n = len(data)
    print(f"Closest of the words: {counts['closest'] / n:.3f} of {n} clips")
    print(f"Transcript is the word: {counts['transcript'] / n:.3f}")
    for threshold in (float(t) for t in args.threshold.split(",")):
        c = command_counts(heard, threshold)
        print(
            f"Threshold {threshold}: commands recognized {c['recognized']} of "
            f"{c['commands']}, in the wrong direction {c['wrong']}, other words "
            f"taken for a command {c['others taken']} of {c['others']}"
        )
