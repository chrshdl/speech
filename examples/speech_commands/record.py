"""
Records a speaker saying the game's commands and other words, one prompt
at a time, from the microphone, such as a child, to measure how a model
recognizes that speaker. Writes the clips and a clips.json like Speech
Commands' sets.
"""

import argparse
import json
import os
import random

import numpy as np
import sounddevice
import soundfile

SAMPLE_RATE = 16000
SECONDS = 2.5
COMMANDS = ["left", "right"]
# What a child might say while playing, which must not move the unicorn.
OTHERS = [
    "yay",
    "oh no",
    "wow",
    "hello",
    "again",
    "yes",
    "no",
    "bye bye",
    "faster",
    "unicorn",
    "look",
    "my name is",
]
# A clip quieter than this, in dBFS at its loudest, likely missed the word.
QUIET = -35


def prompts(repeats):
    words = COMMANDS * repeats + OTHERS[: repeats * len(COMMANDS) // 2]
    random.shuffle(words)
    return words


def record(seconds):
    audio = sounddevice.rec(
        int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16"
    )
    sounddevice.wait()
    return audio[:, 0]


def peak_dbfs(audio):
    peak = np.abs(audio.astype(np.float64)).max()
    return 20 * np.log10(max(peak, 1) / 32768)


def main(out_dir, repeats):
    os.makedirs(out_dir, exist_ok=True)
    words = prompts(repeats)
    clips = []
    i = 0
    print(f"{len(words)} prompts. For each, press Enter, then the word is said.")
    print("Type r and Enter to redo the last one, q and Enter to stop.\n")
    while i < len(words):
        word = words[i]
        answer = input(f"[{i + 1}/{len(words)}]  Say:  {word.upper():12s}  Enter: ")
        if answer.strip() == "q":
            break
        if answer.strip() == "r" and clips:
            clips.pop()
            i -= 1
            print("  Again.")
            continue
        print("  Listening…", flush=True)
        audio = record(SECONDS)
        path = os.path.join(out_dir, f"{i:03d}_{word.replace(' ', '_')}.wav")
        soundfile.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        clips.append({"text": word, "duration": SECONDS, "audio": path})
        level = peak_dbfs(audio)
        note = "  That was quiet, r to redo." if level < QUIET else ""
        print(f"  Saved, loudest {level:.0f} dBFS.{note}")
        i += 1
    with open(os.path.join(out_dir, "clips.json"), "w") as fid:
        fid.writelines(json.dumps(clip) + "\n" for clip in clips)
    print(f"\nWrote {len(clips)} clips to {out_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "out_dir",
        nargs="?",
        default="examples/speech_commands/data/recorded",
        help="Where to save the clips.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=12,
        help="How often to say each command. Half as many other words come with them.",
    )
    args = parser.parse_args()
    main(args.out_dir, args.repeats)
