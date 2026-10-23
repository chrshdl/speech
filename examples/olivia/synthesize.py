"""
Synthesizes "olivia", and words that sound like it, in many voices with
Piper, to train and test the wake word model with more speakers than the
dataset has, as openWakeWord trains its models.

Each multi-speaker Piper voice writes its own sets, so the voices of the
training sets and of the test sets can differ:

- <name>.json: "olivia", with where the word starts and ends, found and
  checked as prepare.py does.
- <name>-words.json: words that sound like it, such as "oliver" and
  "bolivia", which the model must not take for it.
"""

import argparse
import json
import math
import os

import numpy as np
import scipy.signal
import soundfile
import tqdm
from piper import PiperVoice, SynthesisConfig
from prepare import MIN_SCORE, SAMPLE_RATE, WORD, align

from speech import streaming

# How the wake word is said.
SAYINGS = ["Olivia.", "Olivia!", "Olivia?", "Olivia,"]
# Words and phrases that sound like it, or like a part of it.
SIMILAR = [
    "Oliver",
    "olive",
    "olives",
    "olive oil",
    "Bolivia",
    "Livia",
    "Lydia",
    "Sylvia",
    "Octavia",
    "Ophelia",
    "Amelia",
    "Alicia",
    "Cecilia",
    "Lavinia",
    "Virginia",
    "trivia",
    "oblivion",
    "obvious",
    "alleviate",
    "a living",
    "all of you",
    "oh really",
    "Liv",
    "Olivetti",
    "Bolivian",
]


# The silence added before and after each clip, in seconds, as recorded
# clips have.
PAD = 0.25


def to_samples(chunks, rate):
    """
    The 16 bit samples at SAMPLE_RATE of Piper's audio chunks, with PAD
    seconds of silence before and after.
    """
    audio = np.concatenate([c.audio_float_array for c in chunks])
    silence = np.zeros(round(PAD * rate), np.float32)
    audio = np.concatenate([silence, audio, silence])
    g = math.gcd(rate, SAMPLE_RATE)
    audio = scipy.signal.resample_poly(audio, SAMPLE_RATE // g, rate // g)
    return np.clip(np.round(audio * 32767), -32768, 32767).astype(np.int16)


def main(args):
    rng = np.random.default_rng(args.seed)
    voice = PiperVoice.load(args.voice)
    rate = voice.config.sample_rate
    speakers = list(range(voice.config.num_speakers))
    asr = streaming.load(args.asr_model)
    labels = np.array([asr.char_to_int[c] for c in WORD])
    clips = os.path.join(args.out_dir, "clips", args.name)
    os.makedirs(clips, exist_ok=True)

    def synthesize(text, speaker):
        config = SynthesisConfig(
            speaker_id=speaker,
            length_scale=rng.uniform(0.8, 1.3),
            noise_scale=rng.uniform(0.4, 0.9),
            noise_w_scale=rng.uniform(0.5, 1.0),
        )
        return to_samples(voice.synthesize(text, config), rate)

    keyword, dropped = [], 0
    for i in tqdm.tqdm(range(args.keyword), desc=WORD):
        audio = synthesize(SAYINGS[i % len(SAYINGS)], speakers[i % len(speakers)])
        aligned = align(asr, labels, audio)
        if aligned is None or aligned[0] < MIN_SCORE:
            dropped += 1
            continue
        path = os.path.join(clips, f"{WORD}_{i}.wav")
        soundfile.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        score, start, end = aligned
        keyword.append(
            {
                "audio": path,
                "duration": round(len(audio) / SAMPLE_RATE, 3),
                "text": WORD,
                "start": round(start, 3),
                "end": round(end, 3),
                "score": round(score, 3),
            }
        )
    words = []
    for i in tqdm.tqdm(range(args.words), desc="similar words"):
        text = SIMILAR[i % len(SIMILAR)]
        audio = synthesize(text, speakers[rng.integers(len(speakers))])
        path = os.path.join(clips, f"word_{i}.wav")
        soundfile.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        duration = round(len(audio) / SAMPLE_RATE, 3)
        words.append({"audio": path, "duration": duration, "text": text.lower()})

    for suffix, data in (("", keyword), ("-words", words)):
        with open(os.path.join(args.out_dir, f"{args.name}{suffix}.json"), "w") as fid:
            fid.writelines(json.dumps(d) + "\n" for d in data)
    print(
        f"{args.name}: {len(keyword)} of {WORD}, {dropped} not heard, {len(words)} similar"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("voice", help="A multi-speaker Piper voice, a .onnx file.")
    parser.add_argument("asr_model", help="The CTC model of prepare.py.")
    parser.add_argument("name", help="The name of the sets, such as synth-train.")
    parser.add_argument("--keyword", type=int, default=2000)
    parser.add_argument("--words", type=int, default=2000)
    parser.add_argument("--out-dir", default="examples/olivia/data")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
