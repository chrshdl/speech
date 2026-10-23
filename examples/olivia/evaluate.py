"""
Measures how a wake word model of train.py spots "olivia" in streams of
audio, as speech.wake_word.Detector decides: the share of the wake word
clips it misses, and how often it hears the wake word in other audio,
per hour, for each threshold and smoothing. With --asr, a speech model
confirms each time it is heard, as backend.py does.
"""

import argparse
import math
import os

import numpy as np
import soundfile
import tqdm

from speech import streaming, tokens, wake_word
from speech.utils.dataset import read_data_json
from speech.vad import VadModel

# The chunks of audio that a GatedDetector takes, as the microphone's.
CHUNK_SECONDS = 0.1
# The silence around each wake word clip, in seconds.
BEFORE = 1.0
AFTER = 1.0


def read(path):
    return soundfile.read(path, dtype="int16")[0]


def add_noise(audio, clip, noise, snr, rng):
    """The audio with the noise at the signal to noise ratio in dB of the clip."""
    source = noise[rng.integers(len(noise))]
    if len(source) < len(audio):
        source = np.tile(source, math.ceil(len(audio) / len(source)))
    start = rng.integers(len(source) - len(audio) + 1)
    layer = source[start : start + len(audio)].astype(np.float64)
    power = np.mean(clip.astype(np.float64) ** 2)
    gain = math.sqrt(power / max(np.mean(layer**2), 1e-9) / 10 ** (snr / 10))
    return np.clip(np.round(audio + gain * layer), -32768, 32767).astype(np.int16)


class Streams:
    def __init__(self, model, hop, audio, vad=None):
        """
        Hears the wake word in streams of audio, from the functions that
        return each stream's audio, which are called again to confirm its
        detections. Without a voice activity detector, it computes the wake
        word's probability in each window of each stream once, and decides
        on it for each threshold and smoothing. With one, it streams each
        stream through a wake_word.GatedDetector for each.
        """
        self.model = model
        self.hop = hop
        self.audio = audio
        self.vad = vad
        self.scores = []
        self.online = {}
        seconds = 0.0
        for get in tqdm.tqdm(audio, leave=False):
            samples = get()
            seconds += len(samples) / model.sample_rate
            if vad is None:
                self.scores.append(model.scores(samples, hop))
        self.hours = seconds / 3600

    def __len__(self):
        return len(self.audio)

    def detections(self, index, threshold, smoothing):
        """The sample at which each detection in a stream is decided."""
        if self.vad is None:
            blocked = wake_word.blocked_hops(self.model, self.hop, smoothing)
            found = wake_word.detections(
                self.scores[index], threshold, smoothing, blocked
            )
            return [wake_word.decision_end(self.model, i, self.hop) for i in found]
        key = (index, threshold, smoothing)
        if key not in self.online:
            detector = wake_word.GatedDetector(
                self.model, self.vad, threshold, smoothing, self.hop
            )
            audio = self.audio[index]()
            chunk = round(CHUNK_SECONDS * self.model.sample_rate)
            self.online[key] = [
                min(start + chunk, len(audio))
                for start in range(0, len(audio), chunk)
                if detector.push(audio[start : start + chunk])
            ]
        return self.online[key]


class Confirmation:
    def __init__(self, path, keyword, model, threshold):
        """
        Confirms a detection as backend.py does, with the streaming CTC
        model at path, once each.
        """
        self.asr = streaming.load(path)
        labels = tokens.tokenize(keyword, self.asr.char_to_int)
        self.labels = np.array([self.asr.char_to_int[t] for t in labels])
        self.model = model
        self.threshold = threshold
        self.cache = {}

    def __call__(self, streams, index, end):
        key = (id(streams), index, end)
        if key not in self.cache:
            rate = self.model.sample_rate
            audio = streams.audio[index]()
            start = max(0, end - round(wake_word.LOOKBACK_SECONDS * rate))
            stop = end + round(wake_word.CONFIRM_SECONDS * rate)
            transcriber = streaming.Transcriber(self.asr)
            transcriber.push(audio[start:stop])
            aligned = wake_word.align(transcriber.probs(), self.labels, self.asr.blank)
            self.cache[key] = aligned is not None and aligned[0] >= self.threshold
        return self.cache[key]


def detections(streams, index, threshold, smoothing, confirm):
    found = streams.detections(index, threshold, smoothing)
    if confirm is not None:
        found = [end for end in found if confirm(streams, index, end)]
    return found


def main(args):
    model = wake_word.WakeWordModel(args.model, args.threads)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    smoothings = [int(s) for s in args.smoothing.split(",")]
    rate = model.sample_rate
    noise = [read(p) for p in args.noise]
    vad = VadModel(args.vad) if args.vad else None

    def clips(json_path, snr=None):
        """Each clip with silence around it, and noise at the snr in dB."""
        rng = np.random.default_rng(0)
        audio = []
        for datum in read_data_json(json_path):
            clip = read(datum["audio"])
            before = np.zeros(round(BEFORE * rate), np.int16)
            after = np.zeros(round(AFTER * rate), np.int16)
            stream = np.concatenate([before, clip, after])
            if snr is not None:
                stream = add_noise(stream, clip, noise, snr, rng)
            audio.append(lambda stream=stream: stream)
        return Streams(model, args.hop, audio, vad)

    keyword = {}
    for snr in [None, *args.snr]:
        keyword["clean" if snr is None else f"{snr:g} dB"] = clips(args.keyword, snr)
    for json_path in args.extra_keyword:
        keyword[os.path.basename(json_path)] = clips(json_path)
    similar = {os.path.basename(p): clips(p) for p in args.similar}
    others = {}
    for name, json_path in zip(args.other_names, args.other, strict=True):
        paths = [d["audio"] for d in read_data_json(json_path)]
        others[name] = Streams(
            model, args.hop, [lambda p=p: read(p) for p in paths], vad
        )
    others["noise"] = Streams(
        model, args.hop, [lambda p=p: read(p) for p in args.noise], vad
    )

    confirm = None
    if args.asr:
        confirm = Confirmation(args.asr, model.keyword, model, args.asr_threshold)
    header = ["threshold", "smoothing"]
    header += [f"missed {name}" for name in keyword]
    header += [f"taken {name}" for name in similar]
    header += [f"per hour {name} ({s.hours:.1f} h)" for name, s in others.items()]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for smoothing in smoothings:
        for threshold in thresholds:
            row = [f"{threshold:g}", str(smoothing)]
            for streams in keyword.values():
                missed = sum(
                    not detections(streams, i, threshold, smoothing, confirm)
                    for i in range(len(streams))
                )
                row.append(f"{missed / len(streams):.1%}")
            for streams in similar.values():
                taken = sum(
                    bool(detections(streams, i, threshold, smoothing, confirm))
                    for i in range(len(streams))
                )
                row.append(f"{taken / len(streams):.1%}")
            for streams in others.values():
                alarms = sum(
                    len(detections(streams, i, threshold, smoothing, confirm))
                    for i in range(len(streams))
                )
                row.append(f"{alarms / streams.hours:.2f}")
            print("| " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="A model.onnx of train.py.")
    parser.add_argument("--keyword", default="examples/olivia/data/test.json")
    parser.add_argument(
        "--extra-keyword",
        nargs="*",
        default=[],
        help="Jsons of more wake word clips, such as synthesize.py's in other voices.",
    )
    parser.add_argument(
        "--similar",
        nargs="*",
        default=[],
        help="Jsons of words that sound like the wake word, of synthesize.py, of "
        "which it counts how many are taken for it.",
    )
    parser.add_argument(
        "--other",
        nargs="+",
        default=[
            "examples/librispeech/data/LibriSpeech/test-clean.json",
            "examples/speech_commands/data/test.json",
        ],
        help="Jsons of audio without the wake word.",
    )
    parser.add_argument(
        "--other-names", nargs="+", default=["LibriSpeech", "Speech Commands"]
    )
    parser.add_argument(
        "--noise",
        nargs="+",
        default=[
            f"examples/noise/data/{name}.wav"
            for name in (
                "doing_the_dishes",
                "dude_miaowing",
                "exercise_bike",
                "running_tap",
            )
        ],
        help="Background noise, also mixed into the wake word clips at each --snr.",
    )
    parser.add_argument("--snr", type=float, nargs="*", default=[10.0])
    parser.add_argument("--thresholds", default="0.5,0.7,0.8,0.9,0.95,0.98")
    parser.add_argument("--smoothing", default="1,3")
    parser.add_argument("--hop", type=float, default=wake_word.HOP_SECONDS)
    parser.add_argument(
        "--asr",
        help="A streaming CTC model that confirms each detection, as backend.py's "
        "model does.",
    )
    parser.add_argument("--asr-threshold", type=float, default=wake_word.ASR_THRESHOLD)
    parser.add_argument(
        "--vad",
        help="Silero VAD's silero_vad.onnx, which lets the wake word model run only "
        "while it hears speech, as backend.py's --vad does.",
    )
    parser.add_argument("--threads", type=int)
    main(parser.parse_args())
