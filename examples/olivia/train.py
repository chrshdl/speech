"""
Trains a DS-CNN of "Hello Edge" to spot the wake word "olivia", see
README.md, and exports it to ONNX for speech.wake_word.

Each training example is a second of audio, labeled silence, unknown
(other speech) or the wake word:

- The wake word: a window of a clip of prepare.py that holds all of the
  word, sped up or slowed down.
- Unknown: a word of Speech Commands, a word that sounds like the wake
  word, of synthesize.py, a second of a LibriSpeech utterance, a second of
  music, or a window of a wake word clip that holds at most half of the
  word, so only the whole word is heard.
- Silence: background noise at a random level, or nothing.

Noise, music, as of a TV set, reverberation and a random volume are added
to each.
"""

import argparse
import functools
import json
import math
import os
import time

# Worker processes make the examples, each on one core, so the libraries
# compute on one thread each, rather than all of them on every core.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import scipy.signal
import soundfile
import torch
import torch.utils.data as tud

from speech import loader, wake_word
from speech.models import ds_cnn
from speech.utils.dataset import read_data_json

SAMPLE_RATE = 16000
CLIP = round(wake_word.CLIP_SECONDS * SAMPLE_RATE)
LABELS = ["silence", "unknown", "olivia"]
SILENCE, UNKNOWN, KEYWORD = range(3)
# The share of each kind of example.
MIX = {
    "keyword": 0.3,
    "partial": 0.05,
    "word": 0.1,
    "similar": 0.05,
    "speech": 0.25,
    "music": 0.1,
    "silence": 0.15,
}
# The CTC alignment of prepare.py marks the word's labels, which start
# about 0.1 seconds after the word's sound starts and end about 0.05 before
# it ends, at the medians over the training clips, so the word is taken to
# start and end this much earlier and later.
START_MARGIN = 0.15
END_MARGIN = 0.1
# The augmentations, as speech.loader's, see the README there.
NOISE = {"p": 0.8, "snr": [0, 20]}
BABBLE = {"p": 0.1, "snr": [5, 20], "layers": [3, 6]}
MEDIA = {"p": 0.2, "snr": [0, 15]}
REVERB = {"p": 0.2, "delay": [2, 18], "decay": [0.55, 0.85]}
VOLUME = {"p": 0.8, "dbfs": [-40, 0]}
SPEED = [0.85, 1.15]
# The steps and learning rates of the paper.
STEPS = [10000, 10000, 10000]
RATES = [5e-4, 1e-4, 2e-5]


class Examples:
    def __init__(self, keyword, words, similar, speech, music, noise, augment=True):
        """
        Makes the examples of each kind from the datasets.

        Arguments:
            keyword (list): Jsons of wake word clips, of prepare.py or
                synthesize.py, each drawn from as often.
            words (list): Jsons of single words, such as Speech Commands.
            similar (list): Jsons of words that sound like the wake word,
                of synthesize.py. Without them, the mix has no similar
                words.
            speech (list): Jsons of sentences, such as LibriSpeech.
            music (list): Jsons of music, such as MUSAN's, also mixed into
                the examples. Without them, there is no music.
            noise (str): A directory of background noise.
            augment (bool): Whether to augment the examples.
        """
        self.keyword = [read_data_json(path) for path in keyword]
        self.clips = {}
        self.words = read_data_json(words)
        self.similar = read_data_json(similar) if similar else []
        self.music = read_data_json(music) if music else []
        self.media = music[0] if music else None
        # The kinds of examples it has the data of.
        missing = {"similar": not self.similar, "music": not self.music}
        self.mix = {k: v for k, v in MIX.items() if not missing.get(k)}
        # Without the sentences that say the wake word.
        self.speech = [
            d
            for d in read_data_json(speech)
            if LABELS[KEYWORD] not in d["text"].split()
        ]
        self.babble = speech[0] if isinstance(speech, list) else speech
        self.noise = noise
        self.augment = augment

    def kind(self):
        """A random kind of example, by its share of the mix."""
        weights = torch.tensor(list(self.mix.values()))
        return list(self.mix)[int(torch.multinomial(weights, 1))]

    def clip(self, datum):
        path = datum["audio"]
        if path not in self.clips:
            self.clips[path] = soundfile.read(path, dtype="int16")[0]
        return self.clips[path]

    def example(self, kind):
        """A random example of the kind, as 16 bit samples, and its label."""
        if kind == "keyword":
            return self.keyword_window(whole=True), KEYWORD
        if kind == "partial":
            return self.keyword_window(whole=False), UNKNOWN
        if kind in ("word", "similar"):
            data = self.words if kind == "word" else self.similar
            datum = data[loader._randint(0, len(data) - 1)]
            return window(soundfile.read(datum["audio"], dtype="int16")[0], 0), UNKNOWN
        if kind in ("speech", "music"):
            data = self.speech if kind == "speech" else self.music
            datum = data[loader._randint(0, len(data) - 1)]
            with soundfile.SoundFile(datum["audio"]) as f:
                f.seek(loader._randint(0, max(0, f.frames - CLIP)))
                return window(f.read(CLIP, dtype="int16"), 0), UNKNOWN
        return self.silence(), SILENCE

    def keyword_window(self, whole):
        """
        A window of a wake word clip that holds all of the word, or with
        whole False, at most half of it.
        """
        data = self.keyword[loader._randint(0, len(self.keyword) - 1)]
        datum = data[loader._randint(0, len(data) - 1)]
        audio = self.clip(datum)
        start = (datum["start"] - START_MARGIN) * SAMPLE_RATE
        end = (datum["end"] + END_MARGIN) * SAMPLE_RATE
        if self.augment:
            factor = loader._uniform(*SPEED)
            audio = speed(audio, factor)
            start, end = start / factor, end / factor
        start, end = round(start), round(end)
        half = (end - start) // 2
        if whole:
            # The window starts from where it ends with the word to where
            # it starts with it.
            low, high = end - CLIP, start
            if low > high:
                low = high = (low + high) // 2
        elif loader._chance(0.5):
            # Ends within the first half of the word.
            low, high = start - CLIP, start + half - CLIP
        else:
            # Starts within the second half.
            low, high = end - half, end
        offset = loader._randint(low, high) if self.augment else (low + high) // 2
        return window(audio, offset)

    def silence(self):
        if not self.augment or loader._chance(0.1):
            return np.zeros(CLIP, np.int16)
        if self.noise not in loader._noise_sources:
            loader._noise_sources[self.noise] = loader.NoiseSource(self.noise)
        audio = loader._noise_sources[self.noise].segment(CLIP, SAMPLE_RATE)
        peak = np.abs(audio).max()
        if peak == 0:
            return np.zeros(CLIP, np.int16)
        level = loader._uniform(-70, -10)
        return np.round(audio * (32767 * 10 ** (level / 20) / peak)).astype(np.int16)

    def augmented(self, audio):
        """The 16 bit audio with noise, reverberation and a new volume."""
        if not self.augment:
            return audio
        signal = audio.astype(np.float32) / 32768
        signal = loader.noise(signal, self.noise, **NOISE)
        signal = loader.noise(signal, self.babble, **BABBLE)
        if self.media is not None:
            signal = loader.noise(signal, self.media, **MEDIA)
        signal = loader.reverb(signal, **REVERB)
        signal = loader.volume(signal * 32768, **VOLUME)
        return np.clip(np.round(signal), -32768, 32767).astype(np.int16)


def window(audio, offset):
    """CLIP samples of the audio from offset, padded with silence."""
    out = np.zeros(CLIP, np.int16)
    start, stop = max(offset, 0), min(offset + CLIP, len(audio))
    if stop > start:
        out[start - offset : stop - offset] = audio[start:stop]
    return out


def speed(audio, factor):
    """
    The audio sped up by the factor, which raises its pitch, as
    resampling it does.
    """
    up = 100
    down = round(up * factor)
    out = scipy.signal.resample_poly(audio.astype(np.float32), up, down)
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


class Training(tud.IterableDataset):
    def __init__(self, examples):
        """Random augmented examples of the mix of kinds, without end."""
        self.examples = examples
        self.mfcc = wake_word.Mfcc(SAMPLE_RATE)

    def __iter__(self):
        while True:
            kind = self.examples.kind()
            audio, label = self.examples.example(kind)
            if kind != "silence":
                audio = self.examples.augmented(audio)
            yield self.mfcc(audio), label


def dev_set(examples, count, seed=0):
    """
    A fixed set of count examples of the mix of kinds, without
    augmentation, as MFCCs, labels and kinds.
    """
    torch.manual_seed(seed)
    mfcc = wake_word.Mfcc(SAMPLE_RATE)
    features, labels, kinds = [], [], []
    for _ in range(count):
        kind = examples.kind()
        audio, label = examples.example(kind)
        features.append(mfcc(audio))
        labels.append(label)
        kinds.append(kind)
    return torch.from_numpy(np.stack(features)), torch.tensor(labels), kinds


def evaluate(model, dev, device):
    """The dev loss, and the accuracy of each kind of example."""
    features, labels, kinds = dev
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x.to(device)).cpu() for x in features.split(1000)])
    model.train()
    loss = float(torch.nn.functional.cross_entropy(logits, labels))
    right = (logits.argmax(1) == labels).numpy()
    kinds = np.array(kinds)
    return loss, {k: float(right[kinds == k].mean()) for k in MIX if k in kinds}


def seed_worker(seed, worker):
    torch.manual_seed(seed * 1000 + worker)
    # Each worker makes one example at a time, and the workers together
    # keep the cores busy.
    torch.set_num_threads(1)


def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device(args.device)
    train = Examples(
        args.keyword, args.words, args.similar, args.speech, args.music, args.noise
    )
    dev = Examples(
        args.dev_keyword,
        args.dev_words,
        args.dev_similar,
        args.dev_speech,
        args.dev_music,
        args.noise,
        augment=False,
    )
    dev = dev_set(dev, args.dev_examples)
    torch.manual_seed(args.seed)

    data = tud.DataLoader(
        Training(train),
        batch_size=args.batch_size,
        num_workers=args.workers,
        worker_init_fn=functools.partial(seed_worker, args.seed),
        persistent_workers=args.workers > 0,
    )
    batches = iter(data)
    # The features' statistics, from the first batches.
    sample = torch.cat([next(batches)[0] for _ in range(20)])
    model = ds_cnn.DSCNN(LABELS, LABELS[KEYWORD], args.size, args.dropout)
    model.set_normalization(sample.mean(dim=(0, 1)), sample.std(dim=(0, 1)))
    model.to(device)
    print(
        f"DS-CNN-{args.size}: {sum(p.numel() for p in model.parameters()):,} parameters",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=RATES[0])

    best = math.inf
    total = sum(STEPS)
    started = time.time()
    running = 0.0
    for step in range(1, total + 1):
        stage = int(np.searchsorted(np.cumsum(STEPS), step - 1, side="right"))
        for group in optimizer.param_groups:
            group["lr"] = RATES[stage]
        features, labels = next(batches)
        loss = torch.nn.functional.cross_entropy(
            model(features.to(device)), labels.to(device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
        if step % args.eval_every == 0 or step == total:
            dev_loss, accuracy = evaluate(model, dev, device)
            mark = ""
            if dev_loss < best:
                best = dev_loss
                ds_cnn.save(model.cpu(), os.path.join(args.save_path, "model.pt"))
                model.to(device)
                mark = " *"
            kinds = " ".join(f"{k} {v:.3f}" for k, v in accuracy.items())
            print(
                f"step {step} {time.time() - started:.0f}s train loss "
                f"{running / args.eval_every:.4f} dev loss {dev_loss:.4f} {kinds}{mark}",
                flush=True,
            )
            running = 0.0

    model = ds_cnn.load(os.path.join(args.save_path, "model.pt"))
    path = os.path.join(args.save_path, "model.onnx")
    ds_cnn.export(model, SAMPLE_RATE, path)
    with open(os.path.join(args.save_path, "args.json"), "w") as fid:
        json.dump(vars(args), fid, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("save_path", help="The directory to write the model to.")
    parser.add_argument(
        "--keyword",
        nargs="+",
        default=["examples/olivia/data/train.json"],
        help="Jsons of wake word clips, of prepare.py or synthesize.py.",
    )
    parser.add_argument(
        "--dev-keyword", nargs="+", default=["examples/olivia/data/dev.json"]
    )
    parser.add_argument(
        "--similar",
        nargs="*",
        default=[],
        help="Jsons of words that sound like the wake word, of synthesize.py.",
    )
    parser.add_argument("--dev-similar", nargs="*", default=[])
    parser.add_argument(
        "--music",
        nargs="*",
        default=[],
        help="Jsons of music, such as examples/musan's music-train.json. The first "
        "is also mixed into the examples.",
    )
    parser.add_argument("--dev-music", nargs="*", default=[])
    parser.add_argument("--size", default="S", choices=list(ds_cnn.SIZES))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--words",
        nargs="+",
        default=["examples/speech_commands/data/train.json"],
    )
    parser.add_argument(
        "--speech",
        nargs="+",
        default=["examples/librispeech/data/LibriSpeech/train-clean-100.json"],
        help="Sentences of other speech. The first is also the babble.",
    )
    parser.add_argument(
        "--dev-words",
        nargs="+",
        default=["examples/speech_commands/data/dev.json"],
    )
    parser.add_argument(
        "--dev-speech",
        nargs="+",
        default=["examples/librispeech/data/LibriSpeech/dev-clean.json"],
    )
    parser.add_argument("--noise", default="examples/noise/data")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dev-examples", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
