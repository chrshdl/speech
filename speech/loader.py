import glob
import json
import math
import os
import random

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile
import torch
import torch.utils.data as tud

from speech.utils import wave

# The training augmentations a config's data can set, in the order they
# are applied: noise, reverb and volume on the audio, pitch and tempo on
# the spectrogram, then spec_augment on the normalized features.
AUGMENTATIONS = ("noise", "reverb", "volume", "pitch", "tempo", "spec_augment")


class Preprocessor:
    END = "</s>"
    START = "<s>"

    def __init__(self, data_json, max_samples=100, start_and_end=True):
        """
        Builds a preprocessor from a dataset.
        Arguments:
            data_json (string or list): A file containing a json
                representation of each example per line, or a list of
                such files.
            max_samples (int): The maximum number of examples to be used
                in computing summary statistics.
            start_and_end (bool): Include start and end tokens in labels.
        """
        data = read_data_json(data_json)

        # Compute data mean, std from sample
        audio_files = [d["audio"] for d in data]
        random.shuffle(audio_files)
        self.mean, self.std = compute_mean_std(audio_files[:max_samples])
        self._input_dim = self.mean.shape[0]

        # Make char map
        chars = sorted({t for d in data for t in d["text"]})
        if start_and_end:
            # START must be last so it can easily be
            # excluded in the output classes of a model.
            chars.extend([self.END, self.START])
        self.start_and_end = start_and_end
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {v: k for k, v in self.int_to_char.items()}

    def encode(self, text):
        text = list(text)
        if self.start_and_end:
            text = [self.START] + text + [self.END]
        return [self.char_to_int[t] for t in text]

    def decode(self, seq):
        text = [self.int_to_char[s] for s in seq]
        if not self.start_and_end:
            return text

        s = text[0] == self.START
        e = len(text)
        if text[-1] == self.END:
            e = text.index(self.END)
        return text[s:e]

    def preprocess(self, wave_file, text):
        inputs = self.normalize(log_specgram_from_file(wave_file))
        targets = self.encode(text)
        return inputs, targets

    def normalize(self, features):
        return (features - self.mean) / self.std

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def vocab_size(self):
        return len(self.int_to_char)


def compute_mean_std(audio_files):
    samples = [log_specgram_from_file(af) for af in audio_files]
    samples = np.vstack(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return mean, std


class AudioDataset(tud.Dataset):
    def __init__(self, data_json, preproc, batch_size, augment=None):
        """
        Arguments:
            augment (dict, optional): Maps the names in AUGMENTATIONS to
                the arguments of their functions, to augment each example
                as it is loaded, or a list of them for noise. Each can
                also have a from_epoch, the epoch, counted from 0, from
                which it is applied, so the model can learn from clean
                audio first.
        """

        data = read_data_json(data_json)
        self.preproc = preproc
        self.augment = augment or {}
        self.set_epoch(0)

        bucket_diff = 4
        max_len = max(len(x["text"]) for x in data)
        num_buckets = max_len // bucket_diff
        buckets = [[] for _ in range(num_buckets)]
        for d in data:
            bid = min(len(d["text"]) // bucket_diff, num_buckets - 1)
            buckets[bid].append(d)

        # Sort by input length followed by output length
        sort_fn = lambda x: (round(x["duration"], 1), len(x["text"]))
        for b in buckets:
            b.sort(key=sort_fn)
        data = [d for b in buckets for d in b]
        self.data = data

    def __len__(self):
        return len(self.data)

    def set_epoch(self, epoch):
        """
        Sets the training epoch, counted from 0, which decides the
        augmentations that have a from_epoch.
        """
        self.epoch = epoch
        self.active = {}
        for name, configs in self.augment.items():
            # Noise can be one overlay or a list, such as noise and babble.
            configs = configs if isinstance(configs, list) else [configs]
            on = [
                {k: v for k, v in c.items() if k != "from_epoch"}
                for c in configs
                if epoch >= c.get("from_epoch", 0)
            ]
            if on:
                self.active[name] = on if name == "noise" else on[0]

    def __getitem__(self, idx):
        datum = self.data[idx]
        augment = self.active
        if not augment:
            return self.preproc.preprocess(datum["audio"], datum["text"])

        audio, sample_rate = wave.array_from_wave(datum["audio"])
        for overlay in augment.get("noise", []):
            audio = noise(audio, sample_rate=sample_rate, **overlay)
        if "reverb" in augment:
            audio = reverb(audio, sample_rate=sample_rate, **augment["reverb"])
        if "volume" in augment:
            audio = volume(audio, **augment["volume"])
        features = log_specgram(audio, sample_rate)
        if "pitch" in augment:
            features = pitch(features, **augment["pitch"])
        if "tempo" in augment:
            features = tempo(features, **augment["tempo"])
        inputs = self.preproc.normalize(features)
        if "spec_augment" in augment:
            inputs = spec_augment(inputs, **augment["spec_augment"])
        return inputs, self.preproc.encode(datum["text"])


def _uniform(low, high):
    # Augmentations draw from torch's generator, so a resumed training run
    # repeats them.
    return low + (high - low) * float(torch.rand(()))


def _chance(p):
    return float(torch.rand(())) < p


def _randint(low, high):
    return int(torch.randint(low, high + 1, ()))


def _keep_peak(audio, original):
    # Scales the augmented audio back to the original peak, so it never
    # clips and the volume augmentation still sets the level.
    peak = np.abs(audio).max()
    if peak == 0:
        return audio.astype(np.float32)
    return (audio * (np.abs(original).max() / peak)).astype(np.float32)


class NoiseSource:
    def __init__(self, source):
        """
        Recordings to mix into training audio: a directory of audio files,
        which are kept in memory, or a dataset json, whose files are read
        when they are picked, such as speech for babble.
        """
        if os.path.isdir(source):
            self.paths = sorted(
                p
                for ext in ("wav", "flac")
                for p in glob.glob(os.path.join(source, f"*.{ext}"))
            )
            self.cache = {}
        else:
            self.paths = [d["audio"] for d in read_data_json(source)]
            self.cache = None
        if not self.paths:
            raise ValueError(f"No audio files in {source}")

    def segment(self, frames, sample_rate):
        """
        Returns a random segment of a random recording as float samples
        at the sample rate, looping recordings that are shorter.
        """
        path = self.paths[_randint(0, len(self.paths) - 1)]
        audio = self.cache.get(path) if self.cache is not None else None
        if audio is None:
            audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if rate != sample_rate:
                g = math.gcd(rate, sample_rate)
                audio = scipy.signal.resample_poly(audio, sample_rate // g, rate // g)
            if self.cache is not None:
                self.cache[path] = audio
        if len(audio) < frames:
            audio = np.tile(audio, math.ceil(frames / len(audio)))
        start = _randint(0, len(audio) - frames)
        return audio[start : start + frames]


# The noise sources of this process, loaded once each.
_noise_sources = {}


def noise(audio, source, snr, p=1.0, layers=(1, 1), sample_rate=16000):
    """
    Mixes random segments of recordings into the audio at a random
    signal-to-noise ratio, keeping its peak level. DeepSpeech mixed
    background noise into 90% of its examples, and 8 to 12 layers of
    speech, as babble, into 10%, both at 8 to 16 dB.

    Arguments:
        audio (ndarray): The samples to add noise to.
        source (str): A directory of recordings or a dataset json, see
            NoiseSource.
        snr (list): The range of signal-to-noise ratios in dB, as the
            ratio of the powers of the audio and the added noise.
            DeepSpeech compared peak levels instead, so the same numbers
            gave it louder noise.
        p (float): The probability of adding noise.
        layers (list): The range of the number of recordings to mix,
            each at the same power.
        sample_rate (int): The sample rate of the audio.
    """
    power = np.mean(np.square(audio, dtype=np.float64))
    if power == 0 or not _chance(p):
        return audio
    if source not in _noise_sources:
        _noise_sources[source] = NoiseSource(source)
    mix = np.zeros(len(audio))
    for _ in range(_randint(*layers)):
        layer = _noise_sources[source].segment(len(audio), sample_rate)
        layer_power = np.mean(np.square(layer, dtype=np.float64))
        if layer_power > 0:
            mix += layer / math.sqrt(layer_power)
    mix_power = np.mean(np.square(mix))
    if mix_power == 0:
        return audio
    gain = math.sqrt(power / mix_power / 10 ** (_uniform(*snr) / 10))
    return _keep_peak(audio + gain * mix, audio)


def reverb(audio, delay, decay, p=1.0, sample_rate=16000):
    """
    Adds reverberation as DeepSpeech did, with echoes from five comb
    filters whose delays are the base delay times 17/17, 19/17, 23/17,
    29/17 and 31/17, prime ratios so that they do not reinforce each
    other, keeping the peak level. DeepSpeech applied it to 20% of its
    examples with a delay of 2 to 18 ms and a decay of 0.55 to 0.85 dB.

    Arguments:
        audio (ndarray): The samples to reverberate.
        delay (list): The range of the base delay in milliseconds.
        decay (list): The range of the decay of each echo in dB.
        p (float): The probability of adding reverberation.
        sample_rate (int): The sample rate of the audio.
    """
    if not _chance(p) or not np.any(audio):
        return audio
    delay_ms = _uniform(*delay)
    gain = 10 ** (-_uniform(*decay) / 20)
    signal = np.asarray(audio, dtype=np.float64)
    result = signal.copy()
    primes = (17, 19, 23, 29, 31)
    for prime in primes:
        frames = max(16, math.floor(delay_ms * prime / primes[0] * sample_rate / 1000))
        # Each echo repeats the output a delay later: y[n] = x[n] + g y[n - d].
        feedback = np.zeros(frames + 1)
        feedback[0] = 1
        feedback[frames] = -gain
        result += scipy.signal.lfilter([1.0], feedback, signal)
    return _keep_peak(result, signal)


def volume(audio, dbfs, p=1.0):
    """
    Scales 16 bit audio so its peak is at a random level, clipping what
    exceeds full scale. DeepSpeech used this with p 0.2.

    Arguments:
        audio (ndarray): 16 bit samples.
        dbfs (list): The range of peak levels in dBFS, where 0 is full
            scale. [-13, 7] matches DeepSpeech's -10 to 10, which counts a
            full-scale peak as 3 dBFS.
        p (float): The probability of changing the volume.
    """
    peak = np.abs(audio).max()
    if peak == 0 or not _chance(p):
        return audio
    level = _uniform(*dbfs)
    gain = 32768 * 10 ** (level / 20) / peak
    return np.clip(audio * gain, -32768, 32767).astype(np.float32)


def pitch(features, factor, p=1.0):
    """
    Changes the pitch by stretching a log spectrogram with shape (time,
    freq) along its frequency axis by a random factor. Bins emptied by a
    lower pitch get the spectrogram's lowest value, as silence. DeepSpeech
    used factors from 0.9 to 1.1 on every example.
    """
    if not _chance(p):
        return features
    bins = features.shape[1]
    stretched = scipy.ndimage.zoom(features, (1, _uniform(*factor)), order=1)
    if stretched.shape[1] >= bins:
        return stretched[:, :bins]
    pad = np.full(
        (len(features), bins - stretched.shape[1]), features.min(), features.dtype
    )
    return np.concatenate([stretched, pad], axis=1)


def tempo(features, factor, p=1.0):
    """
    Changes the tempo by stretching a spectrogram with shape (time, freq)
    along its time axis, where a factor above 1 is faster speech with
    fewer frames. DeepSpeech used factors from 0.9 to 1.1 on every example.
    """
    if not _chance(p):
        return features
    frames = max(1, round(len(features) / _uniform(*factor)))
    return scipy.ndimage.zoom(features, (frames / len(features), 1), order=1)


def spec_augment(
    features, freq_masks, freq_width, time_masks, time_width, time_ratio=1.0
):
    """
    SpecAugment (Park et al., 2019): masks random bands of frequencies and
    spans of time with zeros, the mean of the normalized features. The
    random choices come from torch's generator, so a resumed training run
    repeats them.

    Arguments:
        features (ndarray): Normalized features with shape (time, freq).
        freq_masks (int): The number of frequency bands to mask.
        freq_width (int): The largest width of a band, in bins.
        time_masks (int): The number of time spans to mask.
        time_width (int): The longest span, in frames.
        time_ratio (float): Caps the longest span at this share of the
            frames, so short utterances keep most of their audio.

    Returns a masked copy of the features.
    """

    def randint(high):
        return int(torch.randint(high + 1, ()))

    features = features.copy()
    frames, bins = features.shape
    for _ in range(freq_masks):
        width = randint(min(freq_width, bins))
        start = randint(bins - width)
        features[:, start : start + width] = 0
    longest = min(time_width, int(time_ratio * frames))
    for _ in range(time_masks):
        width = randint(longest)
        start = randint(frames - width)
        features[start : start + width] = 0
    return features


class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size) for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        # Shuffle a copy, so each epoch's order depends only on the random
        # state and a resumed run repeats it.
        batches = list(self.batches)
        random.shuffle(batches)
        return (i for b in batches for i in b)

    def __len__(self):
        return len(self.data_source)


def collate(batch):
    """
    Turns a list of (inputs, labels) examples into an (inputs, labels)
    pair of tuples. Defined at module level so worker processes can
    pickle it.
    """
    return tuple(zip(*batch))


def make_loader(dataset_json, preproc, batch_size, num_workers=4, augment=None):
    dataset = AudioDataset(dataset_json, preproc, batch_size, augment)
    sampler = BatchRandomSampler(dataset, batch_size)
    loader = tud.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )
    return loader


def log_specgram_from_file(audio_file):
    audio, sr = wave.array_from_wave(audio_file)
    return log_specgram(audio, sr)


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int(step_size * sample_rate / 1e3)
    _, _, spec = scipy.signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )
    return np.log(spec.T.astype(np.float32) + eps)


def read_data_json(data_json):
    """
    Reads a dataset json file, one example per line, or a list of them,
    which are concatenated.
    """
    paths = [data_json] if isinstance(data_json, str) else data_json
    data = []
    for path in paths:
        with open(path) as fid:
            data.extend(json.loads(l) for l in fid)
    return data


class SpecgramStream:
    def __init__(self, sample_rate, window_size=20, step_size=10):
        """
        Computes log_specgram incrementally as audio arrives. The frames
        returned for consecutive chunks are the frames log_specgram
        returns for the whole audio.
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.step_size = step_size
        self.nperseg = int(window_size * sample_rate / 1e3)
        self.hop = self.nperseg - int(step_size * sample_rate / 1e3)
        self.buffer = None

    def push(self, audio):
        """
        Adds the next chunk of audio samples and returns the new frames
        with shape (time, freq).
        """
        if self.buffer is not None:
            audio = np.concatenate([self.buffer, audio])
        n = max((len(audio) - self.nperseg) // self.hop + 1, 0)
        self.buffer = audio[n * self.hop :]
        if n == 0:
            return np.zeros((0, self.nperseg // 2 + 1), dtype=np.float32)
        audio = audio[: (n - 1) * self.hop + self.nperseg]
        return log_specgram(audio, self.sample_rate, self.window_size, self.step_size)
