import json
import random

import numpy as np
import scipy.ndimage
import scipy.signal
import torch
import torch.utils.data as tud

from speech.utils import wave

# The training augmentations a config's data can set, in the order they
# are applied: volume on the audio, pitch and tempo on the spectrogram,
# then spec_augment on the normalized features.
AUGMENTATIONS = ("volume", "pitch", "tempo", "spec_augment")


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
                as it is loaded.
        """

        data = read_data_json(data_json)
        self.preproc = preproc
        self.augment = augment

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

    def __getitem__(self, idx):
        datum = self.data[idx]
        if not self.augment:
            return self.preproc.preprocess(datum["audio"], datum["text"])

        augment = self.augment
        audio, sample_rate = wave.array_from_wave(datum["audio"])
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
