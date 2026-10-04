import json
import random

import numpy as np
import scipy.signal
import torch.utils.data as tud

from speech.utils import wave


class Preprocessor:
    END = "</s>"
    START = "<s>"

    def __init__(self, data_json, max_samples=100, start_and_end=True):
        """
        Builds a preprocessor from a dataset.
        Arguments:
            data_json (string): A file containing a json representation
                of each example per line.
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
        inputs = log_specgram_from_file(wave_file)
        inputs = (inputs - self.mean) / self.std
        targets = self.encode(text)
        return inputs, targets

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
    def __init__(self, data_json, preproc, batch_size):

        data = read_data_json(data_json)
        self.preproc = preproc

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
        datum = self.preproc.preprocess(datum["audio"], datum["text"])
        return datum


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
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


def collate(batch):
    """
    Turns a list of (inputs, labels) examples into an (inputs, labels)
    pair of tuples. Defined at module level so worker processes can
    pickle it.
    """
    return tuple(zip(*batch))


def make_loader(dataset_json, preproc, batch_size, num_workers=4):
    dataset = AudioDataset(dataset_json, preproc, batch_size)
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
    with open(data_json) as fid:
        return [json.loads(l) for l in fid]


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
