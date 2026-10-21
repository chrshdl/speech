"""
The log spectrogram features of the models, computed for whole clips or
incrementally as audio arrives. Needs only NumPy, so it also runs where
PyTorch and SciPy are not installed, and the backend starts without
importing scipy.signal, which takes seconds on a Raspberry Pi.
"""

import numpy as np

# The window and step of the features in milliseconds.
WINDOW_MS = 20
STEP_MS = 10


def log_specgram(
    audio, sample_rate, window_size=WINDOW_MS, step_size=STEP_MS, eps=1e-10
):
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int(step_size * sample_rate / 1e3)
    hop = nperseg - noverlap
    # What scipy.signal.spectrogram(audio, sample_rate, "hann", nperseg,
    # noverlap, detrend=False) computes, which the models were trained on:
    # the one-sided power spectral density of periodic Hann windows, in
    # single precision unless the audio needs double.
    audio = np.asarray(audio)
    dtype = np.result_type(audio, np.float32)
    if len(audio) < nperseg:
        return np.zeros((0, nperseg // 2 + 1), dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, nperseg)[::hop]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
    window = window.astype(dtype)
    spec = np.fft.rfft(frames.astype(dtype) * window, axis=1)
    spec = spec.real**2 + spec.imag**2
    spec *= 1 / (sample_rate * (window * window).sum())
    # Each bin but DC and, for an even nperseg, Nyquist also stands for
    # its negative frequency.
    spec[:, 1 : None if nperseg % 2 else -1] *= 2
    return np.log(spec.astype(np.float32) + eps)


def sample_rate_for(input_dim):
    """
    The sample rate whose features have input_dim frequency bins per
    frame.
    """
    return round((input_dim - 1) * 2 * 1000 / WINDOW_MS)


class SpecgramStream:
    def __init__(self, sample_rate, window_size=WINDOW_MS, step_size=STEP_MS):
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
