"""
The log spectrogram features of the models, computed for whole clips or
incrementally as audio arrives. Needs only NumPy and SciPy, so it also
runs where PyTorch is not installed.
"""

import numpy as np
import scipy.signal

# The window and step of the features in milliseconds.
WINDOW_MS = 20
STEP_MS = 10


def log_specgram(
    audio, sample_rate, window_size=WINDOW_MS, step_size=STEP_MS, eps=1e-10
):
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
