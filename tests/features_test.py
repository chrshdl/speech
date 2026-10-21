import numpy as np
import pytest
import scipy.signal

from speech import features
from speech.utils import wave


def scipy_log_specgram(audio, sample_rate, eps=1e-10):
    """log_specgram as it was computed with SciPy, which the models were trained on."""
    nperseg = int(features.WINDOW_MS * sample_rate / 1e3)
    noverlap = int(features.STEP_MS * sample_rate / 1e3)
    _, _, spec = scipy.signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )
    return np.log(spec.T.astype(np.float32) + eps)


def assert_matches_scipy(audio, sample_rate):
    ours = features.log_specgram(audio, sample_rate)
    theirs = scipy_log_specgram(audio, sample_rate)
    assert ours.dtype == theirs.dtype == np.float32
    assert ours.shape == theirs.shape
    # Both FFTs round in single precision for int16 and float32 audio, each
    # its own way, so compare the power against each frame's loudest bin:
    # the log of a bin far quieter than that is rounding noise in both.
    ours, theirs = np.exp(ours.astype(np.float64)), np.exp(theirs.astype(np.float64))
    peak = theirs.max(axis=1, keepdims=True)
    assert (np.abs(ours - theirs) / peak).max() < 1e-5


@pytest.mark.parametrize("sample_rate", [16000, 8000, 11025])
@pytest.mark.parametrize("length", [321, 479, 480, 16000, 16123])
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_log_specgram_matches_scipy(sample_rate, length, dtype):
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.3, length)
    if dtype is np.int16:
        audio = (audio * 3000).astype(np.int16)
    assert_matches_scipy(audio.astype(dtype), sample_rate)


@pytest.mark.parametrize("wav", ["test0.wav", "test1.wav"])
def test_log_specgram_matches_scipy_on_speech(wav):
    assert_matches_scipy(*wave.array_from_wave(wav))


def test_log_specgram_shorter_than_a_window():
    spec = features.log_specgram(np.zeros(100, dtype=np.int16), 16000)
    assert spec.shape == (0, 161)
    assert spec.dtype == np.float32
