import numpy as np

from speech.utils import wave


def test_load():
    audio, samp_rate = wave.array_from_wave("test0.wav")

    assert samp_rate == 16000
    assert audio.dtype == np.int16


def test_duration():
    duration = wave.wav_duration("test0.wav")

    assert round(duration, 3) == 1.101
