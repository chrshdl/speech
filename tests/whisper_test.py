import numpy as np
import pytest

from speech import whisper_model
from speech.whisper_model import WhisperModel, WhisperTranscriber


def test_mel_filters():
    filters = whisper_model.mel_filters(80)
    assert filters.shape == (80, 201)
    assert (filters >= 0).all()
    # Each filter is a triangle over a few frequency bins, higher ones wider.
    widths = (filters > 0).sum(axis=1)
    assert widths[0] <= widths[-1]


def test_log_mel():
    rng = np.random.default_rng(0)
    features = whisper_model.log_mel(
        rng.standard_normal(16000).astype(np.float32) * 0.1,
        whisper_model.mel_filters(80),
    )
    # Padded to 30 seconds of 10 ms frames, within 8 of the loudest, so
    # within 2 after scaling.
    assert features.shape == (80, 3000)
    assert features.max() - features.min() <= 2.0 + 1e-6


class FakeMeta:
    def __init__(self, metadata):
        self.custom_metadata_map = metadata


class FakeSession:
    """Returns the prefix it is given, then fixed tokens."""

    def __init__(self, tokens, metadata):
        self.tokens = tokens
        self.metadata = metadata

    def get_modelmeta(self):
        return FakeMeta(self.metadata)

    def run(self, outputs, feeds):
        assert feeds["input_features"].shape == (1, 80, 3000)
        prefix = feeds["decoder_input_ids"][0].tolist()
        return [np.array([[prefix + self.tokens]], np.int32)]


METADATA = {
    "model_type": "whisper",
    "format_version": "1",
    # GPT-2's vocabulary spells a space as Ġ.
    "vocabulary": '["Hello", "\\u0120w\\u00c3\\u00b6rld", "!", "<|endoftext|>"]',
    "end_of_text": "3",
    "prefix": "[3, 3]",
    "n_mels": "80",
}


def test_transcribe():
    session = FakeSession([0, 1, 2, 3, 0], METADATA)
    assert whisper_model.is_whisper(session)
    model = WhisperModel(session)
    # The prefix is dropped and the text ends at the end of text token.
    assert model.transcribe(np.zeros(16000, np.int16)) == "Hello wörld!"


def test_other_version():
    with pytest.raises(ValueError, match="version"):
        WhisperModel(FakeSession([], METADATA | {"format_version": "0"}))


class Silent:
    sample_rate = 16000

    def transcribe(self, audio):
        raise AssertionError("Nothing was said.")


def test_transcriber_finds_speech_by_loudness():
    transcriber = WhisperTranscriber(Silent())
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000) * 30).astype(np.int16)
    transcriber.push(noise)
    assert not transcriber.heard_speech
    # Silence alone is never transcribed.
    transcriber.push(np.zeros(0, np.int16), final=True)
    assert transcriber.text(final=True) == ""

    t = np.arange(8000) / 16000
    tone = (8000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    transcriber.push(tone)
    assert transcriber.heard_speech
    assert transcriber.pause_seconds < 0.1
    transcriber.push(noise)
    assert transcriber.pause_seconds == pytest.approx(0.99, abs=0.05)
