import numpy as np
import pytest
import torch
from backend_test import export_model

from speech import streaming, wake_word
from speech.models import ds_cnn
from speech.utils import wave

LABELS = ["silence", "unknown", "olivia"]


def random_model(path, seed=0):
    """Exports a DS-CNN with random weights and batch norm statistics."""
    torch.manual_seed(seed)
    model = ds_cnn.DSCNN(LABELS, "olivia", "S")
    model.set_normalization(torch.randn(10) * 5, torch.rand(10) * 5 + 1)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 2)
            module.weight.data.normal_()
            module.bias.data.normal_()
    model.eval()
    ds_cnn.export(model, 16000, str(path))
    return model


def speech_audio():
    audio, _ = wave.array_from_wave("test0.wav")
    return np.concatenate([audio, np.zeros(8000, np.int16), audio])


def test_mfcc_stream_is_the_whole_audio():
    audio = speech_audio()
    whole = wake_word.Mfcc(16000)(audio)
    stream = wake_word.MfccStream(16000)
    rng = np.random.default_rng(0)
    parts, start = [], 0
    while start < len(audio):
        size = int(rng.integers(0, 2500))
        parts.append(stream.push(audio[start : start + size]))
        start += size
    np.testing.assert_allclose(np.concatenate(parts), whole, rtol=1e-5, atol=1e-4)
    # 40 ms frames every 20 ms.
    assert whole.shape == ((len(audio) - 640) // 320 + 1, 10)


def test_mel_filterbank_and_dct():
    bank = wake_word.mel_filterbank(16000, 1024)
    assert bank.shape == (513, 40)
    peaks = bank.argmax(axis=0) * 16000 / 1024
    assert np.all(np.diff(peaks) > 0)
    assert 20 < peaks[0] < 100 and 3700 < peaks[-1] < 4000
    dct = wake_word.dct_matrix(40, 40)
    np.testing.assert_allclose(dct.T @ dct, np.eye(40), atol=1e-5)


def test_ring_buffer():
    ring = wake_word.RingBuffer(5)
    assert len(ring.read()) == 0
    ring.write(np.array([1, 2, 3]))
    assert ring.read().tolist() == [1, 2, 3]
    ring.write(np.array([4, 5, 6, 7]))
    assert ring.read().tolist() == [3, 4, 5, 6, 7]
    ring.write(np.arange(10, 22))
    assert ring.read().tolist() == [17, 18, 19, 20, 21]
    ring.write(np.array([], np.int16))
    assert ring.read().tolist() == [17, 18, 19, 20, 21]


def test_export_matches_pytorch(tmp_path):
    model = random_model(tmp_path / "model.onnx")
    onnx_model = wake_word.WakeWordModel(str(tmp_path / "model.onnx"))
    assert onnx_model.labels == LABELS
    assert onnx_model.keyword_index == 2
    assert onnx_model.clip_frames == 49
    mfcc = np.random.default_rng(0).normal(size=(4, 49, 10)).astype(np.float32) * 5
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(mfcc)), 1).numpy()
    np.testing.assert_allclose(onnx_model.probs(mfcc), expected, atol=1e-5)


def test_detector_decides_as_detections(tmp_path):
    """The Detector hears the wake word where detections does for the whole stream."""
    random_model(tmp_path / "model.onnx", seed=1)
    model = wake_word.WakeWordModel(str(tmp_path / "model.onnx"))
    audio = speech_audio()
    scores = model.scores(audio)
    threshold = float(np.quantile(scores, 0.2))
    blocked = wake_word.blocked_hops(model, wake_word.HOP_SECONDS, 3)
    assert blocked == 10 + 2
    expected = wake_word.detections(scores, threshold, 3, blocked)
    assert len(expected) >= 2

    # With chunks of one decision each, after the first second's.
    detector = wake_word.Detector(model, threshold, 3)
    first = (model.clip_frames - 1) * 320 + 640
    assert not detector.push(audio[: first - 1600])
    heard = [
        detector.push(audio[s : s + 1600])
        for s in range(first - 1600, len(audio), 1600)
    ]
    assert [i for i, h in enumerate(heard) if h] == expected

    # And with chunks of any size.
    detector = wake_word.Detector(model, threshold, 3)
    rng = np.random.default_rng(0)
    count, start = 0, 0
    while start < len(audio):
        size = int(rng.integers(0, 5000))
        count += detector.push(audio[start : start + size])
        start += size
    assert count == len(expected)


def test_detector_reset_skips_a_window(tmp_path):
    random_model(tmp_path / "model.onnx", seed=1)
    model = wake_word.WakeWordModel(str(tmp_path / "model.onnx"))
    audio = speech_audio()
    # Hears it in every window it can.
    detector = wake_word.Detector(model, threshold=0.0, smoothing=1)
    heard = [detector.push(audio[s : s + 1600]) for s in range(0, len(audio), 1600)]
    first = heard.index(True)
    assert heard[first + 1 : first + 11] == [False] * 10
    assert heard[first + 11]
    detector.reset()
    assert not any(detector.push(np.zeros(1600, np.int16)) for _ in range(10))
    assert detector.push(np.zeros(1600, np.int16))


def ctc_probs(path, blank, labels=3):
    """One-hot output probabilities of a path of labels, one per frame."""
    probs = np.full((len(path), labels + 1), 0.01, np.float32)
    probs[np.arange(len(path)), path] = 1 - 0.01 * labels
    return probs


def test_align():
    blank = 3
    # Something, the word [0, 1], then something else.
    probs = ctc_probs([2, 2, blank, 0, 0, blank, 1, blank, 2, blank], blank)
    score, start, end = wake_word.align(probs, np.array([0, 1]), blank)
    assert score == pytest.approx(0, abs=1e-6)
    assert (start, end) == (3, 7)
    # A word that is not said scores low.
    score, _, _ = wake_word.align(probs, np.array([1, 0]), blank)
    assert score < -2
    assert wake_word.align(probs[:3], np.array([0, 1]), blank) is None


def test_drop_starts_the_transcript_later(tmp_path):
    path = export_model(tmp_path / "model.onnx")
    model = streaming.load(path)
    audio, _ = wave.array_from_wave("test0.wav")
    transcriber = streaming.Transcriber(model)
    transcriber.push(audio)
    probs = transcriber.probs()
    transcriber.drop(20)
    np.testing.assert_array_equal(transcriber.probs(), probs[20:])
    fresh = streaming.Transcriber(model)
    fresh._decode(probs[20:])
    assert transcriber.text() == fresh.text()
    assert transcriber.pause_seconds == fresh.pause_seconds
    # The stream goes on.
    transcriber.push(audio[:16000])
    assert len(transcriber.probs()) > len(probs) - 20


def test_word_end():
    blank, space = 3, 2
    # The rest of a word, "r", then a pause.
    probs = ctc_probs([0, blank, 1, 1, blank, blank, blank, blank, 0], blank)
    assert wake_word.word_end(probs, 1, blank, space, 10) == 4
    # Up to the next space.
    probs = ctc_probs([0, 1, blank, space, 0, 1], blank)
    assert wake_word.word_end(probs, 1, blank, space, 10) == 2
    # Not beyond longest, and nothing when a word ends there.
    probs = ctc_probs([0, 1, 1, 1, 1, 1], blank)
    assert wake_word.word_end(probs, 1, blank, space, 3) == 4
    assert wake_word.word_end(ctc_probs([0, space, 1], blank), 1, blank, space, 3) == 1
