import copy
import subprocess
import sys

import numpy as np
import pytest
import shared
import torch

from speech import export, loader, streaming
from speech.features import log_specgram
from speech.models import CTC
from speech.models.ctc_decoder import PRUNE, BeamSearch
from speech.models.word_lm import WordLM
from speech.onnx_model import OnnxModel
from speech.torch_model import TorchModel
from speech.utils import wave


def deep_config():
    """Two strided convolutions, two GRU layers and a lookahead."""
    config = copy.deepcopy(shared.model_config)
    config["encoder"]["conv"] = [[4, 5, 32, 2], [3, 3, 16, 3]]
    config["encoder"]["rnn"]["layers"] = 2
    config["encoder"]["lookahead"] = 3
    return config


CONFIGS = {
    "plain": shared.model_config,
    "lookahead": shared.model_config
    | {"encoder": shared.model_config["encoder"] | {"lookahead": 2}},
    "deep": deep_config(),
    "batch_norm": shared.batch_norm_config(deep_config()),
}


def make_model(config):
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    model = CTC(preproc.input_dim, preproc.vocab_size, config)
    shared.randomize_lookahead(model)
    shared.randomize_batch_norm(model)
    model.set_eval()
    return model, preproc


def stream_chunks(model, features, sizes):
    """Streams features in chunks of the given sizes, then flushes."""
    state, probs, start = None, [], 0
    for size in sizes:
        p, state = model.stream(features[start : start + size], state)
        probs.append(p)
        start += size
    p, _ = model.stream(features[start:], state, final=True)
    return np.concatenate(probs + [p])


@pytest.mark.parametrize("name", CONFIGS)
def test_export_matches_torch(name, tmp_path):
    model, preproc = make_model(CONFIGS[name])
    audio, sample_rate = wave.array_from_wave("test0.wav")
    features = log_specgram(audio, sample_rate)
    expected = TorchModel(model, preproc).probs(features)

    path = str(tmp_path / "model.onnx")
    export.export(model, preproc, path, int8=False)
    onnx_model = OnnxModel(path, threads=1)

    # Random chunk sizes, including empty ones and ones too short to give
    # an encoder frame, stream as the whole input does.
    rng = np.random.RandomState(0)
    sizes = list(rng.randint(0, 12, size=len(features) // 6))
    probs = stream_chunks(onnx_model, features, sizes)
    assert probs.shape == expected.shape
    assert np.allclose(probs, expected, atol=1e-5)
    assert np.allclose(onnx_model.probs(features), expected, atol=1e-5)

    # int8 weights stay close.
    export.export(model, preproc, path, int8=True)
    probs = OnnxModel(path).probs(features)
    assert np.abs(probs - expected).max() < 0.05


@pytest.mark.parametrize("name", CONFIGS)
def test_flush_without_new_frames(name, tmp_path):
    # A stream can end with no audio left, as stream.py ends one, which
    # gives the frames that wait for their lookahead.
    model, preproc = make_model(CONFIGS[name])
    audio, sample_rate = wave.array_from_wave("test0.wav")
    features = log_specgram(audio, sample_rate)
    expected = TorchModel(model, preproc).probs(features)

    path = str(tmp_path / "model.onnx")
    export.export(model, preproc, path, int8=False)
    onnx_model = OnnxModel(path, threads=1)
    probs, state = onnx_model.stream(features)
    rest, _ = onnx_model.stream(features[:0], state, final=True)
    context = model.lookahead.context if model.lookahead is not None else 0
    assert len(rest) == context
    assert np.allclose(np.concatenate([probs, rest]), expected, atol=1e-5)


@pytest.mark.parametrize("name", CONFIGS)
def test_receptive_field(name):
    model, _ = make_model(CONFIGS[name])
    stride, field = export.receptive_field(model)
    for frames in range(field, field + 20):
        expected = model.conv_out_size(frames, 0)
        assert (frames - field) // stride + 1 == expected


def test_metadata(tmp_path):
    model, preproc = make_model(CONFIGS["lookahead"])
    path = str(tmp_path / "model.onnx")
    export.export(model, preproc, path)
    onnx_model = OnnxModel(path)
    torch_model = TorchModel(model, preproc)

    assert onnx_model.int8
    for attr in ("blank", "sample_rate", "frame_seconds", "char_to_int"):
        assert getattr(onnx_model, attr) == getattr(torch_model, attr)
    labels = [3, 0, 7, 1]
    assert onnx_model.decode(labels) == torch_model.decode(labels)


def test_decode_start_and_end(tmp_path):
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=True)
    model = CTC(preproc.input_dim, preproc.vocab_size, shared.model_config)
    path = str(tmp_path / "model.onnx")
    export.export(model, preproc, path)
    onnx_model = OnnxModel(path)
    start, end = preproc.char_to_int["<s>"], preproc.char_to_int["</s>"]
    for labels in ([start, 3, 4, end], [3, 4], [start, 3, end, 4, end], [3, end, 4]):
        assert onnx_model.decode(labels) == "".join(preproc.decode(labels))


def test_bigrams(tmp_path):
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False, bigrams=True)
    model = CTC(preproc.input_dim, preproc.vocab_size, shared.model_config)
    path = str(tmp_path / "model.onnx")
    export.export(model, preproc, path)
    onnx_model = OnnxModel(path)

    assert onnx_model.char_to_int == preproc.char_to_int
    labels = preproc.encode("hello world")
    assert onnx_model.decode(labels) == "hello world"
    assert TorchModel(model, preproc).decode(labels) == "hello world"


def test_export_rejects_bidirectional(tmp_path):
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    model = CTC(preproc.input_dim, preproc.vocab_size, shared.bidirectional_config())
    with pytest.raises(ValueError, match="bidirectional"):
        export.export(model, preproc, str(tmp_path / "model.onnx"))


def test_transcriber(tmp_path):
    model, preproc = make_model(CONFIGS["lookahead"])
    path = str(tmp_path / "model.onnx")
    export.export(model, preproc, path, int8=False)
    lm = WordLM(["hello world", "hello hi", "world"]).scorer(preproc.char_to_int)
    audio, _ = wave.array_from_wave("test1.wav")

    def transcribe(m, search):
        transcriber = streaming.Transcriber(m, search)
        for start in range(0, len(audio), 1600):
            transcriber.push(audio[start : start + 1600])
        transcriber.push(audio[:0], final=True)
        return transcriber.labels(final=True)

    torch_model = TorchModel(model, preproc)
    onnx_model = streaming.load(path)
    assert transcribe(onnx_model, None) == transcribe(torch_model, None)
    with_lm = transcribe(onnx_model, BeamSearch(model.blank, 8, lm, PRUNE))
    assert with_lm == transcribe(torch_model, BeamSearch(model.blank, 8, lm, PRUNE))


def test_runs_without_torch(tmp_path):
    model, preproc = make_model(CONFIGS["lookahead"])
    path = str(tmp_path / "model.onnx")
    export.export(model, preproc, path)

    # Streaming an exported model never imports PyTorch.
    code = f"""
import sys
from speech import streaming
from speech.utils import wave

model = streaming.load({path!r})
search = streaming.search_factory(model.blank, model.char_to_int)()
transcriber = streaming.Transcriber(model, search)
audio, _ = wave.array_from_wave("test0.wav")
transcriber.push(audio, final=True)
print(transcriber.text())
assert "torch" not in sys.modules, "imported torch"
"""
    subprocess.run([sys.executable, "-c", code], check=True)
