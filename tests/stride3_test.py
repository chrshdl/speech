import json
import os

import numpy as np
import shared
import torch

from speech import export, loader
from speech.models import CTC
from speech.torch_model import TorchModel

CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "commonvoice",
    "ctc_streaming_3750h_stride3_config.json",
)


def stride3_model(rnn_dim=None):
    with open(CONFIG) as fid:
        config = json.load(fid)
    if rnn_dim is not None:
        config["model"]["encoder"]["rnn"]["dim"] = rnn_dim
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    model = CTC(preproc.input_dim, preproc.vocab_size, config["model"])
    shared.randomize_lookahead(model)
    shared.randomize_batch_norm(model)
    model.set_eval()
    return model, preproc


def test_stride3_config():
    model, _ = stride3_model()
    # A frame every 30 ms, with a lookahead of 7 frames, 210 ms.
    assert export.receptive_field(model)[0] == 3
    assert model.lookahead.context == 7
    # The GRU gets the 21 frequency bins of 32 channels of the configs
    # that stride 4.
    assert model.rnn.layers[0].input_size == 32 * 21


def test_stride3_streams_as_a_whole():
    # A narrower GRU streams the same way, faster.
    model, preproc = stride3_model(rnn_dim=64)
    features = np.random.RandomState(0).randn(300, preproc.input_dim)
    features = features.astype(np.float32)
    torch_model = TorchModel(model, preproc)
    expected = torch_model.probs(features)
    state, probs, start = None, [], 0
    for size in [10] * 29:
        p, state = torch_model.stream(features[start : start + size], state)
        probs.append(p)
        start += size
    p, _ = torch_model.stream(features[start:], state, final=True)
    probs = np.concatenate(probs + [p])
    assert probs.shape == expected.shape
    assert len(probs) == model.conv_out_size(300, 0)
    assert np.allclose(probs, expected, atol=1e-5)
