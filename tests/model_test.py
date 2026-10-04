import numpy as np
import pytest
import shared
import torch

import speech.models
from speech.models.model import zero_pad_concat


def test_model():
    time_steps = 100
    freq_dim = 40
    batch_size = 4

    model = speech.models.Model(freq_dim, shared.model_config)

    x = torch.randn(batch_size, time_steps, freq_dim)

    x_enc = model.encode(x)
    t_dim = model.conv_out_size(time_steps, 0)
    expected_size = torch.Size((batch_size, t_dim, model.encoder_dim))

    # Check output size is correct.
    assert x_enc.size() == expected_size

    # Check the device attribute works
    assert model.device.type == "cpu"
    if torch.cuda.is_available():
        model.cuda()
        assert model.device.type == "cuda"


def streaming_config(lookahead):
    return {
        "dropout": 0.0,
        "encoder": {
            "conv": [[8, 5, 11, 2], [8, 3, 5, 2]],
            "rnn": {"dim": 16, "bidirectional": False, "layers": 2},
            "lookahead": lookahead,
        },
    }


def test_lookahead():
    dim, context = 3, 2
    layer = speech.models.model.Lookahead(dim, context)
    x = torch.randn(2, 7, dim)

    # The layer starts as the identity.
    assert torch.equal(layer(x), x)

    torch.nn.init.normal_(layer.conv.weight)
    out = layer(x)
    assert out.size() == x.size()

    # Each output frame is a per-feature weighting of the current
    # and next `context` frames, with zeros past the end.
    w = layer.conv.weight.squeeze(1)
    padded = torch.cat([x, torch.zeros(2, context, dim)], dim=1)
    for t in range(x.size(1)):
        expected = (padded[:, t : t + context + 1] * w.t()).sum(dim=1)
        assert torch.allclose(out[:, t], expected, atol=1e-6)


def test_lookahead_requires_unidirectional():
    config = streaming_config(lookahead=2)
    config["encoder"]["rnn"]["bidirectional"] = True
    with pytest.raises(AssertionError):
        speech.models.Model(40, config)


@pytest.mark.parametrize("lookahead", [0, 1, 4])
def test_encode_stream(lookahead):
    torch.manual_seed(0)
    freq_dim, time_steps = 40, 75
    model = speech.models.Model(freq_dim, streaming_config(lookahead))
    shared.randomize_lookahead(model)
    model.set_eval()
    x = torch.randn(2, time_steps, freq_dim)

    with torch.no_grad():
        full = model.encode(x)

        # Stream random chunk sizes, including empty and single frames.
        rng = np.random.RandomState(0)
        outs = []
        state = None
        start = 0
        while start < time_steps:
            end = min(start + rng.randint(0, 9), time_steps)
            out, state = model.encode_stream(x[:, start:end], state)
            outs.append(out)
            start = end

            # Only the lookahead frames are held back from the output.
            emitted = sum(o.size(1) for o in outs)
            # The convolutions need 9 input frames for their first output.
            available = model.conv_out_size(end, 0) if end >= 9 else 0
            assert emitted == max(available - lookahead, 0)

        out, state = model.encode_stream(x[:, :0], state, final=True)
        outs.append(out)

    streamed = torch.cat(outs, dim=1)
    assert streamed.size() == full.size()
    assert torch.allclose(streamed, full, atol=1e-5)


@pytest.mark.parametrize(
    "bidirectional, lookahead", [(False, 0), (False, 3), (True, 0)]
)
def test_encode_ignores_padding(bidirectional, lookahead):
    torch.manual_seed(0)
    np.random.seed(0)
    freq_dim = 40
    config = streaming_config(lookahead)
    config["encoder"]["rnn"]["bidirectional"] = bidirectional
    model = speech.models.Model(freq_dim, config)
    shared.randomize_lookahead(model)
    model.set_eval()

    inputs, _ = shared.gen_padded_data(freq_dim, 1)
    lengths = [len(i) for i in inputs]
    x = torch.from_numpy(zero_pad_concat(inputs))
    enc_lens = model.encoded_lengths(lengths)

    with torch.no_grad():
        batched = model.encode(x, lengths)
        for i, inp in enumerate(inputs):
            single = model.encode(torch.from_numpy(inp).float().unsqueeze(0))
            n = enc_lens[i]
            assert single.size(1) == n
            assert torch.allclose(batched[i, :n], single[0], atol=1e-5)
            assert (batched[i, n:] == 0).all()
