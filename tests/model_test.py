import copy

import numpy as np
import pytest
import shared
import torch

import speech.models
from speech.models.model import BatchNormGRU, ConvBatchNorm, zero_pad_concat


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


@pytest.mark.parametrize(
    "lookahead, batch_norm", [(0, False), (1, False), (4, False), (2, True)]
)
def test_encode_stream(lookahead, batch_norm):
    torch.manual_seed(0)
    freq_dim, time_steps = 40, 75
    config = streaming_config(lookahead)
    if batch_norm:
        config = shared.batch_norm_config(config)
    model = speech.models.Model(freq_dim, config)
    shared.randomize_lookahead(model)
    shared.randomize_batch_norm(model)
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
    "bidirectional, lookahead, batch_norm",
    [(False, 0, False), (False, 3, False), (True, 0, False), (False, 3, True)],
)
def test_encode_ignores_padding(bidirectional, lookahead, batch_norm):
    torch.manual_seed(0)
    np.random.seed(0)
    freq_dim = 40
    config = streaming_config(lookahead)
    config["encoder"]["rnn"]["bidirectional"] = bidirectional
    if batch_norm:
        config = shared.batch_norm_config(config)
    model = speech.models.Model(freq_dim, config)
    shared.randomize_lookahead(model)
    shared.randomize_batch_norm(model)
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


def test_config_without_batch_norm():
    # Models saved before batch normalization keep their modules and
    # parameter names, so their checkpoints still load.
    model = speech.models.Model(40, streaming_config(0))
    assert type(model.rnn) is torch.nn.GRU
    names = set(model.state_dict())
    assert {"conv.0.weight", "conv.0.bias", "rnn.weight_ih_l1"} <= names
    assert not any("norm" in n or "running" in n for n in names)


def test_batch_norm_modules():
    model = speech.models.Model(40, shared.batch_norm_config(streaming_config(0)))
    convs = [m for m in model.conv if isinstance(m, torch.nn.Conv2d)]
    assert all(c.bias is None for c in convs)
    norms = [m for m in model.conv if isinstance(m, ConvBatchNorm)]
    assert len(norms) == len(convs)
    clips = [m for m in model.conv if isinstance(m, torch.nn.Hardtanh)]
    assert [(c.min_val, c.max_val) for c in clips] == [(0, 20)] * len(convs)
    assert isinstance(model.rnn, BatchNormGRU)
    assert len(model.rnn.layers) == 2 and not model.rnn.bidirectional


@pytest.mark.parametrize("momentum", [0.1, None])
def test_conv_batch_norm_skips_padding(momentum):
    # In training, the statistics are those of the valid frames alone:
    # the same as nn.BatchNorm2d on the examples' valid frames side by
    # side, over several batches, with a momentum or a plain average.
    torch.manual_seed(0)
    channels, freq = 3, 5
    lengths = [7, 4]
    norm = ConvBatchNorm(channels, momentum=momentum)
    torch.nn.init.uniform_(norm.weight, 0.5, 2.0)
    torch.nn.init.normal_(norm.bias)
    reference = torch.nn.BatchNorm2d(channels, momentum=momentum)
    reference.load_state_dict(norm.state_dict())

    for _ in range(3):
        x = torch.randn(2, channels, 9, freq)
        out = norm(x, lengths)
        valid = torch.cat([x[i : i + 1, :, :n] for i, n in enumerate(lengths)], dim=2)
        expected = reference(valid)
        start = 0
        for i, n in enumerate(lengths):
            assert torch.allclose(
                out[i, :, :n], expected[0, :, start : start + n], atol=1e-5
            )
            start += n
    assert torch.allclose(norm.running_mean, reference.running_mean, atol=1e-6)
    assert torch.allclose(norm.running_var, reference.running_var, atol=1e-6)
    assert norm.num_batches_tracked == reference.num_batches_tracked == 3


def test_batch_norm_training_ignores_padding():
    # Extra padding changes neither the encoding of the valid frames nor
    # the running statistics, in the convolutions or the RNN.
    torch.manual_seed(0)
    np.random.seed(0)
    freq_dim = 40
    model = speech.models.Model(freq_dim, shared.batch_norm_config(streaming_config(3)))
    shared.randomize_lookahead(model)
    model.train()
    inputs, _ = shared.gen_padded_data(freq_dim, 1)
    lengths = [len(i) for i in inputs]
    enc_lens = model.encoded_lengths(lengths)
    x = torch.from_numpy(zero_pad_concat(inputs))
    padded = torch.cat([x, torch.zeros(x.size(0), 30, freq_dim)], dim=1)

    results = []
    for batch in [x, padded]:
        torch.manual_seed(1)
        trained = copy.deepcopy(model)
        out = trained.encode(batch, lengths)
        stats = [b.clone() for n, b in trained.named_buffers() if "running" in n]
        results.append((out, stats))
    (out, stats), (out_padded, stats_padded) = results
    for i, n in enumerate(enc_lens):
        assert torch.allclose(out[i, :n], out_padded[i, :n], atol=1e-5)
    for a, b in zip(stats, stats_padded):
        assert torch.allclose(a, b, atol=1e-6)


def test_batch_norm_autocast():
    # Mixed precision runs the convolutions in bfloat16; the batch
    # normalization still computes in float32 and returns their dtype.
    torch.manual_seed(0)
    model = speech.models.Model(40, shared.batch_norm_config(streaming_config(0)))
    x = torch.randn(2, 30, 40)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        h = model.conv_forward(x.unsqueeze(1), [30, 20])
        out = model.encode(x, [30, 20])
    assert h.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
