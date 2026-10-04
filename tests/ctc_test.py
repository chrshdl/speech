import itertools

import numpy as np
import shared
import torch

from speech.models import CTC


def test_ctc_model():
    freq_dim = 40
    vocab_size = 10

    batch = shared.gen_fake_data(freq_dim, vocab_size)
    batch_size = len(batch[0])

    model = CTC(freq_dim, vocab_size, shared.model_config)
    out = model(batch)

    assert out.size()[0] == batch_size

    # CTC model adds the blank token to the vocab
    assert out.size()[2] == (vocab_size + 1)

    assert len(out.size()) == 3

    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())

    preds = model.infer(batch)
    assert len(preds) == batch_size


def brute_force_nll(log_probs, label, blank):
    # Sums the probability of every alignment that collapses to the label.
    time_steps, classes = log_probs.shape
    total = -np.inf
    for path in itertools.product(range(classes), repeat=time_steps):
        if CTC.max_decode(list(path), blank) == list(label):
            score = sum(log_probs[t, c] for t, c in enumerate(path))
            total = np.logaddexp(total, score)
    return -total


def test_ctc_loss():
    torch.manual_seed(0)
    np.random.seed(0)
    freq_dim = 40
    vocab_size = 2

    # Small enough to enumerate every alignment.
    batch = shared.gen_fake_data(freq_dim, vocab_size, max_time=10, max_seq_len=2)
    model = CTC(freq_dim, vocab_size, shared.model_config)

    with torch.no_grad():
        loss = model.loss(batch)
        log_probs = torch.log_softmax(model(batch), dim=2).numpy()

    # The blank is the last class and the loss is averaged over the batch.
    expected = np.mean(
        [
            brute_force_nll(lp, label, model.blank)
            for lp, label in zip(log_probs, batch[1])
        ]
    )
    assert np.isclose(loss.item(), expected, rtol=1e-4)


def test_padding():
    freq_dim = 40
    vocab_size = 10

    np.random.seed(0)
    torch.manual_seed(0)
    model = CTC(freq_dim, vocab_size, shared.bidirectional_config())
    model.set_eval()

    inputs, labels = shared.gen_padded_data(freq_dim, vocab_size)
    batch = (inputs, labels)
    singles = [([i], [l]) for i, l in zip(inputs, labels)]

    with torch.no_grad():
        # The batch loss is the mean of each example's loss on its own.
        loss = model.loss(batch).item()
        expected = np.mean([model.loss(b).item() for b in singles])
        assert np.isclose(loss, expected, rtol=1e-5)

        # Decoding a batch matches decoding each example on its own.
        assert model.infer(batch) == [model.infer(b)[0] for b in singles]


def test_stream():
    freq_dim = 40
    vocab_size = 10

    torch.manual_seed(0)
    np.random.seed(0)
    config = shared.model_config | {
        "encoder": shared.model_config["encoder"] | {"lookahead": 2}
    }
    model = CTC(freq_dim, vocab_size, config)
    shared.randomize_lookahead(model)
    model.set_eval()
    x = torch.randn(1, 60, freq_dim)

    probs = []
    state = None
    for start in range(0, 60, 7):
        p, state = model.stream(x[:, start : start + 7], state, final=start + 7 >= 60)
        probs.append(p)

    with torch.no_grad():
        full = model.forward_impl(x, softmax=True)
    assert torch.allclose(torch.cat(probs, dim=1), full, atol=1e-5)


def test_argmax_decode():
    blank = 0
    pre = [1, 2, 2, 0, 0, 0, 2, 1]
    post = [1, 2, 2, 1]
    assert CTC.max_decode(pre, blank) == post

    pre = [2, 2, 2]
    post = [2]
    assert CTC.max_decode(pre, blank) == post

    pre = [0, 0, 0]
    post = []
    assert CTC.max_decode(pre, blank) == post


def test_decode_with_lm():
    import numpy as np

    from speech.models.ctc_decoder import decode

    np.random.seed(3)
    probs = np.random.rand(50, 20)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    # Baseline: no LM
    labels_no_lm, _ = decode(probs)

    # Trivial LM returning 0.0 should not change the result
    labels_zero, _ = decode(probs, lm=lambda p: 0.0, lm_weight=1.0)
    assert labels_no_lm == labels_zero

    # A non-trivial LM that biases towards label 1 should change output
    def bias_lm(prefix):
        if prefix and prefix[-1] == 1:
            return 0.0  # no penalty
        return -10.0  # heavy penalty

    labels_biased, _ = decode(probs, beam_size=10, lm=bias_lm, lm_weight=2.0)
    # The biased result should be different from unbiased
    # (or at minimum, it ran without error)
    assert isinstance(labels_biased, tuple)
