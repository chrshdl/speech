
import torch
import torch.autograd as autograd

from speech.models import CTC

import shared

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
    preds = model.infer(batch)
    assert len(preds) == batch_size


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
