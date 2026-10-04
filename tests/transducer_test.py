import itertools

import numpy as np
import shared
import torch

from speech.models import Transducer


def make_model(freq_dim, vocab_size, config=None):
    conf = config or shared.model_config
    conf = dict(conf, decoder={"embedding_dim": 8, "layers": 1})
    return Transducer(freq_dim, vocab_size, conf)


def test_model():
    freq_dim = 40
    vocab_size = 10

    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(freq_dim, vocab_size)
    batch = shared.gen_fake_data(freq_dim, vocab_size)
    batch_size = len(batch[0])

    out = model(batch)
    time_steps = model.conv_out_size(batch[0][0].shape[0], 0)
    label_len = len(batch[1][0])
    assert out.size() == (batch_size, time_steps, label_len + 1, vocab_size + 1)

    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())

    preds = model.infer(batch)
    assert len(preds) == batch_size
    assert all(0 <= p < vocab_size for pred in preds for p in pred)


def brute_force_nll(log_probs, label, blank):
    # Sums the probability of every path through the (time, label)
    # lattice. A path emits the labels in order and a blank at each
    # frame, ending with the blank at the last frame.
    time_steps = log_probs.shape[0]
    steps = time_steps + len(label)
    total = -np.inf
    for emit_steps in itertools.combinations(range(steps - 1), len(label)):
        t = u = 0
        score = 0.0
        for step in range(steps):
            if step in emit_steps:
                score += log_probs[t, u, label[u]]
                u += 1
            else:
                score += log_probs[t, u, blank]
                t += 1
        total = np.logaddexp(total, score)
    return -total


def test_loss():
    freq_dim = 40
    vocab_size = 3

    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(freq_dim, vocab_size)
    inputs, labels = shared.gen_padded_data(
        freq_dim, vocab_size, time_steps=(14, 10), label_lens=(3, 2)
    )
    batch = (inputs, labels)

    with torch.no_grad():
        loss = model.loss(batch)
        log_probs = torch.log_softmax(model(batch), dim=3).numpy()

    # The blank is the last class and the loss is averaged over the batch.
    lens = model.encoded_lengths([len(i) for i in inputs])
    expected = np.mean(
        [
            brute_force_nll(lp[:n, : len(label) + 1], label, model.blank)
            for lp, n, label in zip(log_probs, lens, labels)
        ]
    )
    assert np.isclose(loss.item(), expected, rtol=1e-4)


def test_padding():
    freq_dim = 40
    vocab_size = 10

    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(freq_dim, vocab_size, shared.bidirectional_config())
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


def test_infer_ignores_labels():
    freq_dim = 40
    vocab_size = 10

    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(freq_dim, vocab_size)
    inputs, labels = shared.gen_fake_data(freq_dim, vocab_size)
    other_labels = [np.random.randint(0, vocab_size, 5) for _ in labels]

    # Decoding must not see the reference labels.
    assert model.infer((inputs, labels)) == model.infer((inputs, other_labels))


def test_greedy_decode():
    freq_dim = 40
    vocab_size = 4

    torch.manual_seed(0)
    model = make_model(freq_dim, vocab_size)
    x = torch.randn(5, model.encoder_dim)

    # Always predicting label 2 emits max_symbols labels per frame.
    with torch.no_grad():
        model.fc2.bias.fill_(-100)
        model.fc2.bias[2] = 100
    assert model.greedy_decode(x, max_symbols=3) == [2] * 15

    # Always predicting the blank emits nothing.
    with torch.no_grad():
        model.fc2.bias[2] = -100
        model.fc2.bias[model.blank] = 100
    assert model.greedy_decode(x, max_symbols=3) == []
