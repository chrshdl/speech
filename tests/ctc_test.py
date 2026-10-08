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


def brute_force_decode(probs, blank, lm):
    # Scores every label sequence by summing the probabilities of all
    # alignments that collapse to it, then adds its LM score.
    log_probs = np.log(probs)
    totals = {}
    for path in itertools.product(range(probs.shape[1]), repeat=probs.shape[0]):
        labels = tuple(CTC.max_decode(list(path), blank))
        score = sum(log_probs[t, s] for t, s in enumerate(path))
        totals[labels] = np.logaddexp(totals.get(labels, -np.inf), score)
    for labels in totals:
        state, lm_score = lm.initial(), 0.0
        for label in labels:
            state, delta = lm.extend(state, label)
            lm_score += delta
        totals[labels] += lm_score + lm.finish(state)
    best = max(totals, key=totals.get)
    return best, totals[best]


def test_decode_with_lm():
    from speech.models.ctc_decoder import decode
    from speech.models.word_lm import WordLM

    lm = WordLM(["a b", "b a a", "a"])
    char_to_int = {"a": 0, "b": 1, " ": 2}
    blank = 3
    scorer = lm.scorer(char_to_int, weight=0.7, word_bonus=0.3)

    rng = np.random.RandomState(0)
    for _ in range(10):
        probs = rng.rand(6, 4) ** 3
        probs /= probs.sum(axis=1, keepdims=True)

        # A beam that keeps every prefix finds the exact best sequence.
        labels, nll = decode(probs, beam_size=10_000, blank=blank, lm=scorer)
        expected, score = brute_force_decode(probs, blank, scorer)
        assert labels == expected
        assert np.isclose(-nll, score)


def test_lm_changes_decoding():
    from speech.models.ctc_decoder import decode
    from speech.models.word_lm import WordLM

    chars = " abcehrt"
    char_to_int = {c: i for i, c in enumerate(chars)}
    blank = len(chars)

    # The acoustics slightly prefer "the bat" over "the cat".
    frames = ["t", "h", "e", " ", {"b": 0.55, "c": 0.45}, "a", "t"]
    probs = np.full((len(frames), len(chars) + 1), 1e-3)
    for t, frame in enumerate(frames):
        for c, p in (frame if isinstance(frame, dict) else {frame: 1.0}).items():
            probs[t, char_to_int[c]] = p
    probs /= probs.sum(axis=1, keepdims=True)

    def text(labels):
        return "".join(chars[i] for i in labels)

    labels, _ = decode(probs, beam_size=8, blank=blank)
    assert text(labels) == "the bat"

    lm = WordLM(["the cat sat", "the cat ran", "a bat"])
    scorer = lm.scorer(char_to_int, weight=1.0)
    labels, _ = decode(probs, beam_size=8, blank=blank, lm=scorer)
    assert text(labels) == "the cat"


def test_batch_norm_trains():
    # A few steps on one batch lower the loss of a model with batch
    # normalization, and inference afterwards uses its running averages.
    torch.manual_seed(0)
    np.random.seed(0)
    freq_dim, vocab_size = 40, 10
    batch = shared.gen_fake_data(freq_dim, vocab_size, max_time=60, max_seq_len=5)
    model = CTC(freq_dim, vocab_size, shared.batch_norm_config(shared.model_config))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.set_train()
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]
    model.set_eval()
    probs = model.forward_impl(
        torch.from_numpy(batch[0][0]).float().unsqueeze(0), softmax=True
    )
    assert torch.allclose(probs.sum(dim=2), torch.ones(1, probs.size(1)), atol=1e-5)
