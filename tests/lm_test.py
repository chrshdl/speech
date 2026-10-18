import collections
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from speech import tokens
from speech.models import word_lm
from speech.models.word_lm import MixedLM, WordLM


def test_train_and_score():
    sentences = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a cat and a dog",
    ]
    lm = WordLM(sentences)

    # Vocabulary should include all words + <s>
    assert "cat" in lm.vocab
    assert "the" in lm.vocab
    assert WordLM.SOS in lm.vocab

    # log_prob should return a finite negative number for known words
    p = lm.log_prob("cat", context="the")
    assert p < 0.0
    assert math.isfinite(p)

    # Unigram backoff when context is None
    p_uni = lm.log_prob("cat", context=None)
    assert p_uni < 0.0
    assert math.isfinite(p_uni)

    # Unknown word should still get a smoothed probability
    p_unk = lm.log_prob("zzzzz", context="the")
    assert p_unk < 0.0
    assert math.isfinite(p_unk)


def test_scorer():
    sentences = [
        "the cat sat on the mat",
        "the dog sat on the log",
    ]
    lm = WordLM(sentences)
    chars = sorted({c for s in sentences for c in s})
    char_to_int = {c: i for i, c in enumerate(chars)}
    scorer = lm.scorer(char_to_int, weight=0.5, word_bonus=2.0)

    def run(text):
        state, total = scorer.initial(), 0.0
        for c in text:
            state, score = scorer.extend(state, char_to_int[c])
            total += score
        return state, total

    # Nothing is scored until a word is complete.
    state, total = run("the ca")
    expected = 0.5 * lm.log_prob("the", WordLM.SOS) + 2.0
    assert math.isclose(total, expected)
    assert state == (("the",), "ca")

    # Extra spaces complete no word.
    assert run(" the  ") == run("the ")

    # The end scores the last word and the end of the sentence.
    state, total = run("the cat")
    end = scorer.finish(state)
    expected = (
        0.5 * lm.log_prob("cat", "the") + 2.0 + 0.5 * lm.log_prob(WordLM.EOS, "cat")
    )
    assert math.isclose(end, expected)

    # The LM prefers what it has seen.
    assert lm.log_prob("cat", "the") > lm.log_prob("sat", "the")


def test_scorer_with_bigrams():
    lm = WordLM(["the cat sat on the mat"])
    chars = {c: i for i, c in enumerate(sorted(set("the cat sat on mat")))}
    labels = {t: i for i, t in enumerate([*chars, "th", "ca", "sa", "ma", "on"])}

    def run(text, char_to_int):
        scorer = lm.scorer(char_to_int, weight=0.5, word_bonus=2.0)
        state, total = scorer.initial(), 0.0
        for t in tokens.tokenize(text, char_to_int):
            state, score = scorer.extend(state, char_to_int[t])
            total += score
        return state, total + scorer.finish(state)

    # Words spelled with bigrams score as with characters.
    for text in ("the cat", "the mat ", "on the ca"):
        assert run(text, labels) == run(text, chars)


def test_save_load():
    sentences = ["hello world", "hello there"]
    lm = WordLM(sentences)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        lm.save(path)
        lm2 = WordLM.load(path)

        # Loaded model should produce the same scores
        assert lm.log_prob("world", "hello") == lm2.log_prob("world", "hello")
        assert lm.log_prob("there", "hello") == lm2.log_prob("there", "hello")
        assert set(lm.vocab) == set(lm2.vocab)
        assert lm.log_prob("zzz", "hello") == lm2.log_prob("zzz", "hello")
        assert lm.log_prob("zzz") == lm2.log_prob("zzz")
    finally:
        os.unlink(path)


def random_corpus(seed, sentences=200, words=30):
    rng = np.random.RandomState(seed)
    vocab = [f"w{i}" for i in range(words)]
    # Zipf-like word frequencies, as in real text.
    p = 1 / np.arange(1, words + 1)
    p /= p.sum()
    return [
        " ".join(rng.choice(vocab, size=rng.randint(1, 12), p=p))
        for _ in range(sentences)
    ]


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("max_vocab, min_count", [(None, 1), (20, 2), (10, 3)])
def test_probabilities_sum_to_one(order, max_vocab, min_count):
    lm = WordLM(random_corpus(0), max_vocab, min_count, order)
    contexts = [(), *((w,) for w in lm.words), ("w0", "w1"), ("w1", "w0", "w0")]
    for context in contexts:
        # <unk> stands for all the words outside the vocabulary, and <s> is
        # never predicted.
        total = sum(math.exp(lm.log_prob(w, context)) for w in lm.words[1:])
        assert math.isclose(total, 1.0), context

    # Any word outside the vocabulary gets the probability of <unk>.
    assert lm.log_prob("unseen", "w0") == lm.log_prob(WordLM.UNK, "w0")
    assert lm.log_prob("w0", "unseen") == lm.log_prob("w0", WordLM.UNK)


def reference_log_prob(sentences, order, max_vocab, min_count):
    """Interpolated modified Kneser-Ney, counted with plain dictionaries."""
    counts = collections.Counter(w for s in sentences for w in s.split())
    vocab = {w for w, _ in counts.most_common(max_vocab)}
    known = vocab | {WordLM.SOS, WordLM.EOS, WordLM.UNK}
    sents = [
        [WordLM.SOS, *(w if w in vocab else WordLM.UNK for w in s.split()), WordLM.EOS]
        for s in sentences
        if s.split()
    ]
    raw = {
        k: collections.Counter(
            tuple(s[i : i + k]) for s in sents for i in range(len(s) - k + 1)
        )
        for k in range(1, order + 1)
    }
    # The highest order counts the n-grams, the lower ones the different
    # words before them, but n-grams that start a sentence keep their counts.
    adjusted = {order: raw[order]}
    for k in range(order - 1, 0, -1):
        before = collections.Counter(g[1:] for g in raw[k + 1])
        adjusted[k] = {
            g: raw[k][g] if g[0] == WordLM.SOS else before[g]
            for g in raw[k]
            if g != (WordLM.SOS,)
        }

    def discounts(values):
        n1, n2, n3, n4 = (sum(v == c for v in values) for c in (1, 2, 3, 4))
        if not (n1 and n2 and n3):
            return word_lm.FALLBACK_DISCOUNTS
        y = n1 / (n1 + 2 * n2)
        d = [1 - 2 * y * n2 / n1, 2 - 3 * y * n3 / n2, 3 - 4 * y * n4 / n3]
        if not all(0 < x < c for c, x in zip((1, 2, 3), d)):
            return word_lm.FALLBACK_DISCOUNTS
        return d

    D = {k: discounts(list(adjusted[k].values())) for k in adjusted}

    def discount(k, c):
        return D[k][min(c, 3) - 1]

    words = len(known)
    total = sum(adjusted[1].values())
    left = sum(discount(1, c) for c in adjusted[1].values())

    def unigram(w):
        if w == WordLM.SOS:
            return 0.0
        c = adjusted[1].get((w,), 0)
        return (c - discount(1, c) if c else 0) / total + left / total / (words - 1)

    floor = dict(
        enumerate(
            [
                1,
                1,
                *(
                    [min_count] * (order - 1)
                    if isinstance(min_count, int)
                    else min_count
                ),
            ]
        )
    )
    tables = {}
    for k in range(2, order + 1):
        totals, kept = collections.Counter(), {}
        for g, c in adjusted[k].items():
            totals[g[:-1]] += c
            if c >= floor[k] and g[-1] != WordLM.UNK:
                kept[g] = c
        spent = collections.Counter()
        for g, c in kept.items():
            spent[g[:-1]] += c - discount(k, c)
        tables[k] = (totals, kept, spent)

    def log_prob(w, context):
        w = w if w in known else WordLM.UNK
        context = tuple(v if v in known else WordLM.UNK for v in context)
        p = unigram(w)
        for j in range(1, min(len(context), order - 1) + 1):
            h = context[-j:]
            totals, kept, spent = tables[j + 1]
            if not spent[h]:
                continue
            p *= 1 - spent[h] / totals[h]
            if h + (w,) in kept:
                c = kept[h + (w,)]
                p += (c - discount(j + 1, c)) / totals[h]
        return math.log(p) if p else -math.inf

    return log_prob


@pytest.mark.parametrize(
    "order, max_vocab, min_count, memory",
    [
        (2, None, 1, 1e9),
        (3, 15, 2, 1e9),
        (4, None, 1, 1e9),
        (4, None, (1, 2, 2), 1e9),
        # Many parts, also more than fit in a byte.
        (3, 15, 1, 50 * word_lm.BYTES_PER_NGRAM),
        (4, None, 2, 2 * word_lm.BYTES_PER_NGRAM),
    ],
)
def test_matches_reference(monkeypatch, order, max_vocab, min_count, memory):
    # Tiny chunks exercise the chunked reading.
    monkeypatch.setattr(word_lm, "CHUNK", 37)
    monkeypatch.setattr(word_lm, "MEMORY", memory)
    sentences = random_corpus(1, sentences=100, words=12)
    lm = WordLM(sentences, max_vocab, min_count, order)
    reference = reference_log_prob(sentences, order, max_vocab, min_count)
    words = [*lm.words[1:], "unseen"]
    contexts = [(), *((v,) for v in [*lm.words, "unseen"])]
    contexts += [(u, v) for u in lm.words[:6] for v in lm.words[:6]]
    contexts += [(WordLM.SOS, "w0", "w1"), ("w2", "w0", "w0"), ("w1", "w1", "unseen")]
    for context in contexts:
        for w in words:
            expected = reference(w, context)
            assert math.isclose(lm.log_prob(w, context), expected), (context, w)


def test_discounts():
    # Chen and Goodman's estimates from the counts of counts.
    y = 100 / (100 + 2 * 40)
    np.testing.assert_allclose(
        word_lm._discounts([100, 40, 20, 12]),
        [1 - 2 * y * 0.4, 2 - 3 * y * 0.5, 3 - 4 * y * 0.6],
    )
    # No n-gram seen three times would discount those seen twice by 2.
    assert list(word_lm._discounts([7, 2, 0, 0])) == [0.5, 1.0, 1.5]


def test_load_bigram(tmp_path):
    # A bigram LM as saved before n-gram LMs, with interpolated absolute
    # discounting and an add-one smoothed unigram distribution.
    words = [WordLM.SOS, WordLM.EOS, WordLM.UNK, "a", "b"]
    V = len(words)
    unigram = np.array([0, 4, 1, 5, 3])
    history = np.array([4, 0, 1, 4, 3])
    keys = np.array([0 * V + 3, 3 * V + 4, 4 * V + 1, 3 * V + 1])
    counts = np.array([3, 2, 2, 1])
    discount = 0.6
    kept = np.bincount(keys // V, weights=counts - discount, minlength=V)
    backoff = np.ones(V)
    backoff[history > 0] = 1 - kept[history > 0] / history[history > 0]
    path = tmp_path / "bigram.npz"
    np.savez(
        path,
        words=np.array("\n".join(words)),
        unigram=unigram,
        history=history,
        backoff=backoff,
        keys=keys,
        counts=counts,
        discount=np.array(discount),
    )

    def old_log_prob(w, v):
        p = (unigram[w] + 1) / (unigram.sum() + V)
        if history[v] == 0:
            return math.log(p)
        p *= backoff[v]
        for k, c in zip(keys, counts):
            if k == v * V + w:
                p += (c - discount) / history[v]
        return math.log(p)

    lm = WordLM.load(path)
    assert lm.order == 2
    for v in range(V):
        for w in range(V):
            assert math.isclose(lm.log_prob(words[w], words[v]), old_log_prob(w, v)), (
                v,
                w,
            )


# Interpolated LMs.


def two_lms(order=3):
    """Two LMs of the same words, said in different proportions."""
    words = [f"w{i}" for i in range(12)]
    rng = np.random.RandomState(0)
    first = [" ".join(rng.choice(words[:8], size=6)) for _ in range(200)]
    second = [" ".join(rng.choice(words[4:], size=6)) for _ in range(200)]
    # Each says every word at least once, so the vocabularies match.
    first.append(" ".join(words))
    second.append(" ".join(words))
    return WordLM(first, order=order), WordLM(second, order=order)


def test_mixed_lm_interpolates():
    a, b = two_lms()
    mixed = MixedLM([a, b], [0.3, 0.7])
    for word, context in [("w1", ("w2", "w3")), ("w9", ("w5",)), ("w0", None)]:
        expected = math.log(
            0.3 * math.exp(a.log_prob(word, context))
            + 0.7 * math.exp(b.log_prob(word, context))
        )
        assert mixed.log_prob(word, context) == pytest.approx(expected)
    assert mixed.order == 3


def test_mixed_lm_probabilities_sum_to_one():
    a, b = two_lms()
    assert set(a.words) == set(b.words)
    mixed = MixedLM([a, b], [0.4, 0.6])
    for context in [(), ("w1",), ("w2", "w9"), ("w11", "w0", "w5")]:
        total = sum(math.exp(mixed.log_prob(w, context)) for w in mixed.words[1:])
        assert total == pytest.approx(1.0, abs=1e-9)


def test_mixed_lm_knows_the_words_of_each(tmp_path):
    a = WordLM(["turn on the lights", "turn off the lights"])
    b = WordLM(["what is the weather", "how is the weather"])
    mixed = MixedLM([a, b], [0.5, 0.5])
    scorer = mixed.scorer({c: i for i, c in enumerate(" abcdefghijklmnopqrstuvwxyz'")})
    # A word only one LM knows is not unknown, nor penalized as one.
    assert "lights" in mixed.vocab and "weather" in mixed.vocab
    known = scorer.word_score("weather", ("<s>", "the"))
    unknown = scorer.word_score("zebra", ("<s>", "the"))
    assert known > unknown + scorer.unk_penalty * -0.5


def test_mixed_lm_save_load(tmp_path):
    a, b = two_lms()
    mixed = MixedLM([a, b], [0.25, 0.75])
    path = str(tmp_path / "mixed.npz")
    mixed.save(path)
    loaded = WordLM.load(path)
    assert isinstance(loaded, MixedLM)
    assert loaded.weights == [0.25, 0.75]
    for word, context in [("w1", ("w2", "w3")), ("zzz", ("w5",)), ("</s>", None)]:
        assert loaded.log_prob(word, context) == mixed.log_prob(word, context)
    # A plain LM still loads as one.
    a.save(str(tmp_path / "a.npz"))
    assert isinstance(WordLM.load(str(tmp_path / "a.npz")), WordLM)


def test_best_weights_fit_the_dev_text():
    a, b = two_lms()
    rng = np.random.RandomState(1)
    # Text mostly like the first LM's.
    dev = [" ".join(rng.choice([f"w{i}" for i in range(8)], size=6)) for _ in range(50)]
    weights = MixedLM.best_weights([a, b], dev)
    assert weights[0] > 0.5 and sum(weights) == pytest.approx(1.0)
    mixed = MixedLM([a, b], weights)
    # The best weights are at least as good as either LM alone.
    assert mixed.perplexity(dev) <= min(a.perplexity(dev), b.perplexity(dev)) * 1.001


def test_mixed_lm_refuses_bad_weights():
    a, b = two_lms()
    with pytest.raises(ValueError, match="sum to 1"):
        MixedLM([a, b], [0.5, 0.6])
    with pytest.raises(ValueError, match="at least two"):
        MixedLM([a], [1.0])
    with pytest.raises(ValueError, match="not mixtures"):
        MixedLM([MixedLM([a, b], [0.5, 0.5]), a], [0.5, 0.5])


def test_mix_command(tmp_path):
    a, b = two_lms()
    a.save(str(tmp_path / "a.npz"))
    b.save(str(tmp_path / "b.npz"))
    with open(tmp_path / "dev.json", "w") as f:
        f.writelines(
            json.dumps({"text": f"w{i % 8} w{(i + 1) % 8} w{(i + 2) % 8}"}) + "\n"
            for i in range(20)
        )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "speech.models.word_lm",
            str(tmp_path / "a.npz"),
            str(tmp_path / "b.npz"),
            str(tmp_path / "mixed.npz"),
            "--mix",
            "--dev",
            str(tmp_path / "dev.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    assert "Mixed dev perplexity" in result.stdout
    mixed = WordLM.load(str(tmp_path / "mixed.npz"))
    assert isinstance(mixed, MixedLM) and mixed.weights[0] > mixed.weights[1]
