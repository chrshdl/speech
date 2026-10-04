import collections
import math
import os
import tempfile

import numpy as np
import pytest

from speech.models import word_lm
from speech.models.word_lm import WordLM


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
    assert state == ("the", "ca")

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


@pytest.mark.parametrize("max_vocab, min_count", [(None, 1), (20, 2), (10, 3)])
def test_probabilities_sum_to_one(max_vocab, min_count):
    lm = WordLM(random_corpus(0), max_vocab=max_vocab, min_count=min_count)
    for context in lm.words:
        # <unk> stands for all the words outside the vocabulary.
        total = sum(math.exp(lm.log_prob(w, context)) for w in lm.words)
        assert math.isclose(total, 1.0), context

    # Any word outside the vocabulary gets the probability of <unk>.
    assert lm.log_prob("unseen", "w0") == lm.log_prob(WordLM.UNK, "w0")
    assert lm.log_prob("w0", "unseen") == lm.log_prob("w0", WordLM.UNK)


def reference_log_prob(sentences, max_vocab, min_count):
    # The same model, counted with plain dictionaries.
    counts = collections.Counter(w for s in sentences for w in s.split())
    vocab = {w for w, _ in counts.most_common(max_vocab)}
    bigrams = collections.Counter()
    for s in sentences:
        words = [w if w in vocab else WordLM.UNK for w in s.split()]
        bigrams.update(zip([WordLM.SOS, *words], [*words, WordLM.EOS]))
    unigram = collections.Counter()
    history = collections.Counter()
    for (v, w), c in bigrams.items():
        unigram[w] += c
        history[v] += c
    n1 = sum(c == 1 for c in bigrams.values())
    n2 = sum(c == 2 for c in bigrams.values())
    discount = n1 / (n1 + 2 * n2)
    kept = {k: c for k, c in bigrams.items() if c >= min_count and k[1] != WordLM.UNK}
    size = len(vocab) + 3
    total = sum(unigram.values())

    known = vocab | {WordLM.SOS, WordLM.EOS, WordLM.UNK}

    def log_prob(w, v):
        if w not in known:
            w = WordLM.UNK
        if v not in known:
            v = WordLM.UNK
        p = (unigram[w] + 1) / (total + size)
        if not history[v]:
            return math.log(p)
        left = sum(c - discount for (a, _), c in kept.items() if a == v)
        p *= 1 - left / history[v]
        if (v, w) in kept:
            p += (kept[v, w] - discount) / history[v]
        return math.log(p)

    return log_prob


@pytest.mark.parametrize("max_vocab, min_count", [(None, 1), (15, 2)])
def test_matches_reference(monkeypatch, max_vocab, min_count):
    # Tiny chunks exercise the chunked and partitioned counting.
    monkeypatch.setattr(word_lm, "CHUNK", 37)
    sentences = random_corpus(1, sentences=100)
    lm = WordLM(sentences, max_vocab=max_vocab, min_count=min_count)
    reference = reference_log_prob(sentences, max_vocab, min_count)
    for v in [*lm.words, "unseen"]:
        for w in [*lm.words, "unseen"]:
            assert math.isclose(lm.log_prob(w, v), reference(w, v)), (v, w)
