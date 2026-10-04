import math
import os
import tempfile

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
