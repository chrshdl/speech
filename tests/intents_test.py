import itertools

import numpy as np
import pytest

from speech.intents import (
    Intent,
    IntentRecognizer,
    alignment_score,
)

BLANK = 3
CHAR_TO_INT = {"a": 0, "b": 1, " ": 2}


def collapse(path):
    """The labels of a CTC path: repeats merged, then blanks dropped."""
    merged = [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]
    return [p for p in merged if p != BLANK]


def brute_force(relative, labels):
    """The best score of the CTC paths that give the labels, by enumeration."""
    best = -np.inf
    for path in itertools.product(range(BLANK + 1), repeat=len(relative)):
        if collapse(path) == list(labels):
            best = max(best, sum(relative[t, p] for t, p in enumerate(path)))
    return best


def relative(probs):
    log_probs = np.log(probs)
    return log_probs - log_probs.max(axis=1, keepdims=True)


@pytest.mark.parametrize("labels", [[0], [0, 1], [1, 1], [0, 2, 1], [0, 0, 1]])
def test_alignment_score_is_the_best_path(labels):
    rng = np.random.default_rng(len(labels))
    for _ in range(5):
        probs = rng.dirichlet(np.ones(BLANK + 1), size=6)
        expected = brute_force(relative(probs), labels)
        assert alignment_score(relative(probs), np.array(labels), BLANK) == (
            pytest.approx(expected)
        )


def one_hot_probs(path, sharpness=0.97):
    probs = np.full((len(path), BLANK + 1), (1 - sharpness) / BLANK)
    probs[np.arange(len(path)), path] = sharpness
    return probs


def test_recognize():
    intents = [Intent("AB", ("ab", "a b")), Intent("BA", ("ba",))]
    recognizer = IntentRecognizer(intents, CHAR_TO_INT, BLANK, frame_seconds=0.04)

    # An utterance spelled as a sample matches it exactly.
    match = recognizer.recognize(one_hot_probs([BLANK, 0, 0, BLANK, 1, BLANK]))
    assert (match.intent.name, match.sample, match.score) == ("AB", "ab", 0)
    match = recognizer.recognize(one_hot_probs([1, BLANK, 0]))
    assert match.intent.name == "BA"

    # An utterance that sounds like none of them matches nothing, though one
    # is the closest.
    other = one_hot_probs([2, 2, BLANK, 2, 2, 2])
    assert recognizer.recognize(other) is None
    assert recognizer.closest(other).score < recognizer.threshold

    # Nor do empty utterances and ones too long to be commands.
    assert recognizer.recognize(np.zeros((0, BLANK + 1))) is None
    assert recognizer.closest(one_hot_probs([0, 1] * 101)) is None


def test_bigrams():
    # A model of bigrams labels "ab ba" as ab, space, ba.
    char_to_int = {"a": 0, "b": 1, " ": 2, "ab": 3, "ba": 4}
    blank = 5
    intents = [Intent("ABBA", ("ab ba",)), Intent("A", ("a",))]
    recognizer = IntentRecognizer(intents, char_to_int, blank, frame_seconds=0.04)

    path = [3, blank, 2, 4]
    probs = np.full((len(path), blank + 1), 0.01)
    probs[np.arange(len(path)), path] = 0.95
    match = recognizer.recognize(probs / probs.sum(axis=1, keepdims=True))
    assert (match.intent.name, match.score) == ("ABBA", 0)


def test_samples_must_use_the_model_labels():
    with pytest.raises(ValueError, match="characters"):
        IntentRecognizer([Intent("C", ("abc",))], CHAR_TO_INT, BLANK, 0.04)


def test_slots():
    places = ("ab", "ba", "aab")
    intents = [Intent("Go", ("a b", "a b {place}"), {"place": places})]
    recognizer = IntentRecognizer(intents, CHAR_TO_INT, BLANK, frame_seconds=0.04)
    # "a b ba", spelled out frame by frame.
    path = [BLANK, 0, BLANK, 2, 1, BLANK, 2, BLANK, 1, 1, 0, BLANK]
    match = recognizer.recognize(one_hot_probs(path))
    assert (match.sample, match.slots, match.score) == (
        "a b {place}",
        {"place": "ba"},
        0,
    )
    match = recognizer.recognize(one_hot_probs([0, BLANK, 2, 1]))
    assert (match.sample, match.slots) == ("a b", {})


def test_spaced():
    """
    A transcript that starts within a stream starts with a space, which
    only a spaced recognizer allows.
    """
    intents = [Intent("AB", ("ab",)), Intent("BA", ("ba",))]
    path = [BLANK, 2, BLANK, 0, BLANK, 1, BLANK]
    plain = IntentRecognizer(intents, CHAR_TO_INT, BLANK, frame_seconds=0.04)
    spaced = IntentRecognizer(
        intents, CHAR_TO_INT, BLANK, frame_seconds=0.04, spaced=True
    )
    assert plain.closest(one_hot_probs(path)).score < plain.threshold
    match = spaced.recognize(one_hot_probs(path))
    assert (match.intent.name, match.sample, match.score) == ("AB", "ab", 0)


def test_others():
    """
    An utterance closer to one of the other phrases than to any intent's
    sample asks for nothing.
    """
    intents = [Intent("AB", ("ab",))]
    probs = one_hot_probs([BLANK, 1, BLANK, 0, BLANK])
    plain = IntentRecognizer(intents, CHAR_TO_INT, BLANK, frame_seconds=0.04)
    assert plain.closest(probs).intent.name == "AB"
    others = IntentRecognizer(
        intents, CHAR_TO_INT, BLANK, frame_seconds=0.04, others=("ba",)
    )
    assert others.closest(probs) is None
    assert others.recognize(one_hot_probs([0, BLANK, 1])).intent.name == "AB"


def test_slot_needs_values():
    with pytest.raises(ValueError, match="slot"):
        IntentRecognizer([Intent("Go", ("a {x}",))], CHAR_TO_INT, BLANK, 0.04)
