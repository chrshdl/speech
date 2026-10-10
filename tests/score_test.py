import pytest

import speech


def test_cer_and_wer():
    results = [
        (list("the cat sat"), list("the bat sat")),
        (list("a dog"), list("a dog ran")),
    ]
    # One substitution in 11 characters and four insertions in 5.
    assert speech.compute_cer(results) == pytest.approx(5 / 16)
    # One substitution in 3 words and one insertion in 2.
    assert speech.compute_wer(results) == pytest.approx(2 / 5)
