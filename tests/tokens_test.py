import random

from speech import tokens


def test_bigrams():
    # Deep Speech 2's example.
    assert tokens.bigrams("the cat sat") == ["th", "e", " ", "ca", "t", " ", "sa", "t"]
    assert tokens.bigrams("don't") == ["do", "n'", "t"]
    assert tokens.bigrams("") == []
    # Each space is a label of its own, also at the ends and repeated.
    assert tokens.bigrams(" ab  c ") == [" ", "ab", " ", " ", "c", " "]


def test_tokenize():
    chars = {c: i for i, c in enumerate("abc ")}
    assert tokens.tokenize("ab cab", chars) == list("ab cab")

    labels = set(chars) | {"ab", "ca"}
    assert tokens.tokenize("ab cab", labels) == ["ab", " ", "ca", "b"]
    # A bigram the model lacks is labeled by its characters.
    assert tokens.tokenize("abc", labels) == ["ab", "c"]
    assert tokens.tokenize("bca", labels) == ["b", "c", "a"]


def test_tokens_spell_the_text():
    rng = random.Random(0)
    labels = set("ab' ") | {"ab", "ba", "a'"}
    for _ in range(200):
        text = "".join(rng.choice("ab' ") for _ in range(rng.randint(0, 12)))
        assert "".join(tokens.tokenize(text, labels)) == text
