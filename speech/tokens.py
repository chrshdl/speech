"""
Splits transcripts into the labels a model outputs: single characters,
or the non-overlapping bigrams of Deep Speech 2, which halve the labels
of a transcript, so a model with a long time stride has room for them.
Needs only the standard library, so it also runs where PyTorch is not
installed.
"""

SPACE = " "


def bigrams(text):
    """
    Splits text into non-overlapping bigrams, as Deep Speech 2 did: each
    word into pairs of characters from its start, with the last character
    of a word of odd length and each space on their own, so a word always
    has the same labels. "the cat" splits into th, e, space, ca, t.
    """
    tokens = []
    for i, word in enumerate(text.split(SPACE)):
        if i > 0:
            tokens.append(SPACE)
        tokens.extend(word[j : j + 2] for j in range(0, len(word), 2))
    return tokens


def tokenize(text, labels):
    """
    Splits text into the labels of a model: its bigrams where the model
    has them, and single characters otherwise. For a model of single
    characters this is the characters of the text.

    Arguments:
        labels: The labels of the model, such as its char_to_int.
    """
    tokens = []
    for token in bigrams(text):
        if token in labels:
            tokens.append(token)
        else:
            tokens.extend(token)
    return tokens
