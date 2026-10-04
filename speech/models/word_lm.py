"""
Word-level n-gram language model for CTC beam search decoding.

This module provides a simple bigram language model with Laplace
(add-1) smoothing that can be trained from text data and used to
rescore hypotheses during CTC prefix beam search.

Usage::

    from speech.models.word_lm import WordLM

    # Train from sentences
    sentences = ["the cat sat on the mat", "the dog ran"]
    lm = WordLM(sentences)

    # Use with CTC decoder (assuming char_to_int mapping)
    scorer = lm.scorer(char_to_int)
    labels, score = decode(probs, lm=scorer, lm_weight=0.5)

    # Persistence
    lm.save("lm.json")
    lm = WordLM.load("lm.json")
"""

import json
import math
import collections


class WordLM:
    """A word-level bigram language model with Laplace smoothing."""

    SOS = "<s>"   # start-of-sentence
    EOS = "</s>"   # end-of-sentence

    def __init__(self, sentences=None):
        """
        Build a bigram LM from a list of text sentences.

        Arguments:
            sentences: An iterable of strings, each a whitespace-
                separated sentence.  If *None*, creates an empty model
                (use :meth:`load` to restore a saved one).
        """
        # bigram_counts[w_prev][w_cur] = count
        self.bigram_counts = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        # unigram_counts[w] = count
        self.unigram_counts = collections.defaultdict(int)
        self.vocab = set()

        if sentences is not None:
            self._train(sentences)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, sentences):
        """Accumulate bigram and unigram counts from *sentences*."""
        for sent in sentences:
            words = sent.strip().split()
            if not words:
                continue
            words = [self.SOS] + words + [self.EOS]
            for i in range(1, len(words)):
                w_prev = words[i - 1]
                w_cur = words[i]
                self.bigram_counts[w_prev][w_cur] += 1
                self.unigram_counts[w_cur] += 1
                self.vocab.add(w_cur)
            self.unigram_counts[self.SOS] += 1
            self.vocab.add(self.SOS)

    @classmethod
    def from_data_json(cls, data_json):
        """
        Train a :class:`WordLM` from a data JSON file that the rest of
        the speech project uses.  Each line in the file is a JSON object
        with a ``"text"`` field.

        Arguments:
            data_json (str): Path to the JSON-lines data file.

        Returns:
            A trained :class:`WordLM` instance.
        """
        with open(data_json) as fid:
            sentences = [json.loads(line)["text"] for line in fid]
        return cls(sentences)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def log_prob(self, word, context=None):
        """
        Return the Laplace-smoothed log₁₀ probability of *word* given
        an optional *context* word (bigram).

        Falls back to unigram probability when *context* is ``None`` or
        was never observed.

        Arguments:
            word (str): The word to score.
            context (str or None): The preceding word (bigram context).

        Returns:
            float: log-probability (base e) of the word.
        """
        V = len(self.vocab) if self.vocab else 1

        # Bigram probability with Laplace smoothing
        if context is not None and context in self.bigram_counts:
            context_total = sum(self.bigram_counts[context].values())
            count = self.bigram_counts[context].get(word, 0)
            return math.log((count + 1) / (context_total + V))

        # Unigram backoff with Laplace smoothing
        total = sum(self.unigram_counts.values())
        count = self.unigram_counts.get(word, 0)
        if total == 0:
            return 0.0
        return math.log((count + 1) / (total + V))

    def scorer(self, char_to_int):
        """
        Return a callable compatible with the ``lm`` parameter of
        :func:`~speech.models.ctc_decoder.decode`.

        The returned function converts a character-index prefix tuple
        into a string, extracts completed words, and returns the
        bigram log-probability of the **last completed word**.

        Arguments:
            char_to_int (dict): Mapping from character → integer index,
                as used by :class:`~speech.loader.Preprocessor`.

        Returns:
            A callable ``lm(prefix) -> float``.
        """
        int_to_char = {v: k for k, v in char_to_int.items()}

        def _score(prefix):
            # Decode the prefix tuple to a string
            text = "".join(int_to_char.get(idx, "") for idx in prefix)
            # Split into words; only score if at least one complete word
            # exists (i.e. there is a trailing space or it's the last
            # character and forms a complete word).
            words = text.split()
            if not words:
                return 0.0

            last_word = words[-1]
            context = words[-2] if len(words) >= 2 else self.SOS
            return self.log_prob(last_word, context)

        return _score

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save the language model to a JSON file."""
        data = {
            "bigram_counts": {
                k: dict(v) for k, v in self.bigram_counts.items()
            },
            "unigram_counts": dict(self.unigram_counts),
            "vocab": list(self.vocab),
        }
        with open(path, "w") as fid:
            json.dump(data, fid)

    @classmethod
    def load(cls, path):
        """Load a language model from a JSON file."""
        with open(path) as fid:
            data = json.load(fid)
        lm = cls()
        lm.unigram_counts = collections.defaultdict(
            int, data["unigram_counts"]
        )
        lm.bigram_counts = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        for k, v in data["bigram_counts"].items():
            lm.bigram_counts[k] = collections.defaultdict(int, v)
        lm.vocab = set(data["vocab"])
        return lm
