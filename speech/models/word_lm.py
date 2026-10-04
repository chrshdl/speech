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
    scorer = lm.scorer(char_to_int, weight=0.5, word_bonus=1.0)
    labels, score = decode(probs, lm=scorer)

    # Persistence
    lm.save("lm.json")
    lm = WordLM.load("lm.json")
"""

import argparse
import collections
import json
import math

# Defaults for decoding with the LM, tuned with the LibriSpeech streaming
# recipe on its dev set.
LM_WEIGHT = 0.3
WORD_BONUS = 2.0


class WordLM:
    """A word-level bigram language model with Laplace smoothing."""

    SOS = "<s>"  # start-of-sentence
    EOS = "</s>"  # end-of-sentence

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
        self._count_totals()

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

    def _count_totals(self):
        """Caches the totals that log_prob divides by."""
        self._context_totals = {
            w: sum(counts.values()) for w, counts in self.bigram_counts.items()
        }
        self._unigram_total = sum(self.unigram_counts.values())

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
        Return the Laplace-smoothed log probability of *word* given
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
            context_total = self._context_totals[context]
            count = self.bigram_counts[context].get(word, 0)
            return math.log((count + 1) / (context_total + V))

        # Unigram backoff with Laplace smoothing
        total = self._unigram_total
        count = self.unigram_counts.get(word, 0)
        if total == 0:
            return 0.0
        return math.log((count + 1) / (total + V))

    def scorer(self, char_to_int, weight=LM_WEIGHT, word_bonus=WORD_BONUS):
        """
        Return a scorer for the ``lm`` parameter of
        :class:`~speech.models.ctc_decoder.BeamSearch` and
        :func:`~speech.models.ctc_decoder.decode`.

        Arguments:
            char_to_int (dict): Mapping from character → integer index,
                as used by :class:`~speech.loader.Preprocessor`.
            weight (float): Scales the LM log-probabilities.
            word_bonus (float): Added for each word, which offsets the
                LM's preference for fewer words.

        Returns:
            A :class:`WordScorer`.
        """
        return WordScorer(self, char_to_int, weight, word_bonus)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save the language model to a JSON file."""
        data = {
            "bigram_counts": {k: dict(v) for k, v in self.bigram_counts.items()},
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
        lm.unigram_counts = collections.defaultdict(int, data["unigram_counts"])
        lm.bigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for k, v in data["bigram_counts"].items():
            lm.bigram_counts[k] = collections.defaultdict(int, v)
        lm.vocab = set(data["vocab"])
        lm._count_totals()
        return lm


class WordScorer:
    """
    Scores the words of a growing character sequence with a
    :class:`WordLM` for the CTC beam search.

    A word is scored when it is complete, that is when a space follows
    it or the utterance ends, as ``weight * log P(word | previous word) +
    word_bonus``. The state of a prefix is its last complete word and the
    characters of the word in progress.
    """

    def __init__(self, lm, char_to_int, weight, word_bonus):
        self.lm = lm
        self.weight = weight
        self.word_bonus = word_bonus
        self.int_to_char = {v: k for k, v in char_to_int.items()}
        self.space = char_to_int[" "]

    def initial(self):
        return (WordLM.SOS, "")

    def extend(self, state, label):
        """Returns the state after the label and the score it adds."""
        context, word = state
        if label != self.space:
            return (context, word + self.int_to_char[label]), 0.0
        if not word:
            # Leading or repeated spaces complete no word.
            return state, 0.0
        return (word, ""), self.word_score(word, context)

    def finish(self, state):
        """Returns the score of ending the utterance in this state."""
        context, word = state
        score = 0.0
        if word:
            score += self.word_score(word, context)
            context = word
        return score + self.weight * self.lm.log_prob(WordLM.EOS, context)

    def word_score(self, word, context):
        return self.weight * self.lm.log_prob(word, context) + self.word_bonus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a word bigram LM from the text of datasets."
    )
    parser.add_argument("datasets", nargs="+", help="Data json files with text.")
    parser.add_argument("output", help="The json file to save the LM to.")
    args = parser.parse_args()

    sentences = []
    for data_json in args.datasets:
        with open(data_json) as fid:
            sentences.extend(json.loads(line)["text"] for line in fid)
    lm = WordLM(sentences)
    lm.save(args.output)
    print(f"Trained on {len(sentences)} sentences, {len(lm.vocab)} words.")
