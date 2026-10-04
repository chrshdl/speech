"""
Word-level n-gram language model for CTC beam search decoding.

This module provides a bigram language model with interpolated absolute
discounting that can be trained from text data, including corpora far
larger than memory, and used to rescore hypotheses during CTC prefix beam
search.

Usage::

    from speech.models.word_lm import TextCorpus, WordLM

    # Train from sentences
    sentences = ["the cat sat on the mat", "the dog ran"]
    lm = WordLM(sentences)

    # Or from text files with one sentence per line, keeping the 200k most
    # frequent words and the word pairs seen at least twice
    lm = WordLM(TextCorpus(["corpus.txt.gz"]), max_vocab=200_000, min_count=2)

    # Use with CTC decoder (assuming char_to_int mapping)
    scorer = lm.scorer(char_to_int, weight=0.5, word_bonus=1.0)
    labels, score = decode(probs, lm=scorer)

    # Persistence
    lm.save("lm.npz")
    lm = WordLM.load("lm.npz")
"""

import argparse
import array
import collections
import gzip
import json
import math
import os
import tempfile

import numpy as np

# Defaults for decoding with the LM, tuned with tune_lm.py for the LM
# trained on the LibriSpeech LM corpus and the LibriSpeech streaming CTC
# model, on its dev set. See docs/lm-tuning.json.
LM_WEIGHT = 0.5
WORD_BONUS = 4.0
UNK_PENALTY = 0.0

# The number of word ids to process at once while counting.
CHUNK = 20_000_000


class TextCorpus:
    """
    Iterates over the sentences of text files, lowercased. Files ending in
    .json are datasets, one json object with a "text" field per line. Other
    files have one sentence per line and may be gzipped. The files are read
    again on each iteration, so they never have to fit in memory.
    """

    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        for path in self.paths:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as fid:
                for line in fid:
                    if path.endswith(".json"):
                        line = json.loads(line)["text"]
                    yield line.lower()


class WordLM:
    """
    A word-level bigram language model with interpolated absolute
    discounting.

    The probability of a word w after a word v is

        P(w | v) = max(c(v, w) - D, 0) / c(v) + backoff(v) * P(w)

    where c counts the training text, D is the discount, and the backoff
    weight gives the probability taken from the seen pairs to the unigram
    distribution P(w), which is add-one smoothed. A word outside the
    vocabulary gets the probability of <unk>, which counts all such words
    in the training text.
    """

    SOS = "<s>"  # start-of-sentence
    EOS = "</s>"  # end-of-sentence
    UNK = "<unk>"  # words outside the vocabulary while training

    def __init__(self, sentences=None, max_vocab=None, min_count=1):
        """
        Build a bigram LM from text sentences.

        Arguments:
            sentences: An iterable of strings, each a whitespace-
                separated sentence. It is iterated twice, so a large
                corpus should be a :class:`TextCorpus`. If *None*,
                creates an empty model (use :meth:`load` to restore a
                saved one).
            max_vocab (int, optional): Keep only the most frequent
                words. The others count as one unknown word.
            min_count (int): Drop the word pairs seen fewer times. Their
                probability goes to the unigram backoff, which makes the
                model smaller.
        """
        self._set_vocab([self.SOS, self.EOS, self.UNK])
        self.unigram = np.zeros(len(self.words), dtype=np.int64)
        self.total = 0
        self.history = np.zeros(len(self.words), dtype=np.int64)
        self.backoff = np.ones(len(self.words))
        self.keys = np.zeros(0, dtype=np.int64)
        self.counts = np.zeros(0, dtype=np.int64)
        self.discount = 0.5

        if sentences is not None:
            self._train(sentences, max_vocab, min_count)

    def _set_vocab(self, words):
        self.words = words
        # Maps each word to its id.
        self.vocab = {w: i for i, w in enumerate(words)}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, sentences, max_vocab, min_count):
        """Count the unigrams and bigrams of *sentences*."""
        # Pass 1: the vocabulary, most frequent words first.
        word_counts = collections.Counter()
        for sent in sentences:
            word_counts.update(sent.split())
        for w in (self.SOS, self.EOS, self.UNK):
            word_counts.pop(w, None)
        words = [w for w, _ in word_counts.most_common(max_vocab)]
        self._set_vocab([self.SOS, self.EOS, self.UNK, *words])
        del word_counts

        # Pass 2: the sentences as word ids, each between <s> and </s>,
        # written to a temporary file so the corpus need not fit in memory.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ids")
            self._write_ids(sentences, path)
            ids = np.memmap(path, dtype=np.int32, mode="r")
            self._count(ids, min_count)
            del ids

    def _write_ids(self, sentences, path):
        sos, eos, unk = (self.vocab[w] for w in (self.SOS, self.EOS, self.UNK))
        get = self.vocab.get
        buffer = array.array("i")
        with open(path, "wb") as fid:
            for sent in sentences:
                words = sent.split()
                if not words:
                    continue
                buffer.append(sos)
                buffer.extend([get(w, unk) for w in words])
                buffer.append(eos)
                if len(buffer) >= CHUNK:
                    buffer.tofile(fid)
                    buffer = array.array("i")
            buffer.tofile(fid)

    def _pairs(self, ids):
        """Yields the bigrams of the id stream in chunks as (prev, cur)."""
        eos = self.vocab[self.EOS]
        for start in range(0, max(len(ids) - 1, 0), CHUNK):
            prev = np.asarray(ids[start : start + CHUNK], dtype=np.int64)
            cur = np.asarray(ids[start + 1 : start + CHUNK + 1], dtype=np.int64)
            prev = prev[: len(cur)]
            # Pairs across sentences, from </s> to the next <s>, are not
            # bigrams.
            keep = prev != eos
            yield prev[keep], cur[keep]

    def _count(self, ids, min_count):
        V = len(self.words)
        unk = self.vocab[self.UNK]
        self.unigram = np.zeros(V, dtype=np.int64)
        self.history = np.zeros(V, dtype=np.int64)
        for prev, cur in self._pairs(ids):
            self.unigram += np.bincount(cur, minlength=V)
            self.history += np.bincount(prev, minlength=V)
        self.total = int(self.unigram.sum())

        # Count the distinct bigrams by previous word in parts that fit in
        # memory, keeping the frequent ones.
        parts = max(1, len(ids) // (4 * CHUNK))
        n1 = n2 = 0
        keys, counts = [], []
        for part in range(parts):
            part_keys, part_counts = [], []
            for prev, cur in self._pairs(ids):
                mine = prev % parts == part
                k, c = np.unique(prev[mine] * V + cur[mine], return_counts=True)
                part_keys.append(k)
                part_counts.append(c)
            k, inverse = np.unique(np.concatenate(part_keys), return_inverse=True)
            c = np.bincount(inverse, weights=np.concatenate(part_counts))
            c = c.astype(np.int64)
            n1 += np.count_nonzero(c == 1)
            n2 += np.count_nonzero(c == 2)
            # Pairs ending in an unknown word are left to the backoff, so
            # the unknown word gets no bigram probability.
            mine = (c >= min_count) & (k % V != unk)
            keys.append(k[mine])
            counts.append(c[mine])
        order = np.argsort(np.concatenate(keys))
        self.keys = np.concatenate(keys)[order]
        self.counts = np.concatenate(counts)[order]

        # Ney's estimate of the discount.
        self.discount = min(n1 / (n1 + 2 * n2), 0.95) if n1 + 2 * n2 else 0.5

        # The backoff weight is the probability the kept bigrams of a word
        # leave over.
        kept = np.bincount(
            self.keys // V, weights=self.counts - self.discount, minlength=V
        )
        has_history = self.history > 0
        self.backoff = np.ones(V)
        self.backoff[has_history] = 1 - kept[has_history] / self.history[has_history]

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
        return cls(TextCorpus([data_json]))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def log_prob(self, word, context=None):
        """
        Return the log probability of *word* given an optional *context*
        word (bigram).

        Falls back to the unigram probability when *context* is ``None``
        or was never followed by a word in training.

        Arguments:
            word (str): The word to score.
            context (str or None): The preceding word (bigram context).

        Returns:
            float: log-probability (base e) of the word.
        """
        V = len(self.words)
        unk = self.vocab[self.UNK]
        w = self.vocab.get(word, unk)
        p = (self.unigram[w] + 1) / (self.total + V)

        v = self.vocab.get(context, unk) if context is not None else None
        if v is None or self.history[v] == 0:
            return math.log(p)
        p *= self.backoff[v]
        if w != unk:
            key = v * V + w
            i = np.searchsorted(self.keys, key)
            if i < len(self.keys) and self.keys[i] == key:
                p += (self.counts[i] - self.discount) / self.history[v]
        return math.log(p)

    def perplexity(self, sentences):
        """The perplexity of the model on *sentences*, including </s>."""
        total = 0.0
        n = 0
        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            prev = self.SOS
            for w in [*words, self.EOS]:
                total += self.log_prob(w, prev)
                prev = w
                n += 1
        return math.exp(-total / n)

    def scorer(
        self,
        char_to_int,
        weight=LM_WEIGHT,
        word_bonus=WORD_BONUS,
        unk_penalty=UNK_PENALTY,
    ):
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
            unk_penalty (float): Added for each word outside the
                vocabulary. <unk> stands for all unknown words, and any one
                of them, such as a misspelling, is less likely.

        Returns:
            A :class:`WordScorer`.
        """
        return WordScorer(self, char_to_int, weight, word_bonus, unk_penalty)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save the language model to a NumPy .npz file."""
        with open(path, "wb") as fid:
            np.savez(
                fid,
                words=np.array("\n".join(self.words)),
                unigram=self.unigram,
                history=self.history,
                backoff=self.backoff,
                keys=self.keys,
                counts=self.counts,
                discount=np.array(self.discount),
            )

    @classmethod
    def load(cls, path):
        """Load a language model from a NumPy .npz file."""
        lm = cls()
        with np.load(path) as data:
            lm._set_vocab(str(data["words"]).split("\n"))
            lm.unigram = data["unigram"]
            lm.total = int(lm.unigram.sum())
            lm.history = data["history"]
            lm.backoff = data["backoff"]
            lm.keys = data["keys"]
            lm.counts = data["counts"]
            lm.discount = float(data["discount"])
        return lm


class WordScorer:
    """
    Scores the words of a growing character sequence with a
    :class:`WordLM` for the CTC beam search.

    A word is scored when it is complete, that is when a space follows
    it or the utterance ends, as ``weight * log P(word | previous word) +
    word_bonus``, plus unk_penalty for a word outside the vocabulary. The
    state of a prefix is its last complete word and the characters of the
    word in progress.
    """

    def __init__(self, lm, char_to_int, weight, word_bonus, unk_penalty):
        self.lm = lm
        self.weight = weight
        self.word_bonus = word_bonus
        self.unk_penalty = unk_penalty
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
        score = self.weight * self.lm.log_prob(word, context) + self.word_bonus
        if word not in self.lm.vocab or word == WordLM.UNK:
            score += self.unk_penalty
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a word bigram LM from text. Inputs are dataset json "
        "files, or text files with one sentence per line, which may be gzipped."
    )
    parser.add_argument("inputs", nargs="+", help="The text to train on.")
    parser.add_argument("output", help="The .npz file to save the LM to.")
    parser.add_argument(
        "--max-vocab", type=int, help="Keep only the most frequent words."
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Drop the word pairs seen fewer times, to make the model smaller.",
    )
    parser.add_argument("--dev", help="A dataset json to report the perplexity on.")
    args = parser.parse_args()

    lm = WordLM(TextCorpus(args.inputs), args.max_vocab, args.min_count)
    lm.save(args.output)
    print(
        f"Trained on {lm.total:,} words: {len(lm.words):,} in the "
        f"vocabulary, {len(lm.keys):,} bigrams, discount {lm.discount:.2f}."
    )
    if args.dev is not None:
        print(f"Dev perplexity {lm.perplexity(TextCorpus([args.dev])):.1f}")
