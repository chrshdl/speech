"""
Word-level n-gram language model for CTC beam search decoding.

This module provides an n-gram language model, such as a 4-gram, with
interpolated modified Kneser-Ney smoothing that can be trained from text
data, including corpora far larger than memory, and used to rescore
hypotheses during CTC prefix beam search.

Usage::

    from speech.models.word_lm import TextCorpus, WordLM

    # Train from sentences
    sentences = ["the cat sat on the mat", "the dog ran"]
    lm = WordLM(sentences, order=3)

    # Or from text files with one sentence per line, keeping the 200k most
    # frequent words, the bigrams seen after at least one word, and the
    # trigrams and 4-grams after at least two and three
    lm = WordLM(
        TextCorpus(["corpus.txt.gz"]),
        max_vocab=200_000,
        min_count=(1, 2, 3),
        order=4,
    )

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
import functools
import gzip
import json
import math
import os
import tempfile

import numpy as np

# Defaults for decoding with the LM, tuned with tune_lm.py for the bigram LM
# trained on the LibriSpeech LM corpus and the LibriSpeech streaming CTC
# model, on its dev set. See docs/lm-tuning.json.
LM_WEIGHT = 0.5
WORD_BONUS = 4.0
UNK_PENALTY = 0.0

# The number of word ids to process at once while reading the corpus.
CHUNK = 20_000_000
# The memory for counting, in bytes, and about how much an n-gram takes
# while it is counted. The n-grams of each order are counted in as many
# parts as this needs. Counted in parts of 80 bytes per n-gram, a 4-gram
# LM of 106 million words took 3.2 GB.
MEMORY = 2e9
BYTES_PER_NGRAM = 120

# N-grams are stored as 64-bit hashes of their word ids. Of a billion
# distinct n-grams, about 0.03 pairs would share a hash.
_MASK = (1 << 64) - 1
_SEED = 0x2545F4914F6CDD1D


def _mix(x):
    """The splitmix64 finalizer, a bijection of 64-bit integers."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def _mix_np(x):
    """_mix for an array of uint64, whose arithmetic wraps the same way."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _hash(ids):
    """
    The hash of a sequence of word ids, built from its last word to its
    first, so the hashes of the sequences that end the same share the
    work: the hash of (a, b, c) mixes a into the hash of (b, c).
    """
    h = _SEED
    for i in reversed(ids):
        h = _mix(h ^ i)
    return h


def _hash_np(columns):
    """_hash of each row of the columns, uint64 arrays of word ids."""
    h = np.full(len(columns[0]) if columns else 0, _SEED, dtype=np.uint64)
    for column in reversed(columns):
        h = _mix_np(h ^ column)
    return h


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
    A word-level n-gram language model with interpolated modified
    Kneser-Ney smoothing (Chen and Goodman, 1998).

    The probability of a word w after the words h is

        P(w | h) = max(c(hw) - D(c(hw)), 0) / c(h•) + gamma(h) * P(w | h')

    where h' is h without its first word, c(h•) the count of h followed by
    any word, and D one of three discounts, for n-grams seen once, twice and
    more often, estimated from the counts of counts of each order. The
    highest order counts how often each n-gram occurs. The lower orders
    count the different words seen before it instead, its continuation
    count, so a word that follows only one phrase, such as "francisco",
    gets little probability where the phrase was not said. N-grams that
    start a sentence have no word before them, so they keep their counts.
    The unigram distribution is interpolated with the uniform one.

    gamma(h) is the probability the kept n-grams after h leave over. N-grams
    seen fewer times than the order's min_count are not kept, so their
    probability goes to the lower orders, which makes the model smaller. A
    word outside the vocabulary gets the probability of <unk>, which counts
    all such words in the training text.
    """

    SOS = "<s>"  # start-of-sentence
    EOS = "</s>"  # end-of-sentence
    UNK = "<unk>"  # words outside the vocabulary while training

    def __init__(self, sentences=None, max_vocab=None, min_count=1, order=2):
        """
        Build an n-gram LM from text sentences.

        Arguments:
            sentences: An iterable of strings, each a whitespace-
                separated sentence. It is iterated twice, so a large
                corpus should be a :class:`TextCorpus`. If *None*,
                creates an empty model (use :meth:`load` to restore a
                saved one).
            max_vocab (int, optional): Keep only the most frequent
                words. The others count as one unknown word.
            min_count (int or sequence): Drop the n-grams of each order
                from bigrams up whose count, or continuation count below
                the highest order, is lower. Their probability goes to the
                lower orders, which makes the model smaller. One number
                applies to every order.
            order (int): The length of the longest n-grams, at least 2.
        """
        if order < 2:
            raise ValueError("The order must be at least 2.")
        self.order = order
        self._set_vocab([self.SOS, self.EOS, self.UNK])
        # The unigram probabilities, and for each order from 2 up the
        # hashes of the kept n-grams with their discounted probabilities
        # and the hashes of their contexts with their backoff weights.
        self.unigram = np.zeros(len(self.words))
        self.tables = {}
        if sentences is not None:
            if isinstance(min_count, int):
                min_count = [min_count] * (order - 1)
            if len(min_count) != order - 1:
                raise ValueError(f"Give a min_count for each order from 2 to {order}.")
            self._train(sentences, max_vocab, list(min_count))
        self._cache()

    def _set_vocab(self, words):
        self.words = words
        # Maps each word to its id.
        self.vocab = {w: i for i, w in enumerate(words)}

    def _cache(self):
        self._log_prob_ids = functools.lru_cache(maxsize=1 << 20)(self._log_prob)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, sentences, max_vocab, min_count):
        """Count the n-grams of *sentences*."""
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
            self.stats = {}
            self.unigram = self._unigram(ids, tmp)
            for n in range(2, self.order + 1):
                self.tables[n] = self._order(ids, n, min_count[n - 2], tmp)
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

    def _windows(self, ids, length):
        """
        Yields the start of each chunk of windows of the length, the
        columns of their word ids as uint64, and which of them lie within a
        sentence.
        """
        eos = self.vocab[self.EOS]
        starts = len(ids) - length + 1
        for start in range(0, max(starts, 0), CHUNK):
            stop = min(start + CHUNK, starts)
            columns = [
                np.asarray(ids[start + j : stop + j], dtype=np.int64)
                for j in range(length)
            ]
            # Only the last word of a window can end a sentence.
            valid = np.ones(stop - start, dtype=bool)
            for column in columns[:-1]:
                valid &= column != eos
            yield start, [c.astype(np.uint64) for c in columns], valid

    def _parts(self, ids, length, key, path):
        """
        Assigns each window of the length to a part by the hash of its
        columns key, a range, so the windows of a part fit in memory.
        Writes the part of each window to path, the number of parts for one
        outside a sentence, and returns the number of parts.
        """
        parts = max(1, math.ceil(len(ids) * BYTES_PER_NGRAM / MEMORY))
        dtype = np.uint8 if parts < 255 else np.uint32
        out = np.memmap(path, dtype=dtype, mode="w+", shape=max(len(ids), 1))
        # The last positions start no window.
        out[max(len(ids) - length + 1, 0) :] = parts
        for start, columns, valid in self._windows(ids, length):
            part = _hash_np(columns[key]) % np.uint64(parts)
            part[~valid] = parts
            out[start : start + len(part)] = part
        out.flush()
        return parts, out

    def _part_windows(self, ids, length, parts, part):
        """The columns of the windows of the length in the part."""
        found = []
        for start in range(0, len(parts), CHUNK):
            found.append(np.flatnonzero(parts[start : start + CHUNK] == part) + start)
        starts = np.concatenate(found)
        return [np.asarray(ids[starts + j]).astype(np.uint64) for j in range(length)]

    def _unigram(self, ids, tmp):
        """
        The unigram probabilities from the continuation counts, the number
        of different words seen before each word.
        """
        V = len(self.words)
        counts = np.zeros(V, dtype=np.int64)
        parts, part_of = self._parts(ids, 2, slice(0, 2), os.path.join(tmp, "parts"))
        for part in range(parts):
            columns = self._part_windows(ids, 2, part_of, part)
            _, first = np.unique(_hash_np(columns), return_index=True)
            counts += np.bincount(columns[1][first].astype(np.int64), minlength=V)
        del part_of

        n = np.bincount(np.minimum(counts, 5), minlength=6)[1:5]
        discounts = _discounts(n)
        seen = counts > 0
        d = discounts[np.minimum(counts[seen], 3) - 1]
        total = counts.sum()
        p = np.zeros(V)
        p[seen] = (counts[seen] - d) / total
        # The discounted probability goes to all words but <s>, which is
        # never predicted.
        p[1:] += d.sum() / total / (V - 1)
        self.stats[1] = {"ngrams": int(seen.sum()), "discounts": discounts}
        return p

    def _order(self, ids, n, min_count, tmp):
        """
        Counts the n-grams of order n, from the windows of n words at the
        highest order, and from the windows of n + 1 words below it, whose
        different first words are the continuation counts, with the n-grams
        that start a sentence counted as they are. The windows are split
        into parts by the hash of the n-gram's context, so each part has
        every n-gram of its contexts.
        """
        unk = self.vocab[self.UNK]
        highest = n == self.order
        length = n if highest else n + 1
        first = 0 if highest else 1
        context = slice(first, first + n - 1)
        parts, part_of = self._parts(ids, length, context, os.path.join(tmp, "parts"))

        if not highest:
            # The n-grams that start a sentence, by part.
            sos = self.vocab[self.SOS]
            starts, start_parts = [], []
            for start, columns, valid in self._windows(ids, n):
                mine = valid & (columns[0] == sos)
                starts.append(np.flatnonzero(mine) + start)
                h = _hash_np([c[mine] for c in columns[: n - 1]])
                start_parts.append(h % np.uint64(parts))
            starts = np.concatenate(starts)
            start_parts = np.concatenate(start_parts)

        n_counts = np.zeros(4, dtype=np.int64)
        kept_keys, kept_counts, kept_contexts = [], [], []
        context_keys, context_stats = [], []
        for part in range(parts):
            columns = self._part_windows(ids, length, part_of, part)
            grams = columns[first:]
            keys = _hash_np(grams)
            if highest:
                keys, index, counts = np.unique(
                    keys, return_index=True, return_counts=True
                )
                grams = [g[index] for g in grams]
            else:
                # Each different window adds one to its n-gram.
                _, distinct = np.unique(_hash_np(columns), return_index=True)
                keys, index, counts = np.unique(
                    keys[distinct], return_index=True, return_counts=True
                )
                grams = [g[distinct[index]] for g in grams]
                mine = starts[start_parts == np.uint64(part)]
                started = [
                    np.asarray(ids[mine + j]).astype(np.uint64) for j in range(n)
                ]
                s_keys, s_index, s_counts = np.unique(
                    _hash_np(started), return_index=True, return_counts=True
                )
                keys = np.concatenate([keys, s_keys])
                counts = np.concatenate([counts, s_counts])
                grams = [
                    np.concatenate([g, s[s_index]]) for g, s in zip(grams, started)
                ]
            n_counts += np.bincount(np.minimum(counts, 5), minlength=6)[1:5]

            contexts = _hash_np(grams[:-1])
            unique_contexts, inverse = np.unique(contexts, return_inverse=True)
            totals = np.bincount(inverse, weights=counts)
            # N-grams ending in an unknown word are left to the lower orders,
            # so the unknown word gets only the unigram probability.
            kept = (counts >= min_count) & (grams[-1] != np.uint64(unk))
            classes = np.minimum(counts[kept], 3)
            stats = np.zeros((len(unique_contexts), 5))
            stats[:, 0] = totals
            stats[:, 1] = np.bincount(
                inverse[kept], weights=counts[kept], minlength=len(unique_contexts)
            )
            for c in (1, 2, 3):
                stats[:, 1 + c] = np.bincount(
                    inverse[kept][classes == c], minlength=len(unique_contexts)
                )
            has_kept = stats[:, 1] > 0
            context_keys.append(unique_contexts[has_kept])
            context_stats.append(stats[has_kept])
            kept_keys.append(keys[kept])
            kept_counts.append(counts[kept])
            kept_contexts.append(contexts[kept])
        del part_of

        discounts = _discounts(n_counts)
        context_keys = np.concatenate(context_keys)
        context_stats = np.concatenate(context_stats)
        order = np.argsort(context_keys)
        context_keys, context_stats = context_keys[order], context_stats[order]
        totals, sums = context_stats[:, 0], context_stats[:, 1]
        left = sums - context_stats[:, 2:5] @ discounts
        gammas = 1 - left / totals

        keys = np.concatenate(kept_keys)
        counts = np.concatenate(kept_counts)
        contexts = np.concatenate(kept_contexts)
        probs = (counts - discounts[np.minimum(counts, 3) - 1]) / totals[
            np.searchsorted(context_keys, contexts)
        ]
        order = np.argsort(keys)
        self.stats[n] = {"ngrams": len(keys), "discounts": discounts}
        return {
            "keys": keys[order],
            "probs": probs[order],
            "contexts": context_keys,
            "gammas": gammas,
        }

    @classmethod
    def from_data_json(cls, data_json, order=2):
        """
        Train a :class:`WordLM` from a data JSON file that the rest of
        the speech project uses.  Each line in the file is a JSON object
        with a ``"text"`` field.

        Arguments:
            data_json (str): Path to the JSON-lines data file.
            order (int): The length of the longest n-grams.

        Returns:
            A trained :class:`WordLM` instance.
        """
        return cls(TextCorpus([data_json]), order=order)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def log_prob(self, word, context=None):
        """
        Return the log probability of *word* after the words of
        *context*, of which the last order - 1 count.

        Arguments:
            word (str): The word to score.
            context (str, sequence of str or None): The preceding word,
                or words with the last one last. None scores the word
                on its own, by its unigram probability.

        Returns:
            float: log-probability (base e) of the word.
        """
        if context is None:
            context = ()
        elif isinstance(context, str):
            context = (context,)
        unk = self.vocab[self.UNK]
        # The last order - 1 words count.
        context = context[max(len(context) - self.order + 1, 0) :]
        ids = tuple(self.vocab.get(w, unk) for w in context)
        return self._log_prob_ids(self.vocab.get(word, unk), ids)

    def _log_prob(self, w, context):
        p = self.unigram[w]
        # The hashes of the longer and longer ends of the context, and of
        # the n-grams they make with the word.
        context_hash = _SEED
        ngram_hash = _mix(_SEED ^ w)
        for j in range(1, len(context) + 1):
            v = context[-j]
            context_hash = _mix(context_hash ^ v)
            ngram_hash = _mix(ngram_hash ^ v)
            table = self.tables.get(j + 1)
            if table is None:
                break
            i = _find(table["contexts"], context_hash)
            if i is None:
                # The context has no kept n-grams, so all its probability
                # is the lower order's.
                continue
            p *= table["gammas"][i]
            k = _find(table["keys"], ngram_hash)
            if k is not None:
                p += table["probs"][k]
        return math.log(p) if p > 0 else -math.inf

    def perplexity(self, sentences):
        """The perplexity of the model on *sentences*, including </s>."""
        total = 0.0
        n = 0
        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            context = [self.SOS]
            for w in [*words, self.EOS]:
                total += self.log_prob(w, context)
                context.append(w)
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
        arrays = {}
        for n, table in self.tables.items():
            for name, values in table.items():
                arrays[f"{name}{n}"] = values
        with open(path, "wb") as fid:
            np.savez(
                fid,
                words=np.array("\n".join(self.words)),
                order=np.array(self.order),
                unigram=self.unigram,
                **arrays,
            )

    @classmethod
    def load(cls, path):
        """
        Load a language model from a NumPy .npz file, also a bigram LM
        saved before n-gram LMs, which it converts.
        """
        lm = cls()
        with np.load(path) as data:
            lm._set_vocab(str(data["words"]).split("\n"))
            if "order" not in data:
                lm._load_bigram(data)
            else:
                lm.order = int(data["order"])
                lm.unigram = data["unigram"]
                lm.tables = {
                    n: {
                        name: data[f"{name}{n}"]
                        for name in ("keys", "probs", "contexts", "gammas")
                    }
                    for n in range(2, lm.order + 1)
                }
        lm._cache()
        return lm

    def _load_bigram(self, data):
        """
        Converts a bigram LM with interpolated absolute discounting, where

            P(w | v) = max(c(v, w) - D, 0) / c(v) + backoff(v) * P(w)

        and P(w) is add-one smoothed, to the tables of the n-gram LM.
        """
        V = len(self.words)
        unigram = data["unigram"]
        history = data["history"]
        keys = data["keys"]
        discount = float(data["discount"])
        self.order = 2
        self.unigram = (unigram + 1) / (unigram.sum() + V)
        v, w = keys // V, keys % V
        ngrams = _hash_np([v.astype(np.uint64), w.astype(np.uint64)])
        order = np.argsort(ngrams)
        contexts = np.flatnonzero(history > 0)
        context_keys = _hash_np([contexts.astype(np.uint64)])
        context_order = np.argsort(context_keys)
        self.tables[2] = {
            "keys": ngrams[order],
            "probs": ((data["counts"] - discount) / history[v])[order],
            "contexts": context_keys[context_order],
            "gammas": data["backoff"][contexts][context_order],
        }


# The discounts when the counts of counts give none in range, as in
# KenLM's --discount_fallback.
FALLBACK_DISCOUNTS = (0.5, 1.0, 1.5)


def _discounts(n):
    """
    The discounts of n-grams seen once, twice and more often, from the
    numbers of n-grams seen once to four times (Chen and Goodman, 1998).
    A small corpus can lack some of these counts or give a discount that
    takes all of a count, such as 2 for the n-grams seen twice when none
    was seen three times, and then gets the fallback discounts.
    """
    n1, n2, n3, n4 = (float(x) for x in n)
    if not (n1 and n2 and n3):
        return np.array(FALLBACK_DISCOUNTS)
    y = n1 / (n1 + 2 * n2)
    d = [1 - 2 * y * n2 / n1, 2 - 3 * y * n3 / n2, 3 - 4 * y * n4 / n3]
    if not all(0 < x < c for c, x in zip((1, 2, 3), d)):
        return np.array(FALLBACK_DISCOUNTS)
    return np.array(d)


def _find(keys, key):
    """The index of key in the sorted uint64 keys, or None."""
    i = int(np.searchsorted(keys, np.uint64(key)))
    if i < len(keys) and int(keys[i]) == key:
        return i
    return None


class WordScorer:
    """
    Scores the words of a growing character sequence with a
    :class:`WordLM` for the CTC beam search.

    A word is scored when it is complete, that is when a space follows
    it or the utterance ends, as ``weight * log P(word | previous words) +
    word_bonus``, plus unk_penalty for a word outside the vocabulary. The
    state of a prefix is its last order - 1 complete words and the
    characters of the word in progress.
    """

    def __init__(self, lm, char_to_int, weight, word_bonus, unk_penalty):
        self.lm = lm
        self.weight = weight
        self.word_bonus = word_bonus
        self.unk_penalty = unk_penalty
        self.int_to_char = {v: k for k, v in char_to_int.items()}
        self.space = char_to_int[" "]
        self.history = lm.order - 1

    def initial(self):
        return ((WordLM.SOS,), "")

    def extend(self, state, label):
        """Returns the state after the label and the score it adds."""
        context, word = state
        if label != self.space:
            return (context, word + self.int_to_char[label]), 0.0
        if not word:
            # Leading or repeated spaces complete no word.
            return state, 0.0
        return self._next(context, word), self.word_score(word, context)

    def finish(self, state):
        """Returns the score of ending the utterance in this state."""
        context, word = state
        score = 0.0
        if word:
            score += self.word_score(word, context)
            context, _ = self._next(context, word)
        return score + self.weight * self.lm.log_prob(WordLM.EOS, context)

    def _next(self, context, word):
        return ((*context, word)[-self.history :], "")

    def word_score(self, word, context):
        score = self.weight * self.lm.log_prob(word, context) + self.word_bonus
        if word not in self.lm.vocab or word == WordLM.UNK:
            score += self.unk_penalty
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a word n-gram LM from text. Inputs are dataset json "
        "files, or text files with one sentence per line, which may be gzipped."
    )
    parser.add_argument("inputs", nargs="+", help="The text to train on.")
    parser.add_argument("output", help="The .npz file to save the LM to.")
    parser.add_argument(
        "--order", type=int, default=4, help="The length of the longest n-grams."
    )
    parser.add_argument(
        "--max-vocab", type=int, help="Keep only the most frequent words."
    )
    parser.add_argument(
        "--min-count",
        default="1",
        help="Drop the n-grams seen fewer times, from bigrams up, comma "
        "separated, such as 1,2,3 for a 4-gram LM, or one number for all, to "
        "make the model smaller. Below the highest order the count is of the "
        "different words seen before the n-gram.",
    )
    parser.add_argument(
        "--memory",
        type=float,
        default=MEMORY / 1e9,
        help="The memory to count the n-grams in, in GB. Less counts in more "
        "parts, each a pass over the corpus.",
    )
    parser.add_argument("--dev", help="A dataset json to report the perplexity on.")
    args = parser.parse_args()

    MEMORY = args.memory * 1e9
    min_count = [int(x) for x in args.min_count.split(",")]
    if len(min_count) == 1:
        min_count = min_count[0]
    lm = WordLM(TextCorpus(args.inputs), args.max_vocab, min_count, args.order)
    lm.save(args.output)
    print(f"Trained on a vocabulary of {len(lm.words):,} words.")
    for n, stats in lm.stats.items():
        d = ", ".join(f"{x:.2f}" for x in stats["discounts"])
        print(f"Order {n}: {stats['ngrams']:,} n-grams, discounts {d}")
    if args.dev is not None:
        print(f"Dev perplexity {lm.perplexity(TextCorpus([args.dev])):.1f}")
