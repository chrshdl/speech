"""
Author: Awni Hannun

This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.

The algorithm is a prefix beam search for a model trained
with the CTC loss function. It runs frame by frame, so it can
decode a stream as it arrives, and it can add the scores of a
language model.

For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873

"""

import heapq
import math

import numpy as np

NEG_INF = -float("inf")

# Defaults for decoding with a language model. On the LibriSpeech dev set
# a wider beam barely helped and pruning at 1e-3 changed nothing while
# making the search three times faster.
BEAM_SIZE = 32
PRUNE = math.log(1e-3)


def logaddexp(a, b):
    """
    Stable log(exp(a) + exp(b)). The search calls this millions of times,
    so it takes two numbers rather than any number of them.
    """
    if a < b:
        a, b = b, a
    if b == NEG_INF:
        return a
    return a + math.log1p(math.exp(b - a))


class BeamSearch:
    def __init__(self, blank, beam_size=10, lm=None, prune=None):
        """
        A prefix beam search that consumes the output of a CTC model one
        frame at a time.

        Arguments:
          blank (int): Index of the CTC blank label.
          beam_size (int): The number of prefixes to keep.
          lm (optional): Scores prefixes with a language model. It needs
            the methods initial() -> state, extend(state, label) ->
            (state, score) and finish(state) -> score, where the scores
            are added to the log probability of a prefix. See
            WordLM.scorer.
          prune (float, optional): Skip labels whose log probability in
            a frame is below this, which speeds up the search.
        """
        self.blank = blank
        self.beam_size = beam_size
        self.lm = lm
        self.prune = prune
        self.reset()

    def reset(self):
        """Starts a new utterance."""
        state = self.lm.initial() if self.lm is not None else None
        # Each prefix maps to [p_blank, p_no_blank, lm_state, lm_score].
        # The first two are the CTC log probabilities of the prefix given
        # that it ends in a blank or not. The LM score is kept apart, so
        # merging the paths of a prefix never counts it twice.
        self.beam = {(): [0.0, NEG_INF, state, 0.0]}

    def step(self, log_probs):
        """
        Advances the search by one frame of output log probabilities
        with shape (output dim,).
        """
        next_beam = {}

        def entry(prefix, lm_state, lm_score):
            n = next_beam.get(prefix)
            if n is None:
                n = next_beam[prefix] = [NEG_INF, NEG_INF, lm_state, lm_score]
            return n

        labels = [
            (s, p)
            for s, p in enumerate(log_probs)
            if self.prune is None or p >= self.prune
        ]
        for prefix, (p_b, p_nb, lm_state, lm_score) in self.beam.items():
            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            p_total = logaddexp(p_b, p_nb)
            end_t = prefix[-1] if prefix else None
            for s, p in labels:
                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == self.blank:
                    n = entry(prefix, lm_state, lm_score)
                    n[0] = logaddexp(n[0], p_total + p)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated. The LM scores the extension once, when the
                # prefix is first created.
                n_prefix = prefix + (s,)
                if n_prefix not in next_beam and self.lm is not None:
                    n_state, delta = self.lm.extend(lm_state, s)
                    n = entry(n_prefix, n_state, lm_score + delta)
                else:
                    n = entry(n_prefix, None, lm_score)
                if s != end_t:
                    n[1] = logaddexp(n[1], p_total + p)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n[1] = logaddexp(n[1], p_b + p)

                    # If s is repeated at the end we also update the
                    # unchanged prefix. This is the merging case.
                    n = entry(prefix, lm_state, lm_score)
                    n[1] = logaddexp(n[1], p_nb + p)

        # Keep the best prefixes before moving on to the next time-step.
        score = self._score
        best = heapq.nlargest(self.beam_size, next_beam.items(), lambda x: score(x[1]))
        self.beam = dict(best)

    def best(self, final=False):
        """
        Returns the most likely prefix and its log score. When final the
        LM also scores the end of the utterance, such as its last word.
        """

        def score(item):
            total = self._score(item[1])
            if final and self.lm is not None:
                total += self.lm.finish(item[1][2])
            return total

        prefix, value = max(self.beam.items(), key=score)
        return prefix, score((prefix, value))

    @staticmethod
    def _score(value):
        p_b, p_nb, _, lm_score = value
        return logaddexp(p_b, p_nb) + lm_score


def best_path(labels, blank):
    """
    Greedy CTC decoding of the most likely label of each frame: merges
    repeats, then drops blanks.
    """
    seq = []
    prev = blank
    for p in labels:
        if p != blank and p != prev:
            seq.append(p)
        prev = p
    return seq


def decode(probs, beam_size=10, blank=0, lm=None, prune=None):
    """
    Performs inference for the given output probabilities.

    Arguments:
      probs: The output probabilities (e.g. post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.
      lm (optional): A language model scorer, see BeamSearch.
      prune (float, optional): See BeamSearch.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    search = BeamSearch(blank, beam_size, lm, prune)
    with np.errstate(divide="ignore"):
        log_probs = np.log(probs)
    for t in range(log_probs.shape[0]):
        search.step(log_probs[t].tolist())
    labels, score = search.best(final=True)
    return labels, -score


if __name__ == "__main__":
    np.random.seed(3)

    time = 50
    output_dim = 20

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    labels, score = decode(probs)
    print(labels)
    print(f"Score {score:.3f}")
