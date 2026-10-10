"""
Recognizes what a short utterance asks for, such as the weather, from a
CTC model's output probabilities.

Each intent has sample utterances. An utterance matches a sample when the
model's output can be aligned to the sample's characters nearly as well
as to its own best path. Aligning the model's output rather than
comparing the transcript recognizes an intent also when the transcript
is misspelled, as it often is for short commands.
"""

import dataclasses

import numpy as np

# The lowest mean log probability per character, relative to the best
# path, of a match. On synthesized questions for the weather in 16
# voices, -2 matched 99% of them, 2% of other short commands, and none of
# 300 LibriSpeech dev sentences.
THRESHOLD = -2.0
# Utterances longer than this are not commands, so are not matched, which
# also bounds the time to match them.
MAX_SECONDS = 8.0


@dataclasses.dataclass(frozen=True)
class Intent:
    name: str
    # Phrasings of the request, in the model's characters, such as
    # "what's the weather".
    samples: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Match:
    intent: Intent
    # The mean log probability per character of the best sample, relative
    # to the best path, so at most 0.
    score: float
    sample: str


class IntentRecognizer:
    def __init__(self, intents, char_to_int, blank, frame_seconds, threshold=THRESHOLD):
        """
        Arguments:
            intents (list): The Intents to recognize.
            char_to_int (dict): The model's labels.
            blank (int): The model's blank label.
            frame_seconds (float): The time each output frame covers.
            threshold (float): The lowest score of a match.
        """
        self.intents = intents
        self.blank = blank
        self.threshold = threshold
        self.max_frames = int(MAX_SECONDS / frame_seconds)
        self.samples = []
        for intent in intents:
            for sample in intent.samples:
                unknown = set(sample) - set(char_to_int)
                if unknown:
                    raise ValueError(
                        f"{sample!r} has characters {unknown} the model lacks"
                    )
                labels = np.array([char_to_int[c] for c in sample])
                self.samples.append((intent, sample, labels))

    def recognize(self, probs):
        """
        Returns the Match of the intent the utterance asks for, or None.

        Arguments:
            probs (ndarray): The model's output probabilities for the whole
                utterance with shape (frames, labels + 1).
        """
        match = self.closest(probs)
        return match if match is not None and match.score >= self.threshold else None

    def closest(self, probs):
        """
        Returns the Match of the intent closest to the utterance, also if
        it is not close enough, or None for an utterance that is empty or
        too long to be a command.
        """
        if len(probs) == 0 or len(probs) > self.max_frames:
            return None
        with np.errstate(divide="ignore"):
            log_probs = np.log(probs)
        relative = log_probs - log_probs.max(axis=1, keepdims=True)
        best = None
        for intent, sample, labels in self.samples:
            score = alignment_score(relative, labels, self.blank) / len(labels)
            if best is None or score > best.score:
                best = Match(intent, score, sample)
        return best


def alignment_score(relative, labels, blank):
    """
    The log probability of the best CTC alignment of the labels to all
    the frames, from relative log probabilities, the log probabilities
    minus each frame's largest. The best path scores 0.
    """
    # The labels with blanks around and between them, as in CTC.
    states = np.full(2 * len(labels) + 1, blank)
    states[1::2] = labels
    # A label can follow the label before the previous blank, unless the
    # two are the same.
    skip = np.zeros(len(states), bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])

    score = np.full(len(states), -np.inf)
    score[:2] = relative[0, states[:2]]
    for frame in relative[1:]:
        previous = score
        score = previous.copy()
        np.maximum(score[1:], previous[:-1], out=score[1:])
        score[2:][skip[2:]] = np.maximum(score[2:], previous[:-2])[skip[2:]]
        score += frame[states]
    return max(score[-1], score[-2])
