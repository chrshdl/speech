"""
Recognizes what a short utterance asks for, such as the weather, from a
CTC model's output probabilities.

Each intent has sample utterances. An utterance matches a sample when the
model's output can be aligned to the sample's characters nearly as well
as to its own best path. Aligning the model's output rather than
comparing the transcript recognizes an intent also when the transcript
is misspelled, as it often is for short commands.

A sample can have a slot, such as "what's the weather in {city}", whose
values the intent lists. The value is found in two steps: the sample is
aligned with a wildcard in place of the slot, which finds the frames
where the slot was said, and each value is aligned to those frames. The
values that fit best then compete as whole sentences with the samples.
"""

import dataclasses
import re

import numpy as np
from rapidfuzz import fuzz

# The lowest mean log probability per character, relative to the best
# path, of a match. On synthesized questions for the weather in 16
# voices, -2 matched 99% of them, 4% of other short commands, and none of
# 300 LibriSpeech dev sentences.
THRESHOLD = -2.0
# Utterances longer than this are not commands, so are not matched, which
# also bounds the time to match them.
MAX_SECONDS = 8.0
# How many of a slot's values, those that fit the slot's frames best, are
# aligned as whole sentences.
CANDIDATES = 3

SLOT = re.compile(r"\{(\w+)\}")
# Stands for a slot in a label sequence: any label, scored as each frame's
# best.
WILDCARD = -1


@dataclasses.dataclass(frozen=True)
class Intent:
    name: str
    # Phrasings of the request, in the model's characters, such as
    # "what's the weather". A slot, such as {city}, stands for any of its
    # values, one slot per sample.
    samples: tuple[str, ...]
    # The values of each slot, such as {"city": ("berlin", "hamburg")}.
    slots: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Match:
    intent: Intent
    # The mean log probability per character of the best sample, with its
    # slot filled in, relative to the best path, so at most 0.
    score: float
    sample: str
    # The values of the sample's slots, such as {"city": "hamburg"}.
    slots: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _Sample:
    intent: Intent
    text: str
    # The labels of a sample without a slot.
    labels: np.ndarray | None = None
    # For a sample with a slot, its name, the labels before and after it,
    # and its values with their labels.
    slot: str | None = None
    before: np.ndarray | None = None
    after: np.ndarray | None = None
    values: list | None = None


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

        def labels(text):
            unknown = set(text) - set(char_to_int)
            if unknown:
                raise ValueError(f"{text!r} has characters {unknown} the model lacks")
            return np.array([char_to_int[c] for c in text], dtype=np.int64)

        self.samples = []
        for intent in intents:
            values = {
                name: [(value, labels(value)) for value in slot_values]
                for name, slot_values in intent.slots.items()
            }
            for text in intent.samples:
                slots = SLOT.findall(text)
                if not slots:
                    self.samples.append(_Sample(intent, text, labels(text)))
                    continue
                if len(slots) > 1 or not values.get(slots[0]):
                    raise ValueError(f"{text!r} needs one slot with values")
                before, after = SLOT.split(text)[::2]
                self.samples.append(
                    _Sample(
                        intent,
                        text,
                        slot=slots[0],
                        before=labels(before),
                        after=labels(after),
                        values=values[slots[0]],
                    )
                )

    def recognize(self, probs):
        """
        Returns the Match of the intent the utterance asks for, or None.

        Arguments:
            probs (ndarray): The model's output probabilities for the whole
                utterance with shape (frames, labels + 1).
        """
        match = self.closest(probs)
        return match if match is not None and match.score >= self.threshold else None

    def closest_heard(self, transcriber):
        """The closest Match to what a streaming.Transcriber heard."""
        return self.closest(transcriber.probs())

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

        # Each candidate is a sample, its labels and its slot's value.
        candidates = []
        for sample in self.samples:
            if sample.slot is None:
                candidates.append((sample, sample.labels, None))
            else:
                candidates.extend(self._slot_candidates(relative, sample))
        if not candidates:
            return None
        sequences = [labels for _, labels, _ in candidates]
        scores = viterbi(relative, sequences, self.blank)
        scores /= [len(labels) for labels in sequences]
        best = int(np.argmax(scores))
        sample, _, value = candidates[best]
        slots = {sample.slot: value} if value is not None else {}
        return Match(sample.intent, float(scores[best]), sample.text, slots)

    def _slot_candidates(self, relative, sample):
        """
        The sample with each of the values of its slot that fit the frames
        where the slot was said best.
        """
        wildcard = np.concatenate([sample.before, [WILDCARD], sample.after])
        score, path = viterbi(relative, [wildcard], self.blank, path=True)
        if not np.isfinite(score[0]):
            # The utterance is too short for the sample.
            return []
        frames = np.flatnonzero(_states(wildcard, self.blank)[path] == WILDCARD)
        said = relative[frames[0] : frames[-1] + 1]
        values = [labels for _, labels in sample.values]
        fit = viterbi(said, values, self.blank) / [len(labels) for labels in values]
        return [
            (
                sample,
                np.concatenate([sample.before, values[i], sample.after]),
                sample.values[i][0],
            )
            for i in np.argsort(-fit)[:CANDIDATES]
        ]


def _states(labels, blank):
    """The CTC states of a label sequence: its labels with blanks around them."""
    states = np.full(2 * len(labels) + 1, blank)
    states[1::2] = labels
    return states


def viterbi(relative, sequences, blank, path=False):
    """
    The log probability of the best CTC alignment of each label sequence
    to all the frames.

    Arguments:
        relative (ndarray): The log probabilities of each frame minus its
            largest, with shape (frames, labels + 1), so the best path
            scores 0.
        sequences (list): Label sequences. WILDCARD in one stands for any
            label, scored as each frame's best.
        blank (int): The blank label.
        path (bool): Also return the states of the first sequence's best
            alignment, one per frame, as indexes into its CTC states.

    Returns an array with a log probability per sequence, and with path the
    alignment.
    """
    n = len(sequences)
    width = 2 * max(len(labels) for labels in sequences) + 1
    # Each sequence's states, padded with states no alignment can enter.
    states = np.full((n, width), blank)
    valid = np.zeros((n, width), bool)
    for i, labels in enumerate(sequences):
        sequence_states = _states(labels, blank)
        states[i, : len(sequence_states)] = sequence_states
        valid[i, : len(sequence_states)] = True
    # A label can follow the label before the previous blank, unless the
    # two are the same.
    skip = np.zeros((n, width), bool)
    skip[:, 2:] = (states[:, 2:] != blank) & (states[:, 2:] != states[:, :-2])
    wildcard = states == WILDCARD
    columns = np.where(wildcard, 0, states)

    def emissions(frame):
        return np.where(valid, np.where(wildcard, 0.0, frame[columns]), -np.inf)

    score = np.full((n, width), -np.inf)
    score[:, :2] = emissions(relative[0])[:, :2]
    moves = np.full((3, n, width), -np.inf)
    back = np.zeros((len(relative), width), np.int8) if path else None
    for t in range(1, len(relative)):
        # Stay in a state, advance by one, or skip a blank.
        moves[0] = score
        moves[1, :, 1:] = score[:, :-1]
        moves[2, :, 2:] = np.where(skip[:, 2:], score[:, :-2], -np.inf)
        best = moves.argmax(axis=0)
        score = np.take_along_axis(moves, best[None], axis=0)[0] + emissions(
            relative[t]
        )
        if path:
            back[t] = best[0]

    # An alignment ends in the last label or the blank after it.
    last = np.array([2 * len(labels) for labels in sequences])
    rows = np.arange(n)
    ends = np.stack([score[rows, last], score[rows, np.maximum(last - 1, 0)]])
    final = ends.max(axis=0)
    if not path:
        return final
    state = last[0] - int(ends[1, 0] > ends[0, 0])
    alignment = [state]
    for t in range(len(relative) - 1, 0, -1):
        state -= back[t, state]
        alignment.append(state)
    return final, np.array(alignment[::-1])


def alignment_score(relative, labels, blank):
    """The log probability of the best alignment of a label sequence."""
    return float(viterbi(relative, [labels], blank)[0])


# The lowest score of a match by TextIntentRecognizer: a similarity of a
# transcript to a sample of 0.8, minus 1.
TEXT_THRESHOLD = -0.2


# Words before a sample that TextIntentRecognizer passes over, such as
# "hello unicorn" in "hello unicorn, what's the weather?", at most this
# many, each lowering the similarity by the penalty.
LEADING_WORDS = 3
LEADING_PENALTY = 0.02


class TextIntentRecognizer:
    def __init__(self, intents, threshold=TEXT_THRESHOLD):
        """
        Recognizes intents in transcripts, such as from Whisper, rather
        than in a model's output. A transcript matches a sample when they
        are similar, which allows for small mistakes such as "whether"
        for "weather", also after a few other words, such as a greeting.
        The words where a sample has a slot are its value, whatever they
        are.
        """
        self.intents = intents
        self.threshold = threshold

    def recognize(self, text):
        """Returns the Match of the intent the text asks for, or None."""
        match = self.closest(text)
        return match if match is not None and match.score >= self.threshold else None

    def closest_heard(self, transcriber):
        """The closest Match to the final transcript of a transcriber."""
        return self.closest(transcriber.text(final=True))

    def closest(self, text):
        """
        Returns the Match of the intent closest to the text, or None for
        an empty text. Its score is the similarity minus 1, so at most 0.
        """
        words = normalize(text).split()
        if not words:
            return None
        best = None
        for skip in range(min(LEADING_WORDS, len(words) - 1) + 1):
            for intent in self.intents:
                for sample in intent.samples:
                    similarity, slots = _similarity(words[skip:], sample)
                    similarity -= skip * LEADING_PENALTY
                    if best is None or similarity > best[0]:
                        best = (similarity, intent, sample, slots)
        similarity, intent, sample, slots = best
        return Match(intent, similarity - 1, sample, slots)


def normalize(text):
    """Lowercase words without punctuation, keeping apostrophes."""
    text = text.lower().replace("’", "'")
    return " ".join(re.sub(r"[^\w' ]", " ", text).split())


def _similarity(words, sample):
    """
    The similarity of the words to the sample, and the values of its
    slots: for a sample with a slot, at the best split of the words into
    the parts before, of and after the slot.
    """
    text = " ".join(words)
    slot = SLOT.search(sample)
    if slot is None:
        return fuzz.ratio(text, sample) / 100, {}
    before, after = sample[: slot.start()].strip(), sample[slot.end() :].strip()
    best = (0.0, {})
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            fixed = " ".join(words[:i]) + " | " + " ".join(words[j:])
            similarity = fuzz.ratio(fixed, before + " | " + after) / 100
            if similarity > best[0]:
                best = (similarity, {slot.group(1): " ".join(words[i:j])})
    return best
