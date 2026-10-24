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

from speech import tokens

# The lowest mean log probability per label, relative to the best path,
# of a match. It depends on how confident the model is. On synthesized
# questions for the weather in 16 voices, -1.5 matched 99% of them with
# the 3,750 hour model of characters, 4% of other short commands, and none
# of 300 LibriSpeech dev sentences. The 100 hour model scores lower, and
# matched as many at -2. A model of bigrams has about half the labels per
# sample, so check the threshold for it.
THRESHOLD = -1.5
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
    # The mean log probability per label of the best sample, with its
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
    def __init__(
        self,
        intents,
        char_to_int,
        blank,
        frame_seconds,
        threshold=THRESHOLD,
        spaced=False,
        others=(),
    ):
        """
        Arguments:
            intents (list): The Intents to recognize.
            char_to_int (dict): The model's labels.
            blank (int): The model's blank label.
            frame_seconds (float): The time each output frame covers.
            threshold (float): The lowest score of a match.
            spaced (bool): Also match the samples with a space before or
                after them, which an utterance has when its transcript
                starts within a stream, such as each command after the
                first in a stream of commands.
            others (tuple): Phrases that compete with the intents' samples
                but match none, such as everyday words, so an utterance
                that sounds more like one of them asks for nothing.
        """
        self.intents = intents
        self.blank = blank
        self.threshold = threshold
        self.max_frames = int(MAX_SECONDS / frame_seconds)

        def labels(text):
            unknown = set(text) - set(char_to_int)
            if unknown:
                raise ValueError(f"{text!r} has characters {unknown} the model lacks")
            # Split as the model's transcripts were, into its bigrams if it
            # has them.
            labels = tokens.tokenize(text, char_to_int)
            return np.array([char_to_int[t] for t in labels], dtype=np.int64)

        self.samples = []
        self.other = Intent("", tuple(others))
        for intent in [*intents, self.other]:
            values = {
                name: [(value, labels(value)) for value in slot_values]
                for name, slot_values in intent.slots.items()
            }
            for text in intent.samples:
                slots = SLOT.findall(text)
                if slots and (len(slots) > 1 or not values.get(slots[0])):
                    raise ValueError(f"{text!r} needs one slot with values")
                spellings = [text]
                if spaced:
                    spellings += [" " + text, text + " ", " " + text + " "]
                for spelling in spellings:
                    if not slots:
                        sample = _Sample(intent, text, labels(spelling))
                    else:
                        before, after = SLOT.split(spelling)[::2]
                        sample = _Sample(
                            intent,
                            text,
                            slot=slots[0],
                            before=labels(before),
                            after=labels(after),
                            values=values[slots[0]],
                        )
                    self.samples.append(sample)

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

    def stream(self):
        """
        Matches an utterance as its frames come, such as a command as it is
        said, aligning each frame once, rather than all the frames each
        time, as closest does. Only for samples without slots.
        """
        return _Stream(self)

    def closest(self, probs):
        """
        Returns the Match of the intent closest to the utterance, also if
        it is not close enough, or None for an utterance that is empty, too
        long to be a command, or closest to one of the other phrases.
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
        if sample.intent is self.other:
            return None
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


class _Stream:
    """Matches an utterance as its frames come, see IntentRecognizer.stream."""

    def __init__(self, recognizer):
        if any(sample.slot is not None for sample in recognizer.samples):
            raise ValueError("Only samples without slots can be matched as a stream.")
        self.recognizer = recognizer
        sequences = [sample.labels for sample in recognizer.samples]
        self.alignments = Alignments(sequences, recognizer.blank)
        self.lengths = np.array([len(labels) for labels in sequences])
        # The frames pushed.
        self.frames = 0

    def push(self, probs):
        """Adds the utterance's next frames, as closest's probs."""
        self.frames += len(probs)
        # Too long to be matched, it is aligned no further.
        if len(probs) == 0 or self.frames > self.recognizer.max_frames:
            return
        with np.errstate(divide="ignore"):
            log_probs = np.log(probs)
        self.alignments.push(log_probs - log_probs.max(axis=1, keepdims=True))

    def closest(self):
        """As IntentRecognizer.closest, for the frames pushed."""
        recognizer = self.recognizer
        if self.frames == 0 or self.frames > recognizer.max_frames:
            return None
        scores = self.alignments.scores() / self.lengths
        best = int(np.argmax(scores))
        sample = recognizer.samples[best]
        if sample.intent is recognizer.other:
            return None
        return Match(sample.intent, float(scores[best]), sample.text)

    def closest_heard(self, transcriber):
        """
        As IntentRecognizer.closest_heard, pushing the frames the transcriber
        added since, so it must be the one transcriber, until it is cleared,
        which needs a new stream.
        """
        self.push(transcriber.probs()[self.frames :])
        return self.closest()


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
    alignments = Alignments(sequences, blank, path)
    alignments.push(relative)
    if not path:
        return alignments.scores()
    return alignments.scores(), alignments.path()


class Alignments:
    """
    The best CTC alignments of label sequences to frames that come in turn,
    such as those of a stream, see viterbi: each frame pushed extends them,
    so the frames before are not aligned again.
    """

    def __init__(self, sequences, blank, path=False):
        n = len(sequences)
        width = 2 * max(len(labels) for labels in sequences) + 1
        # Each sequence's states, padded with states no alignment can enter.
        states = np.full((n, width), blank)
        self.valid = np.zeros((n, width), bool)
        for i, labels in enumerate(sequences):
            sequence_states = _states(labels, blank)
            states[i, : len(sequence_states)] = sequence_states
            self.valid[i, : len(sequence_states)] = True
        # A label can follow the label before the previous blank, unless the
        # two are the same.
        self.skip = np.zeros((n, width), bool)
        self.skip[:, 2:] = (states[:, 2:] != blank) & (states[:, 2:] != states[:, :-2])
        self.wildcard = states == WILDCARD
        self.columns = np.where(self.wildcard, 0, states)
        # An alignment ends in the last label or the blank after it.
        self.last = np.array([2 * len(labels) for labels in sequences])
        self.score = None
        self.moves = np.full((3, n, width), -np.inf)
        # For path, each frame's move into each state of the first sequence.
        self.back = [] if path else None

    def _emissions(self, frame):
        return np.where(
            self.valid, np.where(self.wildcard, 0.0, frame[self.columns]), -np.inf
        )

    def push(self, relative):
        """Aligns the next frames, as viterbi's relative."""
        moves = self.moves
        for frame in relative:
            if self.score is None:
                self.score = np.full(self.valid.shape, -np.inf)
                self.score[:, :2] = self._emissions(frame)[:, :2]
                if self.back is not None:
                    self.back.append(np.zeros(self.valid.shape[1], np.int8))
                continue
            # Stay in a state, advance by one, or skip a blank.
            score = self.score
            moves[0] = score
            moves[1, :, 1:] = score[:, :-1]
            moves[2, :, 2:] = np.where(self.skip[:, 2:], score[:, :-2], -np.inf)
            best = moves.argmax(axis=0)
            self.score = np.take_along_axis(moves, best[None], axis=0)[
                0
            ] + self._emissions(frame)
            if self.back is not None:
                self.back.append(best[0].astype(np.int8))

    def _ends(self):
        rows = np.arange(len(self.last))
        return np.stack(
            [
                self.score[rows, self.last],
                self.score[rows, np.maximum(self.last - 1, 0)],
            ]
        )

    def scores(self):
        """The log probability of each sequence's best alignment so far."""
        if self.score is None:
            return np.full(len(self.last), -np.inf)
        return self._ends().max(axis=0)

    def path(self):
        """The states of the first sequence's best alignment so far."""
        ends = self._ends()
        state = self.last[0] - int(ends[1, 0] > ends[0, 0])
        alignment = [state]
        for t in range(len(self.back) - 1, 0, -1):
            state -= self.back[t][state]
            alignment.append(state)
        return np.array(alignment[::-1])


def alignment_score(relative, labels, blank):
    """The log probability of the best alignment of a label sequence."""
    return float(viterbi(relative, [labels], blank)[0])
