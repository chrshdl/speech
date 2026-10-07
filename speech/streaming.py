"""
Streaming transcription: feeds audio to a CTC model chunk by chunk, as it
arrives from a microphone, and decodes greedily or with a beam search and
a language model.
"""

import math

import numpy as np
import torch
from torch import nn

from speech import loader
from speech.models.ctc_decoder import BEAM_SIZE, PRUNE, BeamSearch
from speech.models.word_lm import LM_WEIGHT, UNK_PENALTY, WORD_BONUS, WordLM

# The window of the training features in milliseconds, see log_specgram.
WINDOW_MS = 20


def model_sample_rate(preproc):
    """
    The sample rate the model was trained on, from the number of
    frequency bins in each feature frame.
    """
    return round((preproc.input_dim - 1) * 2 * 1000 / WINDOW_MS)


def search_factory(
    blank,
    char_to_int,
    lm_path=None,
    beam_size=None,
    prune=PRUNE,
    lm_weight=LM_WEIGHT,
    word_bonus=WORD_BONUS,
    unk_penalty=UNK_PENALTY,
):
    """
    Returns a function that creates a beam search for one stream, or
    returns None for greedy decoding, which is used without an LM and
    with a beam size of 1, the default then.
    """
    lm = None
    if lm_path is not None:
        lm = WordLM.load(lm_path).scorer(
            char_to_int, lm_weight, word_bonus, unk_penalty
        )
    beam_size = beam_size or (BEAM_SIZE if lm is not None else 1)
    if lm is None and beam_size == 1:
        return lambda: None
    return lambda: BeamSearch(blank, beam_size, lm, prune)


class Transcriber:
    def __init__(self, model, preproc, sample_rate, search=None):
        """
        Transcribes a stream of audio chunk by chunk. Decodes greedily,
        or with the given ctc_decoder.BeamSearch, which can include a
        language model.
        """
        self.model = model
        self.preproc = preproc
        self.search = search
        self.specgram = loader.SpecgramStream(sample_rate, window_size=WINDOW_MS)
        self.state = None
        self.greedy = []
        self.prev = model.blank
        # The number of blank frames since the last label.
        self.blanks = 0

        # Each encoded frame covers the feature hop times the time stride
        # of the convolutions.
        stride = math.prod(c.stride[0] for c in model.conv if isinstance(c, nn.Conv2d))
        self.frame_seconds = stride * self.specgram.hop / sample_rate

    def push(self, audio, final=False):
        """
        Adds the next chunk of audio samples. Pass final with the last
        chunk to flush the frames held back for the lookahead.
        """
        frames = self.specgram.push(audio)
        frames = (frames - self.preproc.mean) / self.preproc.std
        frames = torch.from_numpy(frames).float().unsqueeze(0)
        probs, self.state = self.model.stream(frames, self.state, final)
        probs = probs[0].cpu().numpy()

        # Greedy CTC decoding: merge repeats, then drop blanks. It also
        # tracks pauses for the beam search.
        for p in probs.argmax(axis=1).tolist():
            if p != self.prev and p != self.model.blank:
                self.greedy.append(p)
            self.blanks = self.blanks + 1 if p == self.model.blank else 0
            self.prev = p

        if self.search is not None:
            with np.errstate(divide="ignore"):
                log_probs = np.log(probs)
            for frame in log_probs.tolist():
                self.search.step(frame)

    def labels(self, final=False):
        """
        The best labels so far. With final, a language model also scores
        the end of the utterance, such as its last word.
        """
        if self.search is None:
            return self.greedy
        return list(self.search.best(final)[0])

    def text(self, final=False):
        return "".join(self.preproc.decode(self.labels(final)))

    @property
    def pause_seconds(self):
        """The time since the last label."""
        return self.blanks * self.frame_seconds

    def take_line(self, pause):
        """
        Returns the finished line and starts a new one, once there is text
        and a pause of at least the given seconds, or returns None. Pauses
        can decode to spaces at the ends of a line, which are stripped.
        """
        if not self.text().strip() or self.pause_seconds < pause:
            return None
        line = self.text(final=True).strip()
        self.clear()
        return line

    def clear(self):
        """Starts a new transcript, keeping the stream going."""
        self.greedy = []
        self.blanks = 0
        if self.search is not None:
            self.search.reset()
