"""
Streaming transcription: feeds audio to a CTC model chunk by chunk, as it
arrives from a microphone, and decodes greedily or with a beam search and
a language model. The model is a PyTorch model or one exported to ONNX,
which runs without PyTorch.
"""

import numpy as np

from speech.features import SpecgramStream
from speech.models.ctc_decoder import BEAM_SIZE, PRUNE, BeamSearch
from speech.models.word_lm import LM_WEIGHT, UNK_PENALTY, WORD_BONUS, WordLM


def load(path, tag="best", device="cpu", threads=None):
    """
    Loads a streaming CTC model: a .onnx file from speech.export, run
    with ONNX Runtime, or a directory saved by train.py, run with
    PyTorch.

    Arguments:
        path (str): The .onnx file or model directory.
        tag (str): For a directory, "best" for the model with the lowest
            dev CER, or None for the last one.
        device (str): For a directory, the device to run the model on.
        threads (int, optional): The CPU threads to compute with.

    Returns an OnnxModel or TorchModel.
    """
    if path.endswith(".onnx"):
        from speech.onnx_model import OnnxModel

        return OnnxModel(path, threads)

    import torch

    import speech
    from speech.models import CTC
    from speech.torch_model import TorchModel

    if threads is not None:
        torch.set_num_threads(threads)
    model, preproc = speech.load(path, tag=tag)
    if not isinstance(model, CTC) or model.rnn.bidirectional:
        raise ValueError(f"{path} is not a unidirectional CTC model, which can stream")
    model.to(device)
    model.set_eval()
    return TorchModel(model, preproc)


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
    def __init__(self, model, search=None):
        """
        Transcribes a stream of audio chunk by chunk with an OnnxModel or
        TorchModel, see load. Decodes greedily, or with the given
        ctc_decoder.BeamSearch, which can include a language model.
        """
        self.model = model
        self.search = search
        self.specgram = SpecgramStream(model.sample_rate)
        self.state = None
        self.greedy = []
        self.prev = model.blank
        # The number of blank frames since the last label.
        self.blanks = 0
        self.frame_seconds = model.frame_seconds
        # The output probabilities of the transcript's frames.
        self.frames = []

    def push(self, audio, final=False):
        """
        Adds the next chunk of audio samples. Pass final with the last
        chunk to flush the frames held back for the lookahead.
        """
        frames = self.specgram.push(audio)
        probs, self.state = self.model.stream(frames, self.state, final)
        self.frames.append(probs)

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
        return self.model.decode(self.labels(final))

    def probs(self):
        """
        The output probabilities of the transcript's frames, with shape
        (frames, labels + 1).
        """
        if not self.frames:
            return np.zeros((0, self.model.blank + 1), np.float32)
        return np.concatenate(self.frames)

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
        self.frames = []
        if self.search is not None:
            self.search.reset()
