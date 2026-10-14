"""
Runs a streaming CTC model exported by speech.export with ONNX Runtime.
Needs only NumPy and ONNX Runtime, not PyTorch, so it suits small
devices such as a Raspberry Pi.
"""

import collections
import json

import numpy as np
import onnxruntime

from speech.features import SpecgramStream

# The version of the graph's inputs, outputs and metadata that speech.export
# writes, bumped when they change.
FORMAT_VERSION = 2

# Carries a stream across calls to OnnxModel.stream: the feature frames not
# yet encoded, the GRU state, and the encoder frames whose lookahead is not
# complete yet, see speech.export.
OnnxState = collections.namedtuple("OnnxState", ["frames", "hidden", "pending"])


def open_session(path, threads=None):
    """
    An ONNX Runtime session of a .onnx file, computing with the threads,
    by default one per CPU core.
    """
    options = onnxruntime.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
    # Waiting threads sleep instead of spinning. Each chunk is little work,
    # and spinning kept every core busy between chunks, which slowed the
    # beam search running alongside and heats a Raspberry Pi until it
    # throttles.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return onnxruntime.InferenceSession(
        path, options, providers=["CPUExecutionProvider"]
    )


class OnnxModel:
    def __init__(self, path, threads=None):
        """
        Loads a model exported by speech.export.

        Arguments:
            path (str): The .onnx file.
            threads (int, optional): The threads to compute with, by
                default one per CPU core.
        """
        self.session = open_session(path, threads)
        meta = self.session.get_modelmeta().custom_metadata_map
        if int(meta.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("The model is from another version of speech.export.")
        self.labels = json.loads(meta["labels"])
        self.char_to_int = {c: i for i, c in enumerate(self.labels)}
        self.start_and_end = json.loads(meta["start_and_end"])
        self.blank = int(meta["blank"])
        self.sample_rate = int(meta["sample_rate"])
        self.stride = int(meta["stride"])
        self.receptive_field = int(meta["receptive_field"])
        self.int8 = json.loads(meta["int8"])

        inputs = {i.name: i.shape for i in self.session.get_inputs()}
        self.input_dim = inputs["features"][2]
        self.hidden_shape = inputs["hidden"]
        self.output_dim = self.session.get_outputs()[0].shape[2]
        hop = SpecgramStream(self.sample_rate).hop
        self.frame_seconds = self.stride * hop / self.sample_rate

    def stream(self, features, state=None, final=False):
        """
        Returns the output probabilities for the frames that the next
        chunk of feature frames makes available, and the state for the
        next call. Concatenating the outputs over a stream gives the
        output for the whole input.

        Arguments:
            features (ndarray): The next log spectrogram frames, not
                normalized, with shape (time, freq). May be empty.
            state (OnnxState): The state returned by the previous call, or
                None at the start of a stream.
            final (bool): Marks the end of the stream and flushes the
                frames held back for the lookahead.

        Returns probabilities with shape (frames, labels + 1), the last
        being the blank, and the state.
        """
        if state is None:
            state = OnnxState(
                np.zeros((0, self.input_dim), np.float32),
                np.zeros(self.hidden_shape, np.float32),
                np.zeros((0, self.hidden_shape[2]), np.float32),
            )
        frames = np.concatenate([state.frames, features.astype(np.float32)])

        # Encoder frame j depends on the feature frames from stride * j to
        # stride * j + receptive_field, so encode all the frames that are
        # complete and keep the rest, which later frames share.
        n = 0
        if len(frames) >= self.receptive_field:
            n = (len(frames) - self.receptive_field) // self.stride + 1
        if n > 0:
            window = frames[: (n - 1) * self.stride + self.receptive_field]
        elif final:
            # The pending frames are flushed with a window of padding, whose
            # encoder frame the graph drops.
            window = np.zeros((self.receptive_field, self.input_dim), np.float32)
        else:
            empty = np.zeros((0, self.output_dim), np.float32)
            return empty, OnnxState(frames, state.hidden, state.pending)
        logits, hidden, pending = self.session.run(
            None,
            {
                "features": window[None],
                "hidden": state.hidden,
                "pending": state.pending,
                "final": np.array([final]),
                "encode": np.array([n > 0]),
            },
        )
        state = OnnxState(frames[n * self.stride :], hidden, pending)
        return softmax(logits[0]), state

    def probs(self, features):
        """The output probabilities for a whole input, see stream."""
        return self.stream(features, final=True)[0]

    def decode(self, labels):
        """The text of a sequence of label indices."""
        text = [self.labels[l] for l in labels]
        # As speech.loader.Preprocessor.decode.
        if self.start_and_end and text:
            start = int(text[0] == "<s>")
            end = text.index("</s>") if text[-1] == "</s>" else len(text)
            text = text[start:end]
        return "".join(text)


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
