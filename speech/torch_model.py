"""
Gives a PyTorch CTC model the streaming interface of
speech.onnx_model.OnnxModel, so speech.streaming can run either.
"""

import math

import torch
from torch import nn

from speech.features import SpecgramStream, sample_rate_for


class TorchModel:
    def __init__(self, model, preproc):
        """A unidirectional CTC model and its preprocessor."""
        self.model = model
        self.preproc = preproc
        self.blank = model.blank
        self.char_to_int = preproc.char_to_int
        self.sample_rate = sample_rate_for(preproc.input_dim)
        # Each encoded frame covers the feature hop times the time stride
        # of the convolutions.
        stride = math.prod(c.stride[0] for c in model.conv if isinstance(c, nn.Conv2d))
        hop = SpecgramStream(self.sample_rate).hop
        self.frame_seconds = stride * hop / self.sample_rate

    def _inputs(self, features):
        return torch.from_numpy(self.preproc.normalize(features)).float().unsqueeze(0)

    def stream(self, features, state=None, final=False):
        """
        Returns the output probabilities for the frames that the next
        chunk of feature frames, not normalized, makes available, and the
        state for the next call. See Model.encode_stream.
        """
        probs, state = self.model.stream(self._inputs(features), state, final)
        return probs[0].cpu().numpy(), state

    @torch.no_grad()
    def probs(self, features):
        """The output probabilities for a whole input."""
        return (
            self.model.forward_impl(self._inputs(features), softmax=True)[0]
            .cpu()
            .numpy()
        )

    def decode(self, labels):
        """The text of a sequence of label indices."""
        return "".join(self.preproc.decode(labels))
