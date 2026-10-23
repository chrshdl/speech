"""
The depthwise separable CNN (DS-CNN) of "Hello Edge: Keyword Spotting on
Microcontrollers" (Zhang et al., 2017), which classifies a second of
MFCCs as one of a few labels, such as silence, other speech and a wake
word, and its export to ONNX for speech.wake_word.

A convolution over time and frequency is followed by depthwise
separable convolutions, each a 3x3 convolution of every channel on its
own and a 1x1 convolution that mixes the channels, all with batch norm
and ReLU, then the average over time and frequency and a linear layer.
"""

import json

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from speech import wake_word
from speech.export import IR_VERSION, OPSET

# The sizes of the paper, each the channels, the first convolution's
# stride over time and frequency, and the strides of the separable
# convolutions.
SIZES = {
    "S": (64, (2, 2), [(1, 1)] * 4),
    "M": (172, (2, 1), [(2, 2)] + [(1, 1)] * 3),
    "L": (276, (2, 1), [(2, 2)] + [(1, 1)] * 4),
}
# The first convolution's kernel over time and frequency.
KERNEL = (10, 4)


class DSCNN(nn.Module):
    def __init__(self, labels, keyword, size="S", dropout=0.0):
        """
        Arguments:
            labels (list): The names of the outputs, such as
                ["silence", "unknown", "olivia"].
            keyword (str): The label of the wake word.
            size (str): "S", "M" or "L", see SIZES.
            dropout (float): The dropout before the linear layer.
        """
        super().__init__()
        self.config = {
            "labels": labels,
            "keyword": keyword,
            "size": size,
            "dropout": dropout,
        }
        channels, stride, strides = SIZES[size]
        # Padded as TensorFlow's "SAME", as in the paper.
        self.pad = nn.ZeroPad2d((1, 2, 4, 5))
        self.layers = nn.ModuleList([self._block(1, channels, KERNEL, stride)])
        for stride in strides:
            self.layers.append(
                self._block(channels, channels, (3, 3), stride, groups=channels)
            )
            self.layers.append(self._block(channels, channels, (1, 1), (1, 1)))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, len(labels))
        # The features' mean and standard deviation, set from the
        # training data with set_normalization.
        self.register_buffer("mean", torch.zeros(wake_word.COEFFICIENTS))
        self.register_buffer("std", torch.ones(wake_word.COEFFICIENTS))

    @staticmethod
    def _block(inputs, outputs, kernel, stride, groups=1):
        padding = tuple(k // 2 for k in kernel) if kernel != KERNEL else 0
        return nn.Sequential(
            nn.Conv2d(
                inputs, outputs, kernel, stride, padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
        )

    def set_normalization(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean))
        self.std.copy_(torch.as_tensor(std))

    def forward(self, mfcc):
        """
        The logits of the labels for MFCCs with shape (batch, frames,
        coefficients).
        """
        x = ((mfcc - self.mean) / self.std)[:, None]
        x = self.pad(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.dropout(x.mean(dim=(2, 3))))


def save(model, path):
    torch.save(
        {"config": model.config, "state_dict": model.state_dict()},
        path,
    )


def load(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = DSCNN(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build(model, sample_rate):
    """
    The ONNX graph of a trained DSCNN, which takes MFCCs "mfcc" with shape
    (batch, frames, coefficients) and gives the probabilities of the
    labels "probs", with shape (batch, labels). The batch norms are folded
    into the convolutions.
    """
    model = model.eval()
    initializers = []
    nodes = []

    def const(name, array):
        initializers.append(
            numpy_helper.from_array(np.asarray(array, np.float32), name)
        )
        return name

    def node(op, inputs, name, **attributes):
        nodes.append(helper.make_node(op, inputs, [name], name=name, **attributes))
        return name

    x = node("Sub", ["mfcc", const("mean", model.mean.numpy())], "centered")
    x = node("Div", [x, const("std", model.std.numpy())], "normalized")
    x = node("Unsqueeze", [x, _int64("channel", [1], initializers)], "image")
    pad = model.pad.padding  # left, right, top, bottom
    for i, layer in enumerate(model.layers):
        conv, norm = layer[0], layer[1]
        scale = (norm.weight / torch.sqrt(norm.running_var + norm.eps)).detach()
        weight = conv.weight.detach() * scale[:, None, None, None]
        bias = norm.bias.detach() - norm.running_mean * scale
        if i == 0:
            pads = [pad[2], pad[0], pad[3], pad[1]]
        else:
            pads = [conv.padding[0], conv.padding[1]] * 2
        x = node(
            "Conv",
            [x, const(f"w{i}", weight.numpy()), const(f"b{i}", bias.numpy())],
            f"conv{i}",
            kernel_shape=list(conv.kernel_size),
            strides=list(conv.stride),
            pads=pads,
            group=conv.groups,
        )
        x = node("Relu", [x], f"relu{i}")
    x = node("GlobalAveragePool", [x], "pooled")
    x = node("Flatten", [x], "flat")
    weight = model.fc.weight.detach().numpy().T
    x = node(
        "Gemm",
        [x, const("fc_w", weight), const("fc_b", model.fc.bias.detach().numpy())],
        "logits",
    )
    node("Softmax", [x], "probs", axis=1)

    labels = model.config["labels"]
    frames = wake_word.Mfcc(sample_rate).frames(
        round(wake_word.CLIP_SECONDS * sample_rate)
    )
    graph = helper.make_graph(
        nodes,
        "ds_cnn",
        [
            helper.make_tensor_value_info(
                "mfcc", TensorProto.FLOAT, ["batch", frames, wake_word.COEFFICIENTS]
            )
        ],
        [
            helper.make_tensor_value_info(
                "probs", TensorProto.FLOAT, ["batch", len(labels)]
            )
        ],
        initializers,
    )
    onnx_model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="speech"
    )
    onnx_model.ir_version = IR_VERSION
    metadata = {
        "format_version": wake_word.FORMAT_VERSION,
        "labels": json.dumps(labels),
        "keyword": model.config["keyword"],
        "sample_rate": sample_rate,
    }
    helper.set_model_props(onnx_model, {k: str(v) for k, v in metadata.items()})
    onnx.checker.check_model(onnx_model)
    return onnx_model


def _int64(name, values, initializers):
    initializers.append(numpy_helper.from_array(np.asarray(values, np.int64), name))
    return name


def export(model, sample_rate, path):
    """Writes the ONNX graph of a model, see build."""
    onnx.save(build(model, sample_rate), path)
