"""
Exports a streaming CTC model to ONNX for speech.onnx_model, which runs
it with ONNX Runtime and without PyTorch, such as on a Raspberry Pi.

The graph encodes a window of feature frames that yields one or more
encoder frames, carrying the GRU state from call to call:

    features (1, time, freq), hidden (layers, 1, dim)
        -> terms (1, frames, lookahead + 1, labels), hidden_out

At inference batch normalization is a fixed scale and shift, which is
folded into the weights before it. The lookahead convolution and the
output layer are linear, so the logits of frame t are

    fc(sum_k w_k * y[t + k]) = b + sum_k W_k y[t + k]

with W_k = fc.weight * w_k. The graph returns the terms W_k y[s] of each
new frame s, with the bias in k = 0, and the runtime sums them as the
future frames arrive. Their sum over the frames that exist at the end of
a stream is the zero padding of the lookahead.

With int8 the GRU weights, nearly all of the model and of the memory read
for each frame, are stored as 8 bit integers with a scale per output,
and their products use ONNX Runtime's integer kernels. The inputs of the
products are quantized as the graph runs, each frame with its own scale,
so how a stream is split into chunks hardly changes its output. On the
LibriSpeech dev set int8 left the CER and WER of the 100 hour model
unchanged.
"""

import functools
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from speech.features import sample_rate_for
from speech.models import CTC
from speech.models.model import BatchNormGRU, ConvBatchNorm
from speech.onnx_model import FORMAT_VERSION

OPSET = 18
IR_VERSION = 10


def _as_numpy(t):
    return t.detach().cpu().double().numpy()


def _batch_norm_affine(bn):
    """The scale and shift of a batch normalization at inference."""
    scale = _as_numpy(bn.weight) / np.sqrt(_as_numpy(bn.running_var) + bn.eps)
    return scale, _as_numpy(bn.bias) - _as_numpy(bn.running_mean) * scale


def conv_layers(model):
    """
    Returns the convolutions as (weight, bias, stride, clip) with batch
    normalization folded in, where clip is the ReLU's upper limit or None.
    """
    layers = []
    for layer in model.conv:
        if isinstance(layer, nn.Conv2d):
            weight = _as_numpy(layer.weight)
            bias = np.zeros(weight.shape[0])
            if layer.bias is not None:
                bias = _as_numpy(layer.bias)
            layers.append([weight, bias, layer.stride, None])
        elif isinstance(layer, ConvBatchNorm):
            scale, shift = _batch_norm_affine(layer)
            layers[-1][0] = layers[-1][0] * scale[:, None, None, None]
            layers[-1][1] = layers[-1][1] * scale + shift
        elif isinstance(layer, nn.Hardtanh):
            assert layer.min_val == 0, "Only a ReLU clipped at the top is supported."
            layers[-1][3] = layer.max_val
        else:
            assert isinstance(layer, (nn.ReLU, nn.Dropout)), f"Unexpected {layer}"
    return [tuple(l) for l in layers]


def gru_layers(model):
    """
    Returns each GRU layer's (weight_ih, bias_ih, weight_hh, bias_hh) in
    PyTorch's layout, with the batch normalization of its input folded
    into the input weights.
    """
    rnn = model.rnn
    layers = []
    if isinstance(rnn, BatchNormGRU):
        for norm, gru in zip(rnn.norms, rnn.layers):
            w_ih, b_ih, w_hh, b_hh = (
                _as_numpy(getattr(gru, f"{n}_l0"))
                for n in ("weight_ih", "bias_ih", "weight_hh", "bias_hh")
            )
            # W (x * scale + shift) + b = (W * scale) x + (W shift + b)
            scale, shift = _batch_norm_affine(norm)
            layers.append((w_ih * scale, b_ih + w_ih @ shift, w_hh, b_hh))
    else:
        for l in range(rnn.num_layers):
            layers.append(
                tuple(
                    _as_numpy(getattr(rnn, f"{n}_l{l}"))
                    for n in ("weight_ih", "bias_ih", "weight_hh", "bias_hh")
                )
            )
    return layers


def head_terms(model):
    """
    Returns the weight with shape (dim, (lookahead + 1) * labels) and the
    bias of the terms W_k y, see the module docstring.
    """
    fc_w, fc_b = _as_numpy(model.fc.weight), _as_numpy(model.fc.bias)
    if model.lookahead is None:
        taps = np.ones((fc_w.shape[1], 1))
    else:
        taps = _as_numpy(model.lookahead.conv.weight)[:, 0, :]
    weight = np.concatenate([(fc_w * taps[:, k]).T for k in range(taps.shape[1])], 1)
    bias = np.zeros((taps.shape[1], fc_w.shape[0]))
    bias[0] = fc_b
    return weight, bias.reshape(-1)


def receptive_field(model):
    """
    The total stride of the convolutions over time and the number of
    input frames that the first encoder frame depends on. Encoder frame j
    depends on the input frames from stride * j to stride * j + field.
    """
    stride, field = 1, 1
    for layer in model.conv:
        if isinstance(layer, nn.Conv2d):
            k, s = layer.kernel_size[0], layer.stride[0]
            field += (k - 1) * stride
            stride *= s
    return stride, field


class _Graph:
    """Collects the nodes and weights of a graph and its Scan bodies."""

    def __init__(self, int8):
        self.int8 = int8
        self.initializers = []
        self.names = set()

    def const(self, name, value, dtype=np.float32):
        """A weight or constant, which subgraphs can also use."""
        if name not in self.names:
            self.names.add(name)
            array = np.asarray(value, dtype=dtype)
            self.initializers.append(numpy_helper.from_array(array, name))
        return name

    @staticmethod
    def node(nodes, op, inputs, outputs, **attrs):
        """
        Appends a node to nodes and returns its output, or its outputs
        when given a list.
        """
        names = outputs if isinstance(outputs, list) else [outputs]
        nodes.append(helper.make_node(op, inputs, names, name=names[0], **attrs))
        return outputs

    def matmul(self, nodes, x, weight, bias, name, int8=None):
        """
        x @ weight + bias for x with shape (rows, in) and a weight with
        shape (in, out), with the weight in int8 when quantizing.

        Quantized, each row of x is quantized as the graph runs, with its
        own scale. For the input projections of a chunk of frames that
        makes each frame's result independent of the frames it shares the
        chunk with, so a stream gives nearly the same output however its
        audio is split into chunks.
        """
        int8 = self.int8 if int8 is None else int8
        node = functools.partial(self.node, nodes)
        if not int8:
            y = node("MatMul", [x, self.const(name + "_w", weight)], name + "_mm")
            return node("Add", [y, self.const(name + "_b", bias)], name)
        # Symmetric, with a scale per output, so outputs with large weights
        # do not cost the others their precision.
        w_scale = np.abs(weight).max(axis=0) / 127
        w_scale[w_scale == 0] = 1
        w_quantized = self.const(name + "_wq", np.round(weight / w_scale), np.int8)

        # Symmetric per row: x ~ x_scale * round(x / x_scale), with x_scale
        # the row's largest magnitude over 127.
        absmax = node(
            "ReduceMax",
            [node("Abs", [x], name + "_abs"), self.const("axis_1", [1], np.int64)],
            name + "_absmax",
            keepdims=1,
        )
        x_scale = node(
            "Max",
            [
                node("Div", [absmax, self.const("int8_max", 127)], name + "_x_scale0"),
                self.const("tiny", np.finfo(np.float32).tiny),
            ],
            name + "_x_scale",
        )
        x_rounded = node(
            "Round", [node("Div", [x, x_scale], name + "_x_div")], name + "_x_round"
        )
        x_quantized = node("Cast", [x_rounded], name + "_xq", to=TensorProto.INT8)
        product = node("MatMulInteger", [x_quantized, w_quantized], name + "_int")
        product = node("Cast", [product], name + "_float", to=TensorProto.FLOAT)
        y = node("Mul", [product, x_scale], name + "_x_scaled")
        y = node("Mul", [y, self.const(name + "_scale", w_scale)], name + "_scaled")
        return node("Add", [y, self.const(name + "_b", bias)], name)


def _gru_step(graph, l, w_hh, b_hh):
    """
    The graph of one step of GRU layer l for Scan, with PyTorch's
    equations

        r, z = sigmoid(x_rz + W_hr,hz h + b_hr,hz)
        n = tanh(x_n + r * (W_hn h + b_hn))
        h' = n + z * (h - n)

    where x holds the frame's input projections, which are computed for
    the whole chunk beforehand.
    """
    dim = w_hh.shape[1]
    p = f"gru{l}_"
    h, x, nodes = p + "h", p + "x", []
    node = functools.partial(graph.node, nodes)
    split_rz_n = graph.const("split_rz_n", [2 * dim, dim], np.int64)
    split_r_z = graph.const("split_r_z", [dim, dim], np.int64)

    gates = graph.matmul(nodes, h, w_hh.T, b_hh, p + "hh")
    x_rz, x_n = node("Split", [x, split_rz_n], [p + "x_rz", p + "x_n"], axis=0)
    h_rz, h_n = node("Split", [gates, split_rz_n], [p + "h_rz", p + "h_n"], axis=1)
    rz = node("Sigmoid", [node("Add", [x_rz, h_rz], p + "rz_in")], p + "rz")
    r, z = node("Split", [rz, split_r_z], [p + "r", p + "z"], axis=1)
    n_in = node("Add", [x_n, node("Mul", [r, h_n], p + "r_h_n")], p + "n_in")
    n = node("Tanh", [n_in], p + "n")
    z_diff = node("Mul", [z, node("Sub", [h, n], p + "h_minus_n")], p + "z_diff")
    h_next = node("Add", [n, z_diff], p + "h_next")
    y = node("Identity", [h_next], p + "y")

    def value(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    return helper.make_graph(
        nodes,
        p + "step",
        [value(h, [1, dim]), value(x, [3 * dim])],
        [value(h_next, [1, dim]), value(y, [1, dim])],
    )


def build(model, preproc, int8=True):
    """
    Returns the ONNX graph of a unidirectional CTC model, see the module
    docstring.
    """
    if not isinstance(model, CTC):
        raise TypeError("Only CTC models can be exported.")
    if model.rnn.bidirectional:
        raise ValueError("A bidirectional model cannot stream, so it is not exported.")

    graph = _Graph(int8)
    nodes = []
    node = functools.partial(graph.node, nodes)
    i64 = functools.partial(graph.const, dtype=np.int64)

    # Normalize the features and add a channel: (1, 1, time, freq).
    x = node("Sub", ["features", graph.const("mean", preproc.mean)], "centered")
    x = node("Mul", [x, graph.const("inv_std", 1 / preproc.std)], "normalized")
    x = node("Unsqueeze", [x, i64("axis_1", [1])], "conv_in")
    for i, (weight, bias, stride, clip) in enumerate(conv_layers(model)):
        x = node(
            "Conv",
            [x, graph.const(f"conv{i}_w", weight), graph.const(f"conv{i}_b", bias)],
            f"conv{i}",
            strides=list(stride),
        )
        if clip is None:
            x = node("Relu", [x], f"conv{i}_relu")
        else:
            limits = [graph.const("zero", 0), graph.const(f"conv{i}_clip", clip)]
            x = node("Clip", [x] + limits, f"conv{i}_relu")
    # (1, channels, time, freq) -> (time, channels * freq)
    x = node("Transpose", [x], "conv_time_first", perm=[0, 2, 1, 3])
    rnn_in = model.rnn.weight_ih_l0.shape[1]
    x = node("Reshape", [x, i64("conv_shape", [-1, rnn_in])], "rnn_in")

    layers = gru_layers(model)
    dim = model.encoder_dim
    hiddens = [f"hidden{l}" for l in range(len(layers))]
    node("Split", ["hidden"], hiddens, axis=0, num_outputs=len(layers))
    finals = []
    for l, (w_ih, b_ih, w_hh, b_hh) in enumerate(layers):
        projected = graph.matmul(nodes, x, w_ih.T, b_ih, f"gru{l}_ih")
        h0 = node("Squeeze", [hiddens[l], i64("axis_0", [0])], f"gru{l}_h0")
        h_n, ys = node(
            "Scan",
            [h0, projected],
            [f"gru{l}_hn", f"gru{l}_ys"],
            body=_gru_step(graph, l, w_hh, b_hh),
            num_scan_inputs=1,
        )
        finals.append(node("Unsqueeze", [h_n, "axis_0"], f"hidden_out{l}"))
        x = node("Reshape", [ys, i64("rnn_shape", [-1, dim])], f"gru{l}_out")
    node("Concat", finals, "hidden_out", axis=0)

    # The head stays in float32. It is small, and the output probabilities
    # decide the beam search.
    weight, bias = head_terms(model)
    terms = graph.matmul(nodes, x, weight, bias, "head", int8=False)
    taps = len(bias) // (model.blank + 1)
    node(
        "Reshape", [terms, i64("terms_shape", [1, -1, taps, model.blank + 1])], "terms"
    )

    def value(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    onnx_graph = helper.make_graph(
        nodes,
        "streaming_ctc",
        [
            value("features", [1, "time", preproc.input_dim]),
            value("hidden", [len(layers), 1, dim]),
        ],
        [
            value("terms", [1, "frames", taps, model.blank + 1]),
            value("hidden_out", [len(layers), 1, dim]),
        ],
        graph.initializers,
    )
    onnx_model = helper.make_model(
        onnx_graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="speech",
    )
    onnx_model.ir_version = IR_VERSION

    stride, field = receptive_field(model)
    labels = [preproc.int_to_char[i] for i in range(preproc.vocab_size)]
    metadata = {
        "format_version": FORMAT_VERSION,
        "labels": json.dumps(labels),
        "start_and_end": json.dumps(preproc.start_and_end),
        "blank": model.blank,
        "sample_rate": sample_rate_for(preproc.input_dim),
        "stride": stride,
        "receptive_field": field,
        "int8": json.dumps(int8),
    }
    helper.set_model_props(onnx_model, {k: str(v) for k, v in metadata.items()})
    onnx.checker.check_model(onnx_model)
    return onnx_model


def export(model, preproc, path, int8=True):
    """Writes the ONNX graph of a model, see build."""
    onnx.save(build(model, preproc, int8), path)
