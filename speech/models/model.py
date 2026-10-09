import collections
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import rnn as rnn_utils

# Carries an encoder stream across calls to Model.encode_stream.
StreamState = collections.namedtuple("StreamState", ["conv", "rnn", "lookahead"])


class Model(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        encoder_cfg = config["encoder"]
        conv_cfg = encoder_cfg["conv"]
        # Batch normalization and a clipped ReLU as in Deep Speech 2 keep
        # the activations of deep models from growing until the RNN
        # saturates and stops learning.
        batch_norm = encoder_cfg.get("batch_norm", False)
        relu_clip = encoder_cfg.get("relu_clip")

        convs = []
        in_c = 1
        for out_c, h, w, s in conv_cfg:
            # Batch normalization removes the mean, which cancels a bias.
            conv = nn.Conv2d(
                in_c, out_c, (h, w), stride=(s, s), padding=0, bias=not batch_norm
            )
            convs.append(conv)
            if batch_norm:
                convs.append(ConvBatchNorm(out_c))
            convs.append(nn.Hardtanh(0, relu_clip) if relu_clip else nn.ReLU())
            if config["dropout"] != 0:
                convs.append(nn.Dropout(p=config["dropout"]))
            in_c = out_c

        self.conv = nn.Sequential(*convs)
        conv_out = out_c * self.conv_out_size(input_dim, 1)
        assert conv_out > 0, "Convolutional ouptut frequency dimension is negative."

        rnn_cfg = encoder_cfg["rnn"]
        self.rnn = (BatchNormGRU if batch_norm else nn.GRU)(
            input_size=conv_out,
            hidden_size=rnn_cfg["dim"],
            num_layers=rnn_cfg["layers"],
            batch_first=True,
            dropout=config["dropout"],
            bidirectional=rnn_cfg["bidirectional"],
        )
        self._encoder_dim = rnn_cfg["dim"]

        # A lookahead convolution gives a unidirectional model a fixed
        # amount of future context, keeping it streamable.
        lookahead = encoder_cfg.get("lookahead", 0)
        assert not (lookahead and rnn_cfg["bidirectional"]), (
            "A lookahead layer requires a unidirectional RNN."
        )
        self.lookahead = None
        if lookahead:
            self.lookahead = Lookahead(self._encoder_dim, lookahead)

    def conv_out_size(self, n, dim):
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                # assuming a valid convolution
                k = c.kernel_size[dim]
                s = c.stride[dim]
                n = (n - k + 1) / s
                n = math.ceil(n)
        return n

    def forward(self, batch):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def encoded_lengths(self, lengths):
        """
        Maps the number of input frames of each example to its number
        of encoded frames.
        """
        return torch.tensor([self.conv_out_size(int(n), 0) for n in lengths])

    def encode(self, x, lengths=None):
        """
        Arguments:
            x (FloatTensor): Input with shape (batch, time, freq).
            lengths (optional): The number of input frames in each
                example. When given, the padding past each example does
                not affect its encoding, and the encoded frames past its
                end are zero. Without lengths every example is assumed
                to fill the batch.

        Returns the encoding with shape (batch, time, encoder_dim).
        """
        x = self.conv_forward(x.unsqueeze(1), lengths)
        x = self.flatten_conv(x)
        # Autocast can leave the convolutions in a lower precision than the
        # RNN weights, and on some devices, such as MPS, it does not cast
        # the RNN to match.
        x = x.to(self.rnn.weight_ih_l0.dtype)

        if lengths is None:
            x, _ = self.rnn(x)
        else:
            # The convolutions are valid over time, so only the RNN can
            # carry padding into an example's encoded frames.
            enc_lens = self.encoded_lengths(lengths)
            assert (enc_lens > 0).all(), "An input is too short to encode."
            total_length = x.size(1)
            x = rnn_utils.pack_padded_sequence(
                x, enc_lens, batch_first=True, enforce_sorted=False
            )
            x, _ = self.rnn(x)
            x, _ = rnn_utils.pad_packed_sequence(
                x, batch_first=True, total_length=total_length
            )

        if self.rnn.bidirectional:
            half = x.size()[-1] // 2
            x = x[:, :, :half] + x[:, :, half:]

        if self.lookahead is not None:
            x = self.lookahead(x)

        return x

    def encode_stream(self, x, state=None, final=False):
        """
        Encodes the next chunk of an input stream. Concatenating the
        outputs over a whole stream gives the same result as calling
        encode on the full input. Only unidirectional models can stream.

        Arguments:
            x (FloatTensor): The next chunk of input frames with shape
                (batch, time, freq). The chunk may be any length,
                including zero.
            state (StreamState): The state returned by the previous
                call, or None at the start of a stream.
            final (bool): Marks the end of the stream and flushes the
                frames held back for the lookahead.

        Returns the newly available encoded frames with shape
        (batch, time, encoder_dim) and the state for the next call.
        """
        assert not self.rnn.bidirectional, (
            "A bidirectional model cannot encode a stream."
        )
        if state is None:
            state = StreamState([None] * len(self.conv), None, None)

        # Each convolution holds back the input frames that are not yet
        # enough for its next output frame.
        conv_state = list(state.conv)
        empty = x.new_zeros((x.size(0), 0, self.encoder_dim))
        x = x.unsqueeze(1)
        for i, layer in enumerate(self.conv):
            if isinstance(layer, nn.Conv2d):
                if conv_state[i] is not None:
                    x = torch.cat([conv_state[i], x], dim=2)
                k = layer.kernel_size[0]
                s = layer.stride[0]
                n = max((x.size(2) - k) // s + 1, 0)
                conv_state[i] = x[:, :, n * s :]
                if n == 0:
                    x = None
                    break
                x = layer(x[:, :, : (n - 1) * s + k])
            else:
                x = layer(x)

        # The RNN rejects empty sequences, so skip it when the
        # convolutions have no new frames.
        rnn_state = state.rnn
        if x is None:
            x = empty
        else:
            x, rnn_state = self.rnn(self.flatten_conv(x), rnn_state)

        lookahead_state = None
        if self.lookahead is not None:
            x, lookahead_state = self.lookahead.stream(x, state.lookahead, final)

        return x, StreamState(conv_state, rnn_state, lookahead_state)

    def conv_forward(self, x, lengths=None):
        """
        Applies the convolutional front end to input with shape (batch,
        1, time, freq). Given the number of input frames of each
        example, its batch normalization skips the padding past them.
        """
        for layer in self.conv:
            if isinstance(layer, ConvBatchNorm):
                x = layer(x, lengths)
            else:
                x = layer(x)
            if isinstance(layer, nn.Conv2d) and lengths is not None:
                k, s = layer.kernel_size[0], layer.stride[0]
                lengths = [math.ceil((int(n) - k + 1) / s) for n in lengths]
        return x

    @staticmethod
    def flatten_conv(x):
        """
        Reshapes the convolution output from (batch, channels, time,
        freq) to (batch, time, channels * freq) for the RNN.
        """
        x = torch.transpose(x, 1, 2).contiguous()
        b, t, c, f = x.size()
        return x.view((b, t, c * f))

    def loss(self, x, y):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def set_eval(self):
        """
        Set the model to evaluation mode.
        """
        self.eval()

    def set_train(self):
        """
        Set the model to training mode.
        """
        self.train()

    def infer(self, x):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def encoder_dim(self):
        return self._encoder_dim


class ConvBatchNorm(nn.BatchNorm2d):
    """
    Batch normalization of convolution outputs with shape (batch,
    channels, time, freq), per channel over all time steps and
    frequencies, as in Deep Speech 2. It computes in float32, also under
    autocast. At inference it uses the running averages from training,
    a fixed scale and shift per channel, so streaming is unaffected.
    """

    def forward(self, x, lengths=None):
        """
        Arguments:
            lengths (optional): The number of valid frames of each
                example. In training, the statistics then skip the
                padding past them.
        """
        if not self.training or lengths is None:
            return super().forward(x.float()).to(x.dtype)

        valid = torch.arange(x.size(2), device=x.device) < torch.as_tensor(
            lengths, device=x.device
        ).unsqueeze(1)
        mask = valid[:, None, :, None].float()
        count = mask.sum() * x.size(3)
        xf = x.float()
        mean = (xf * mask).sum(dim=(0, 2, 3)) / count
        var = ((xf - mean[:, None, None]) ** 2 * mask).sum(dim=(0, 2, 3)) / count
        with torch.no_grad():
            self.num_batches_tracked += 1
            # Without a momentum, as in nn.BatchNorm2d, the running
            # averages are plain averages over all batches.
            m = self.momentum
            if m is None:
                m = 1 / self.num_batches_tracked.item()
            self.running_mean.mul_(1 - m).add_(mean, alpha=m)
            # The running variance is unbiased, as in nn.BatchNorm2d.
            self.running_var.mul_(1 - m).add_(var * count / (count - 1), alpha=m)
        out = (xf - mean[:, None, None]) / torch.sqrt(var[:, None, None] + self.eps)
        out = out * self.weight[:, None, None] + self.bias[:, None, None]
        return out.to(x.dtype)


class BatchNormGRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_first,
        dropout,
        bidirectional,
    ):
        """
        A stack of GRU layers with sequence-wise batch normalization of
        each layer's input, adapted from Deep Speech 2. Its statistics
        are over all frames of all sequences in the batch. A packed
        sequence holds only the frames within each sequence's length, so
        padding never enters them. Deep Speech 2 normalizes the input
        projection W·x inside the recurrence instead, which PyTorch's
        fused GRU does not allow. At inference the running averages from
        training normalize each frame on its own, so streaming is
        unaffected. Takes the arguments of nn.GRU and, like it, returns
        the output and the final state of every layer.
        """
        super().__init__()
        assert batch_first, "BatchNormGRU only supports batch_first."
        self.bidirectional = bidirectional
        out_size = hidden_size * (2 if bidirectional else 1)
        sizes = [input_size] + [out_size] * (num_layers - 1)
        self.norms = nn.ModuleList(nn.BatchNorm1d(n) for n in sizes)
        self.layers = nn.ModuleList(
            nn.GRU(n, hidden_size, batch_first=True, bidirectional=bidirectional)
            for n in sizes
        )
        # As in nn.GRU, on the output of every layer but the last.
        self.dropout = nn.Dropout(dropout)

    @property
    def weight_ih_l0(self):
        return self.layers[0].weight_ih_l0

    def forward(self, x, state=None):
        directions = 2 if self.bidirectional else 1
        states = [None] * len(self.layers) if state is None else state.split(directions)
        finals = []
        for i, (norm, gru) in enumerate(zip(self.norms, self.layers)):
            if i > 0:
                x = self.frames(x, self.dropout)
            x, h = gru(self.frames(x, norm), states[i])
            finals.append(h)
        return x, torch.cat(finals)

    @staticmethod
    def frames(x, fn):
        """
        Applies fn to the frames of a packed sequence or of a tensor
        with shape (batch, time, features).
        """
        if isinstance(x, rnn_utils.PackedSequence):
            return x._replace(data=fn(x.data))
        b, t, f = x.size()
        return fn(x.reshape(b * t, f)).view(b, t, f)


class Lookahead(nn.Module):
    def __init__(self, dim, context):
        """
        The lookahead (row) convolution from Deep Speech 2. Each output
        frame is a per-feature weighted sum of the input frame and the
        next `context` frames, so a unidirectional model sees a fixed
        amount of future input. The context is in encoder frames, after
        any striding in the convolutional front end.
        """
        super().__init__()
        self.context = context
        self.conv = nn.Conv1d(dim, dim, kernel_size=context + 1, groups=dim, bias=False)

        # Start as the identity, so the layer passes the RNN output
        # through and learns how much future context to mix in. A random
        # start scrambles the RNN output over time and slows training.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, 0] = 1.0

    def forward(self, x, pad=True):
        """
        Arguments:
            x (FloatTensor): Input with shape (batch, time, dim).
            pad (bool): Zero pad the end of the input so every frame
                has an output. Without padding the last `context`
                frames have no output.
        """
        x = torch.transpose(x, 1, 2)
        if pad:
            x = F.pad(x, (0, self.context))
        x = self.conv(x)
        return torch.transpose(x, 1, 2)

    def stream(self, x, state, final):
        """
        Applies the lookahead to the next chunk of a stream, holding
        back the last `context` frames until their future arrives.
        Returns the output and the held back frames for the next call.
        """
        if state is not None:
            x = torch.cat([state, x], dim=1)
        if final:
            return (self(x) if x.size(1) > 0 else x), None
        n = x.size(1) - self.context
        if n <= 0:
            return x[:, :0], x
        return self(x, pad=False), x[:, n:]


def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t, inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, : inp.shape[0], :] = inp
    return input_mat
