import math
import random

import numpy as np
import torch
from torch import nn

from . import model


class Seq2Seq(model.Model):
    def __init__(self, freq_dim, vocab_size, config):
        super().__init__(freq_dim, config)

        # For decoding
        decoder_cfg = config["decoder"]
        rnn_dim = self.encoder_dim
        embed_dim = decoder_cfg["embedding_dim"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_rnn = nn.GRUCell(input_size=embed_dim, hidden_size=rnn_dim)

        self.attend = NNAttention(rnn_dim, log_t=decoder_cfg.get("log_t", False))

        self.sample_prob = decoder_cfg.get("sample_prob", 0)
        self.scheduled_sampling = self.sample_prob != 0

        # *NB* we predict vocab_size - 1 classes since we
        # never need to predict the start of sequence token.
        self.fc = nn.Linear(rnn_dim, vocab_size - 1)

    def set_eval(self):
        """
        Set the model to evaluation mode.
        """
        self.eval()
        self.scheduled_sampling = False

    def set_train(self):
        """
        Set the model to training mode.
        """
        self.train()
        self.scheduled_sampling = self.sample_prob != 0

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        x = x.to(self.device)
        y = y.to(self.device)
        out, _ = self.forward_impl(x, y, x_lens)
        batch_size, _, out_dim = out.size()

        # Ignore the end padding past each label.
        targets = y[:, 1:].clone()
        steps = torch.arange(targets.size(1), device=y.device)
        targets[steps >= (y_lens.to(y.device) - 1).unsqueeze(1)] = -100
        loss = nn.functional.cross_entropy(
            out.reshape(-1, out_dim).float(),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return loss / batch_size

    def forward_impl(self, x, y, x_lens=None):
        x = self.encode(x, x_lens)
        mask = self.attention_mask(x, x_lens)
        out, alis = self.decode(x, y, mask)
        return out, alis

    def forward(self, batch):
        x, y, x_lens, _ = self.collate(*batch)
        x = x.to(self.device)
        y = y.to(self.device)
        return self.forward_impl(x, y, x_lens)[0]

    def attention_mask(self, x, x_lens):
        """
        Returns a (batch, time) mask of the encoded frames in each
        example, or None if there are no lengths.
        """
        if x_lens is None:
            return None
        lens = self.encoded_lengths(x_lens).to(x.device)
        return torch.arange(x.size(1), device=x.device) < lens.unsqueeze(1)

    def decode(self, x, y, mask=None):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        mask (optional) marks the frames of x to attend to
        """

        inputs = self.embedding(y[:, :-1])

        out = []
        aligns = []

        # Under autocast the RNN cell needs inputs in its own precision on
        # devices that do not cast it, such as MPS.
        hx = x.new_zeros((x.shape[0], x.shape[2]), dtype=self.dec_rnn.weight_hh.dtype)
        ax = None
        sx = None
        for t in range(y.size()[1] - 1):
            sample = out and self.scheduled_sampling
            if sample and random.random() < self.sample_prob:
                ix = torch.max(out[-1], dim=2)[1]
                ix = self.embedding(ix)
            else:
                ix = inputs[:, t : t + 1, :]

            if sx is not None:
                ix = ix + sx

            hx = self.dec_rnn(ix.squeeze(dim=1).to(hx.dtype), hx)
            ox = hx.unsqueeze(dim=1)

            sx, ax = self.attend(x, ox, ax, mask)
            aligns.append(ax)
            out.append(self.fc(ox + sx))

        out = torch.cat(out, dim=1)
        aligns = torch.stack(aligns, dim=1)
        return out, aligns

    def decode_step(self, x, y, state=None, softmax=False, mask=None):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        mask (optional) marks the frames of x to attend to
        """
        if state is None:
            hx = x.new_zeros(
                (x.shape[0], x.shape[2]), dtype=self.dec_rnn.weight_hh.dtype
            )
            ax = None
            sx = None
        else:
            hx, ax, sx = state

        ix = self.embedding(y)
        if sx is not None:
            ix = ix + sx
        hx = self.dec_rnn(ix.squeeze(dim=1).to(hx.dtype), hx=hx)
        ox = hx.unsqueeze(dim=1)
        sx, ax = self.attend(x, ox, ax=ax, mask=mask)
        out = ox + sx
        out = self.fc(out.squeeze(dim=1))
        if softmax:
            out = nn.functional.log_softmax(out.float(), dim=1)
        return out, (hx, ax, sx)

    def predict(self, batch):
        probs = self(batch)
        argmaxs = torch.max(probs, dim=2)[1]
        return argmaxs.tolist()

    def infer_decode(self, x, y, end_tok, max_len, mask=None):
        probs = []
        argmaxs = [y]
        state = None
        for e in range(max_len):
            out, state = self.decode_step(x, y, state=state, mask=mask)
            probs.append(out)
            y = torch.max(out, dim=1)[1]
            y = y.unsqueeze(dim=1)
            argmaxs.append(y)
            if (y == end_tok).all():
                break

        probs = torch.cat(probs)
        argmaxs = torch.cat(argmaxs, dim=1)
        return probs, argmaxs

    @torch.no_grad()
    def infer(self, batch, max_len=200):
        """
        Infer a likely output. No beam search yet.
        """
        x, y, x_lens, _ = self.collate(*batch)
        end_tok = y[0, -1].item()  # TODO
        x = self.encode(x.to(self.device), x_lens)
        mask = self.attention_mask(x, x_lens)

        # needs to be the start token, TODO
        y = y[:, 0:1].to(self.device)
        _, argmaxs = self.infer_decode(x, y, end_tok, max_len, mask)
        return argmaxs.tolist()

    @torch.no_grad()
    def beam_search(self, batch, beam_size=10, max_len=200):
        x, y, x_lens, _ = self.collate(*batch)
        start_tok = y[0, 0].item()
        end_tok = y[0, -1].item()  # TODO
        x = x.to(self.device)
        y = y.to(self.device)
        x = self.encode(x, x_lens)
        mask = self.attention_mask(x, x_lens)

        y = y[:, 0:1].clone()

        beam = [((start_tok,), 0, None)]
        complete = []
        for _ in range(max_len):
            new_beam = []
            for hyp, score, state in beam:
                y[0] = hyp[-1]
                out, state = self.decode_step(
                    x, y, state=state, softmax=True, mask=mask
                )
                out = out.squeeze(dim=0).tolist()
                for i, p in enumerate(out):
                    new_score = score + p
                    new_hyp = hyp + (i,)
                    new_beam.append((new_hyp, new_score, state))
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)

            # Remove complete hypotheses
            for cand in new_beam[:beam_size]:
                if cand[0][-1] == end_tok:
                    complete.append(cand)

            beam = [c for c in new_beam if c[0][-1] != end_tok][:beam_size]

            if len(beam) == 0:
                break

            # Stopping criteria:
            # complete contains beam_size more probable
            # candidates than anything left in the beam
            if sum(c[1] > beam[0][1] for c in complete) >= beam_size:
                break

        complete = sorted(complete, key=lambda x: x[1], reverse=True)
        if len(complete) == 0:
            complete = beam
        hyp, score, _ = complete[0]
        return [hyp]

    def collate(self, inputs, labels):
        """
        Returns the padded inputs, the end padded labels, the number of
        input frames in each example and the length of each label.
        """
        x_lens = torch.tensor([i.shape[0] for i in inputs])
        y_lens = torch.tensor([len(l) for l in labels])
        inputs = torch.from_numpy(model.zero_pad_concat(inputs))
        labels = torch.from_numpy(end_pad_concat(labels))
        return inputs, labels, x_lens, y_lens


def end_pad_concat(labels):
    # Assumes last item in each example is the end token.
    batch_size = len(labels)
    end_tok = labels[0][-1]
    max_len = max(len(l) for l in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=end_tok, dtype=np.int64)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    return cat_labels


def attention_softmax(pax, mask=None, log_t=False):
    """
    Normalizes attention scores with shape (batch size, time) over the
    time steps the mask keeps. With log_t the scores are scaled by the
    log of the number of kept time steps.
    """
    if log_t:
        if mask is None:
            pax = math.log(pax.size(1)) * pax
        else:
            pax = torch.log(mask.sum(dim=1, keepdim=True).to(pax.dtype)) * pax
    if mask is not None:
        pax = pax.masked_fill(~mask, float("-inf"))
    return nn.functional.softmax(pax, dim=1)


class Attention(nn.Module):
    def __init__(self, kernel_size=11, log_t=False):
        """
        Module which Performs a single attention step along the
        second axis of a given encoded input. The module uses
        both 'content' and 'location' based attention.

        The 'content' based attention is an inner product of the
        decoder hidden state with each time-step of the encoder
        state.

        The 'location' based attention performs a 1D convollution
        on the previous attention vector and adds this into the
        next attention vector prior to normalization.

        *NB* Should compute attention differently if using cuda or cpu
        based on performance. See
        https://gist.github.com/awni/9989dd31642d42405903dec8ab91d1f0
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding)
        self.log_t = log_t

    def forward(self, eh, dhx, ax=None, mask=None):
        """
        Arguments:
            eh (FloatTensor): the encoder hidden state with
                shape (batch size, time, hidden dimension).
            dhx (FloatTensor): one time step of the decoder hidden
                state with shape (batch size, hidden dimension).
                The hidden dimension must match that of the
                encoder state.
            ax (FloatTensor): one time step of the attention
                vector.
            mask (BoolTensor): marks the time steps of eh to attend
                to, with shape (batch size, time).

        Returns the summary of the encoded hidden state
        and the corresponding alignment.
        """
        # Compute inner product of decoder slice with every
        # encoder slice.
        # location attention
        pax = eh * dhx
        pax = torch.sum(pax, dim=2)

        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).squeeze(dim=1)
            pax = pax + ax

        ax = attention_softmax(pax, mask, self.log_t)

        # At this point sx should have size (batch size, time).
        # Reduce the encoder state accross time weighting each
        # slice by its corresponding value in sx.
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax


class ProdAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, eh, dhx, ax=None, mask=None):
        pax = eh * dhx
        pax = torch.sum(pax, dim=2)

        ax = attention_softmax(pax, mask)

        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax


class NNAttention(nn.Module):
    def __init__(self, n_channels, kernel_size=15, log_t=False):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, n_channels, kernel_size, padding=padding)
        self.nn = nn.Sequential(nn.ReLU(), nn.Linear(n_channels, 1))
        self.log_t = log_t

    def forward(self, eh, dhx, ax=None, mask=None):
        pax = eh + dhx
        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).transpose(1, 2)
            pax = pax + ax

        pax = self.nn(pax)
        pax = pax.squeeze(dim=2)
        ax = attention_softmax(pax, mask, self.log_t)

        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax
