import torch
from torch import nn
from torch.nn import functional as F

from . import model
from .ctc_decoder import decode


class CTC(model.Model):
    def __init__(self, freq_dim, output_dim, config):
        super().__init__(freq_dim, config)

        # include the blank token
        self.blank = output_dim
        self.fc = nn.Linear(self.encoder_dim, output_dim + 1)

    def forward(self, batch):
        x, _, x_lens, _ = self.collate(*batch)
        return self.forward_impl(x, x_lens)

    def forward_impl(self, x, x_lens=None, softmax=False):
        x = self.encode(x.to(self.device), x_lens)
        x = self.fc(x)
        if softmax:
            return F.softmax(x, dim=2)
        return x

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        out = self.forward_impl(x, x_lens)

        # ctc_loss expects log probabilities with shape (time, batch, classes).
        log_probs = F.log_softmax(out, dim=2).transpose(0, 1)
        loss = F.ctc_loss(
            log_probs,
            y.to(out.device),
            self.encoded_lengths(x_lens),
            y_lens,
            blank=self.blank,
            reduction="sum",
            zero_infinity=True,
        )
        # Average over the batch but not the label lengths, like seq2seq.
        return loss / out.size(0)

    def collate(self, inputs, labels):
        """
        Returns the padded inputs, the concatenated labels, the number
        of input frames in each example and the length of each label.
        """
        x_lens = torch.tensor([i.shape[0] for i in inputs], dtype=torch.int32)
        x = torch.from_numpy(model.zero_pad_concat(inputs))
        y_lens = torch.tensor([len(l) for l in labels], dtype=torch.int32)
        y = torch.tensor([l for label in labels for l in label], dtype=torch.int32)
        return [x, y, x_lens, y_lens]

    @torch.no_grad()
    def stream(self, x, state=None, final=False):
        """
        Returns the output probabilities for the frames that the next
        chunk of input frames makes available, and the state for the
        next call. See Model.encode_stream.
        """
        x, state = self.encode_stream(x.to(self.device), state, final)
        return F.softmax(self.fc(x), dim=2), state

    @torch.no_grad()
    def infer(self, batch):
        x, _, x_lens, _ = self.collate(*batch)
        probs = self.forward_impl(x, x_lens, softmax=True)
        probs = probs.cpu().numpy()
        lens = self.encoded_lengths(x_lens)
        return [
            decode(p[:n], beam_size=1, blank=self.blank)[0] for p, n in zip(probs, lens)
        ]

    @staticmethod
    def max_decode(pred, blank):
        prev = pred[0]
        seq = [prev] if prev != blank else []
        for p in pred[1:]:
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        return seq
