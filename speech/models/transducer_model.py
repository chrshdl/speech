import torch
from torch import nn
from torchaudio.functional import rnnt_loss

from . import model


class Transducer(model.Model):
    def __init__(self, freq_dim, vocab_size, config):
        super().__init__(freq_dim, config)

        # For decoding
        decoder_cfg = config["decoder"]
        rnn_dim = self.encoder_dim
        embed_dim = decoder_cfg["embedding_dim"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=rnn_dim,
            num_layers=decoder_cfg["layers"],
            batch_first=True,
            dropout=config["dropout"],
        )

        # include the blank token
        self.blank = vocab_size
        self.fc1 = nn.Linear(rnn_dim, rnn_dim)
        self.fc2 = nn.Linear(rnn_dim, vocab_size + 1)

    def forward(self, batch):
        x, y, x_lens, _ = self.collate(*batch)
        return self.forward_impl(x, y, x_lens)

    def forward_impl(self, x, y, x_lens=None):
        """
        Returns the joint network scores for every pair of encoded frame
        and label prefix, with shape (batch, time, label length + 1,
        vocab size + 1).
        """
        x = self.encode(x.to(self.device), x_lens)
        y, _ = self.predict(y.to(self.device))
        return self.joint(x.unsqueeze(dim=2), y.unsqueeze(dim=1))

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        out = self.forward_impl(x, y, x_lens)
        loss = rnnt_loss(
            out,
            y.to(out.device, torch.int32),
            self.encoded_lengths(x_lens).to(out.device, torch.int32),
            y_lens.to(out.device),
            blank=self.blank,
            reduction="sum",
        )
        # Average over the batch but not the label lengths, like seq2seq.
        return loss / out.size(0)

    def predict(self, y, state=None):
        """
        Runs the prediction network on labels with shape (batch,
        length). Without a state the network starts from a zero input,
        which adds one step to the output.

        Returns the output with shape (batch, steps, encoder_dim) and
        the state to continue from.
        """
        y = self.embedding(y)
        if state is None:
            start = y.new_zeros((y.size(0), 1, y.size(2)))
            y = torch.cat([start, y], dim=1)
        return self.dec_rnn(y, state)

    def joint(self, x, y):
        """
        Combines encoder and prediction network outputs into scores
        over the vocabulary and the blank.
        """
        out = self.fc1(x) + self.fc1(y)
        out = nn.functional.relu(out)
        return self.fc2(out)

    def collate(self, inputs, labels):
        """
        Returns the padded inputs, the padded labels, the number of
        input frames in each example and the length of each label.
        """
        x_lens = torch.tensor([i.shape[0] for i in inputs], dtype=torch.int32)
        y_lens = torch.tensor([len(l) for l in labels], dtype=torch.int32)
        x = torch.from_numpy(model.zero_pad_concat(inputs))
        # The padding value is ignored since it is past each label.
        y = torch.zeros((len(labels), int(y_lens.max())), dtype=torch.long)
        for e, l in enumerate(labels):
            y[e, : len(l)] = torch.as_tensor(l)
        return x, y, x_lens, y_lens

    @torch.no_grad()
    def infer(self, batch, max_symbols=10):
        x, _, x_lens, _ = self.collate(*batch)
        x = self.encode(x.to(self.device), x_lens)
        lens = self.encoded_lengths(x_lens)
        return [self.greedy_decode(e[:n], max_symbols) for e, n in zip(x, lens)]

    def greedy_decode(self, x, max_symbols):
        """
        Decodes one encoded example with shape (time, encoder_dim) by
        taking the best output at each step. Emits at most max_symbols
        labels per frame.
        """
        labels = []
        start = torch.zeros((1, 0), dtype=torch.long, device=x.device)
        y, state = self.predict(start)
        for t in range(x.size(0)):
            for _ in range(max_symbols):
                k = self.joint(x[t], y[0, -1]).argmax().item()
                if k == self.blank:
                    break
                labels.append(k)
                y, state = self.predict(torch.tensor([[k]], device=x.device), state)
        return labels
