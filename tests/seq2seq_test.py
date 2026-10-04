import numpy as np
import shared
import torch

from speech.models import Seq2Seq


def test_model():
    freq_dim = 120
    vocab_size = 10

    np.random.seed(1337)
    torch.manual_seed(1337)

    conf = shared.model_config
    rnn_dim = conf["encoder"]["rnn"]["dim"]
    conf["decoder"] = {"embedding_dim": rnn_dim, "layers": 2}
    model = Seq2Seq(freq_dim, vocab_size + 1, conf)
    batch = shared.gen_fake_data(freq_dim, vocab_size)
    batch_size = len(batch[0])

    out = model(batch)
    assert torch.isfinite(model.loss(batch))

    assert out.size()[0] == batch_size
    assert out.size()[2] == vocab_size
    assert len(out.size()) == 3

    x, y, x_lens, _ = model.collate(*batch)
    x_enc = model.encode(x, x_lens)
    mask = model.attention_mask(x_enc, x_lens)

    state = None
    out_s = []
    for t in range(y.size()[1] - 1):
        ox, state = model.decode_step(x_enc, y[:, t : t + 1], state=state, mask=mask)
        out_s.append(ox)
    out_s = torch.stack(out_s, dim=1)
    assert out.size() == out_s.size()
    assert torch.allclose(out_s, out, rtol=1e-5, atol=1e-7)


def test_padding():
    freq_dim = 40
    vocab_size = 10
    start, end = vocab_size, vocab_size - 1

    np.random.seed(0)
    torch.manual_seed(0)

    conf = shared.bidirectional_config()
    conf["decoder"] = {"embedding_dim": conf["encoder"]["rnn"]["dim"], "log_t": True}
    model = Seq2Seq(freq_dim, vocab_size + 1, conf)
    model.set_eval()

    inputs, labels = shared.gen_padded_data(freq_dim, end)
    labels = [[start, *l, end] for l in labels]
    batch = (inputs, labels)
    singles = [([i], [l]) for i, l in zip(inputs, labels)]

    with torch.no_grad():
        # The batch loss is the mean of each example's loss on its own.
        loss = model.loss(batch).item()
        expected = np.mean([model.loss(b).item() for b in singles])
        assert np.isclose(loss, expected, rtol=1e-5)

        # Decoding a batch matches decoding each example on its own,
        # up to the first end token.
        def trim(seq):
            return seq[: seq.index(end) + 1] if end in seq else seq

        preds = model.infer(batch, max_len=20)
        for pred, single in zip(preds, singles):
            assert trim(pred) == trim(model.infer(single, max_len=20)[0])
