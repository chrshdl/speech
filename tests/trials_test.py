import os
import sys

import shared
import torch

from speech import loader
from speech.models import CTC

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "commonvoice")
)
import trials


def test_char_loss():
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False, bigrams=True)
    model = CTC(preproc.input_dim, preproc.vocab_size, shared.model_config)
    char_loss = trials.CharLoss(model, preproc)
    # The characters of the bigrams, and a blank.
    assert set(char_loss.char_to_int) == set("".join(preproc.char_to_int))
    assert char_loss.fc.out_features == len(char_loss.char_to_int) + 1

    data = [
        preproc.preprocess(f"test{i}.wav", t) for i, t in enumerate(["hello", "wo"])
    ]
    batch = tuple(zip(*data))
    bigrams, total = char_loss(batch)
    assert torch.allclose(bigrams, model.loss(batch))
    assert total > bigrams
    total.backward()
    assert char_loss.fc.weight.grad.abs().sum() > 0
    assert model.fc.weight.grad.abs().sum() > 0
