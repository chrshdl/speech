import copy
import os
import random

import torch

import speech
import train


def make_config(save_path, epochs):
    return {
        "seed": 0,
        "save_path": str(save_path),
        "data": {
            "train_set": "test.json",
            "dev_set": "test.json",
            "start_and_end": False,
            "num_workers": 0,
        },
        "optimizer": {
            "type": "adam",
            # With one example per batch, the shuffled order of the two
            # recordings in test.json takes one of 70 orders per epoch.
            "batch_size": 1,
            "epochs": epochs,
            "learning_rate": 1e-3,
        },
        "model": {
            "class": "CTC",
            # Dropout makes the result depend on the random number state.
            "dropout": 0.2,
            "encoder": {
                "conv": [[4, 5, 32, 2]],
                "rnn": {"dim": 8, "bidirectional": False, "layers": 2},
                "lookahead": 1,
            },
        },
    }


def train_run(config, seed, resume=False):
    random.seed(seed)
    torch.manual_seed(seed)
    train.run(copy.deepcopy(config), torch.device("cpu"), resume=resume)


def weights(save_path):
    model, _ = speech.load(str(save_path))
    return model.state_dict()


def test_resume(tmp_path):
    train_run(make_config(tmp_path / "full", 3), seed=0)

    # Stop after two epochs, then resume for the third. The second seed
    # shows that the resumed run restores the random number state.
    train_run(make_config(tmp_path / "resumed", 2), seed=0)
    train_run(make_config(tmp_path / "resumed", 3), seed=1, resume=True)

    full = weights(tmp_path / "full")
    resumed = weights(tmp_path / "resumed")
    for name, value in full.items():
        assert torch.equal(value, resumed[name]), name

    state = torch.load(tmp_path / "resumed" / train.STATE)
    assert state["epoch"] == 3


def test_resume_without_state(tmp_path):
    # Checkpoints from before resume support have no training state.
    save_path = tmp_path / "old"
    train_run(make_config(save_path, 2), seed=0)
    os.remove(save_path / train.STATE)
    epoch, best = train.logged_progress(str(save_path))
    assert epoch == 2

    # The resumed run continues the epochs and the best dev CER.
    train_run(make_config(save_path, 3), seed=0, resume=True)
    state = torch.load(save_path / train.STATE)
    assert state["epoch"] == 3
    assert state["best_so_far"] <= best
    assert train.logged_progress(str(save_path))[0] == 3
