import copy
import os
import random

import pytest
import torch

import speech
import train


def make_config(save_path, epochs, mixed_precision=None):
    return {
        "seed": 0,
        "mixed_precision": mixed_precision,
        "save_path": str(save_path),
        "data": {
            "train_set": "test.json",
            "dev_set": "test.json",
            "start_and_end": False,
            "num_workers": 0,
            "noise": {"source": ".", "snr": [5, 15], "p": 0.5},
            # Starts after the resumed run's stop, at the third epoch.
            "reverb": {
                "delay": [2, 18],
                "decay": [0.55, 0.85],
                "p": 0.5,
                "from_epoch": 2,
            },
            "volume": {"dbfs": [-13, 7], "p": 0.5},
            "pitch": {"factor": [0.9, 1.1]},
            "tempo": {"factor": [0.9, 1.1]},
            "spec_augment": {
                "freq_masks": 2,
                "freq_width": 20,
                "time_masks": 2,
                "time_width": 20,
                "time_ratio": 0.2,
            },
        },
        "optimizer": {
            "type": "adam",
            # With one example per batch, the shuffled order of the two
            # recordings in test.json takes one of 70 orders per epoch.
            "batch_size": 1,
            "epochs": epochs,
            "learning_rate": 1e-3,
            "lr_decay": {"factor": 0.5, "patience": 0},
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


@pytest.mark.parametrize("precision", [None, "bf16", "fp16"])
def test_resume(tmp_path, precision, capsys):
    train_run(make_config(tmp_path / "full", 4, precision), seed=0)

    # Reverb starts at the third epoch, counted from 0.
    log = capsys.readouterr().out
    assert "Epoch 1, augmentations: noise, volume" in log
    assert "Epoch 2, augmentations: noise, reverb, volume" in log

    # Stop after two epochs, then resume for two more. The second seed
    # shows that the resumed run restores the random number state.
    train_run(make_config(tmp_path / "resumed", 2, precision), seed=0)
    train_run(make_config(tmp_path / "resumed", 4, precision), seed=1, resume=True)

    full = weights(tmp_path / "full")
    resumed = weights(tmp_path / "resumed")
    for name, value in full.items():
        assert torch.equal(value, resumed[name]), name

    # The learning rate schedule continues too.
    full_state = torch.load(tmp_path / "full" / train.STATE)
    state = torch.load(tmp_path / "resumed" / train.STATE)
    assert state["epoch"] == 4
    assert state["scheduler"] == full_state["scheduler"]
    assert state["scaler"] == full_state["scaler"]
    assert all(torch.isfinite(v).all() for v in resumed.values())


def test_mixed_precision_changes_training(tmp_path):
    # bf16 must really take effect, or the precision tests prove nothing.
    train_run(make_config(tmp_path / "fp32", 1), seed=0)
    train_run(make_config(tmp_path / "bf16", 1, "bf16"), seed=0)
    fp32, bf16 = weights(tmp_path / "fp32"), weights(tmp_path / "bf16")
    assert any(not torch.equal(fp32[k], bf16[k]) for k in fp32)


def test_unknown_precision():
    with pytest.raises(ValueError):
        train.MixedPrecision("fp8", torch.device("cpu"))


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
