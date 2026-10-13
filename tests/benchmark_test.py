import json
import sys

import numpy as np
import pytest
import shared
import torch

import benchmark
import train
from speech import loader, models


def test_clip_lengths():
    durations = np.linspace(1, 20, 100)
    lengths = benchmark.clip_lengths(durations)
    assert lengths[0] == 1.0
    assert lengths[-1] == 20.0
    assert lengths == sorted(set(lengths))


def test_batch_lengths():
    lengths = benchmark.batch_lengths([1, 3, 2, 5, 4, 6, 7], 2)
    # The last clip does not fill a batch, so the loader drops it.
    assert list(lengths) == [3, 5, 6]


def test_model_hours():
    durations = [1, 1, 2, 2, 4, 4]
    seconds = {1: 10.0, 4: 40.0}
    # Batches of 1, 2 and 4 s take 10, 20 and 40 s, and 3 batches in steps
    # of 2 make 2 optimizer steps.
    hours = benchmark.model_hours(durations, 2, 2, seconds, 5.0)
    assert hours == pytest.approx((10 + 20 + 40 + 2 * 5) / 3600)
    # The longest batch ran out of memory.
    assert benchmark.model_hours(durations, 2, 2, {1: 10.0, 4: None}, 5.0) is None


def test_time_step():
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    config = dict(shared.model_config, **{"class": "CTC"})
    device = torch.device("cpu")
    model = models.CTC(preproc.input_dim, preproc.vocab_size, config)
    batch = benchmark.synthetic_batch(preproc, 2, 1.0, 10.0)
    precision = train.MixedPrecision(None, device)
    seconds, peak = benchmark.time_step(model, batch, precision, 2, 1, device)
    assert seconds > 0
    assert peak is None


def test_loader_rate():
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    rate = benchmark.loader_rate("test.json", preproc, {}, 2, 0, 4)
    assert rate > 0


def test_main(tmp_path, monkeypatch, capsys):
    config = {
        "save_path": str(tmp_path),
        "data": {"train_set": "test.json", "start_and_end": False},
        "optimizer": {
            "batch_size": 2,
            "accumulate": 2,
            "grad_clip": 400,
            "learning_rate": 1e-3,
        },
        "model": dict(shared.model_config, **{"class": "CTC"}),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    argv = ["benchmark.py", str(path), "--batch-sizes", "2,4", "--steps", "1"]
    argv += ["--epoch-hours", "1", "--loader-clips", "0", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    benchmark.main()
    out = capsys.readouterr().out
    assert "8 clips" in out
    # The config's batch size scales to the given hour.
    assert "    2     2   float32" in out
    assert "    4     1   float32" in out
    assert "scaled to 1 h" in out
