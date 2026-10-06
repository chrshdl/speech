import json

import shared
import torch

import benchmark


def test_benchmark():
    config = dict(shared.model_config, **{"class": "CTC"})
    speed = benchmark.benchmark(
        config, "test.json", 2, None, steps=2, device=torch.device("cpu"), warmup=1
    )
    assert speed > 0


def test_training_hours(tmp_path):
    hours = benchmark.training_hours({"train_set": "test.json"})
    with open("test.json") as fid:
        expected = sum(json.loads(line)["duration"] for line in fid) / 3600
    assert hours == expected
    assert (
        benchmark.training_hours({"train_set": str(tmp_path / "missing.json")}) is None
    )
