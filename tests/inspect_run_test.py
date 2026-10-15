import os
import sys

import train_test

import speech

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "commonvoice")
)
import inspect_run


def test_inspect_run(tmp_path):
    train_test.train_run(train_test.make_config(tmp_path, 1), seed=0)
    curve = inspect_run.loss_curve(str(tmp_path))
    # A logged batch per clip of test.json, in at most 20 parts and a
    # last one of the rest.
    with open("test.json") as fid:
        batches = sum(1 for _ in fid)
    starts = [start for start, _, _ in curve]
    assert starts[0] == 0 and starts[-1] < batches and len(starts) <= 21
    assert all(low <= mean for _, mean, low in curve)

    model, preproc = speech.load(str(tmp_path), epoch=0)
    outputs = inspect_run.greedy_outputs(model, preproc, "test.json", clips=1)
    assert len(outputs) == 1
    ref, hyp, blank = outputs[0]
    assert ref and isinstance(hyp, str) and 0 <= blank <= 1
