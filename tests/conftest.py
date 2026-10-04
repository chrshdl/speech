import os

import pytest


@pytest.fixture(autouse=True)
def tests_dir(monkeypatch):
    # The test data refers to its audio files relative to this directory.
    monkeypatch.chdir(os.path.dirname(__file__))
