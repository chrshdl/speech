import importlib.util
import io
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile

# Loaded under its own name, as other examples have a prepare.py too.
spec = importlib.util.spec_from_file_location(
    "voxpopuli_prepare",
    os.path.join(
        os.path.dirname(__file__), "..", "examples", "voxpopuli", "prepare.py"
    ),
)
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)


def wav_bytes(path, repeat=1):
    audio, rate = soundfile.read(path, dtype="float32")
    out = io.BytesIO()
    soundfile.write(out, np.tile(audio, repeat), rate, format="WAV")
    return out.getvalue()


def write_parquet(path, rows):
    table = pa.table(
        {
            "audio_id": [r[0] for r in rows],
            "language": [0] * len(rows),
            "audio": [{"bytes": r[1], "path": r[0]} for r in rows],
            "raw_text": [r[2] for r in rows],
            "normalized_text": [r[2] for r in rows],
            "gender": ["female"] * len(rows),
            "speaker_id": ["1"] * len(rows),
            "is_gold_transcript": [r[3] for r in rows],
            "accent": ["None"] * len(rows),
        }
    )
    # Small row groups, as the real files have several.
    pq.write_table(table, path, row_group_size=2)


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Serves small Parquet files in place of the downloads."""
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        ("a", wav_bytes("test0.wav"), "the parliament agreed.", True),
        ("b", wav_bytes("test1.wav"), "we vote on the report", True),
        ("c", wav_bytes("test0.wav"), "article 7 applies", True),
        ("d", wav_bytes("test1.wav"), "not checked", False),
        ("e", wav_bytes("test0.wav", repeat=3), "a long speech", True),
    ]
    write_parquet(source / "train", rows)
    write_parquet(source / "dev", [("f", wav_bytes("test1.wav"), "dev words", True)])
    write_parquet(source / "test", [("g", wav_bytes("test0.wav"), "test words", True)])
    monkeypatch.setattr(
        vp, "FILES", {s: [f"{s}.parquet"] for s in ("train", "dev", "test")}
    )
    downloads = []

    def download(name, path):
        downloads.append(name)
        shutil.copy(source / name.removesuffix(".parquet"), path)

    monkeypatch.setattr(vp, "download", download)
    return downloads


def read(path):
    with open(path) as fid:
        return [json.loads(line) for line in fid]


def test_prepare(tmp_path, served):
    out = tmp_path / "out"
    summary = vp.prepare(str(out), max_seconds=2.0)

    train = read(out / "train.json")
    assert [d["text"] for d in train] == [
        "the parliament agreed",
        "we vote on the report",
        "not checked",
    ]
    # Digits need a guess to write out, and the long clip is too long.
    assert summary["train"]["dropped"] == {"text": 1, "too long": 1}
    for d in train:
        info = soundfile.info(d["audio"])
        assert (info.samplerate, info.channels) == (16000, 1)
        assert d["duration"] == pytest.approx(info.duration)
    assert not os.path.exists(out / "clips" / "e.flac")
    assert [d["text"] for d in read(out / "dev.json")] == ["dev words"]
    assert summary["test"]["clips"] == 1

    # The downloads are deleted, and a second run downloads nothing.
    assert not any(name.endswith(".parquet") for name in os.listdir(out))
    assert len(served) == 3
    assert vp.prepare(str(out), max_seconds=2.0) == summary
    assert len(served) == 3


def test_gold_only(tmp_path, served):
    summary = vp.prepare(str(tmp_path / "out"), gold_only=True, max_seconds=2.0)
    assert summary["train"]["dropped"] == {"text": 1, "too long": 1, "not gold": 1}
