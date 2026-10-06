import json
import os
import sys

import numpy as np
import pytest
import soundfile
from scipy.signal import resample_poly

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "commonvoice")
)
import preprocess as cv


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("Hello, World!", "hello world"),
        ("Don’t stop.", "don't stop"),
        ("The students' books", "the students' books"),
        ("'Quoted' words", "quoted words"),
        ("Naïve résumé at the café", "naive resume at the cafe"),
        ("A well-known fact — really", "a well known fact really"),
        ("“Why?” she asked…", "why she asked"),
        ("Route 66", None),
        ("Straße", None),
        ("...", None),
        ("", None),
    ],
)
def test_normalize(sentence, expected):
    assert cv.normalize(sentence) == expected


COLUMNS = ["client_id", "path", "sentence", "up_votes", "down_votes", "locale"]


def write_tsv(path, rows):
    with open(path, "w") as fid:
        fid.write("\t".join(COLUMNS) + "\n")
        fid.writelines(
            f"{speaker}\t{clip}\t{sentence}\t2\t0\ten\n"
            for speaker, clip, sentence in rows
        )


def make_corpus(root):
    # Common Voice clips are 48 kHz MP3.
    clips = os.path.join(root, "clips")
    os.makedirs(clips)
    for name, wav in [("a", "test0"), ("c", "test0"), ("d", "test1"), ("f", "test1")]:
        audio, _ = soundfile.read(f"{wav}.wav", dtype="float32")
        soundfile.write(
            os.path.join(clips, f"{name}.mp3"), resample_poly(audio, 3, 1), 48000
        )
    with open(os.path.join(clips, "bad.mp3"), "wb") as fid:
        fid.write(b"not audio")

    write_tsv(os.path.join(root, "dev.tsv"), [("s2", "c.mp3", "Dev sentence.")])
    write_tsv(os.path.join(root, "test.tsv"), [("s3", "d.mp3", "Test sentence!")])
    write_tsv(os.path.join(root, "train.tsv"), [("s1", "a.mp3", "Hello, world!")])
    write_tsv(
        os.path.join(root, "validated.tsv"),
        [
            ("s1", "a.mp3", "Hello, world!"),
            ("s1", "b.mp3", "Route 66"),
            # A dev speaker, a dev clip and a test sentence must not leak.
            ("s2", "e.mp3", "Hello again."),
            ("s2", "c.mp3", "Dev sentence."),
            ("s1", "g.mp3", "Test sentence"),
            ("s1", "bad.mp3", "Broken audio"),
            ("s1", "f.mp3", "Hi there."),
        ],
    )


def read_json(path):
    with open(path) as fid:
        return [json.loads(line) for line in fid]


def test_prepare(tmp_path):
    corpus = tmp_path / "en"
    make_corpus(str(corpus))
    out = tmp_path / "out"

    summary = cv.prepare(str(corpus), str(out), workers=2)
    assert summary["train"]["clips"] == 2
    assert summary["train"]["dropped_text"] == 1
    assert summary["train"]["dropped_audio"] == 1
    assert summary["dev"]["clips"] == summary["test"]["clips"] == 1

    train = read_json(out / "train.json")
    assert [d["text"] for d in train] == ["hello world", "hi there"]
    for datum, wav in zip(train, ["test0", "test1"]):
        info = soundfile.info(datum["audio"])
        assert info.samplerate == 16000
        assert info.channels == 1
        assert datum["duration"] == pytest.approx(info.frames / 16000)
        # MP3 can pad a few milliseconds.
        assert datum["duration"] == pytest.approx(
            soundfile.info(f"{wav}.wav").duration, abs=0.06
        )
    assert read_json(out / "dev.json")[0]["text"] == "dev sentence"

    # A second run reuses the converted clips.
    mtimes = {d["audio"]: os.path.getmtime(d["audio"]) for d in train}
    assert cv.prepare(str(corpus), str(out), workers=2) == summary
    assert {a: os.path.getmtime(a) for a in mtimes} == mtimes
    assert read_json(out / "train.json") == train


def test_official_train_split(tmp_path):
    corpus = tmp_path / "en"
    make_corpus(str(corpus))
    summary = cv.prepare(str(corpus), str(tmp_path / "out"), "train", workers=1)
    assert summary["train"]["clips"] == 1
    assert np.isclose(
        summary["train"]["hours"] * 3600,
        soundfile.info("test0.wav").duration,
        atol=0.06,
    )
