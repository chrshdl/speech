import importlib.util
import json
import os

import numpy as np
import pytest
import soundfile
from scipy.signal import resample_poly

# Loaded under its own name, as other examples have a prepare.py too.
spec = importlib.util.spec_from_file_location(
    "myst_prepare",
    os.path.join(os.path.dirname(__file__), "..", "examples", "myst", "prepare.py"),
)
myst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(myst)


@pytest.mark.parametrize(
    "transcript, text, reason",
    [
        ("it's the Magnet that sticks", "it's the magnet that sticks", None),
        ("um the light <noise> goes through", "um the light goes through", None),
        ("[laugh] I think it's wa- water", "i think it's wa water", None),
        ("<whisper> the wire </whisper> is hot", "the wire is hot", None),
        ("<side speech> it moves", "it moves", None),
        ("<<DISCARD>>", None, "label"),
        ("<<NO_SIGNAL>>", None, "label"),
        ("the (( )) circuit", None, "untranscribed speech"),
        ("the ((battery)) circuit", None, "untranscribed speech"),
        ("it is <indiscernible> bigger", None, "untranscribed speech"),
        ("a <mystery> tag", None, "unknown tag"),
        ("there are 3 bulbs", None, "text"),
        ("<noise>", None, "text"),
    ],
)
def test_clean(transcript, text, reason):
    cleaned, why, _ = myst.clean(transcript)
    assert (cleaned, why) == (text, reason)


def test_clean_lists_tags():
    _, _, tags = myst.clean("<noise> the [laugh] wire <<SILENCE>>")
    assert sorted(tags) == ["<<SILENCE>>", "laugh", "noise"]


def write_utterance(root, partition, name, wav, transcript, rate=16000, repeat=1):
    folder = root / partition / "000001" / "myst_000001_session"
    folder.mkdir(parents=True, exist_ok=True)
    audio, _ = soundfile.read(wav, dtype="float32")
    audio = np.tile(audio, repeat)
    if rate != 16000:
        # A stereo clip at another rate is converted.
        audio = np.stack([resample_poly(audio, rate // 100, 160)] * 2, axis=1)
    soundfile.write(folder / f"{name}.flac", audio, rate)
    if transcript is not None:
        (folder / f"{name}.trn").write_text(transcript)
    return folder / f"{name}.flac"


def test_prepare(tmp_path):
    corpus = tmp_path / "myst" / "data"
    kept = write_utterance(corpus, "train", "a", "test0.wav", "the light bends")
    write_utterance(corpus, "train", "b", "test1.wav", "the bulb lights", rate=44100)
    write_utterance(corpus, "train", "c", "test0.wav", None)
    write_utterance(corpus, "train", "d", "test1.wav", "<<DISCARD>>")
    write_utterance(corpus, "train", "e", "test0.wav", "it is long", repeat=3)
    write_utterance(corpus, "development", "f", "test0.wav", "the dev one")
    write_utterance(corpus, "test", "g", "test1.wav", "the test <noise> one")

    out = tmp_path / "out"
    summary, tags = myst.prepare(str(tmp_path / "myst"), str(out), 2.0, workers=1)

    with open(out / "train.json") as fid:
        train = [json.loads(line) for line in fid]
    assert [d["text"] for d in train] == ["the light bends", "the bulb lights"]
    assert summary["train"]["dropped"] == {
        "not transcribed": 1,
        "label": 1,
        "too long": 1,
    }
    # A 16 kHz mono clip is used where it is, others are converted.
    assert train[0]["audio"] == str(kept)
    converted = train[1]["audio"]
    assert converted.startswith(str(out / "clips"))
    info = soundfile.info(converted)
    assert (info.samplerate, info.channels) == (16000, 1)
    assert train[1]["duration"] == pytest.approx(1.570625, abs=1e-3)
    assert summary["dev"]["clips"] == 1
    assert summary["test"]["clips"] == 1
    assert tags == {"<<DISCARD>>": 1, "noise": 1}


def test_needs_the_partitions(tmp_path):
    with pytest.raises(SystemExit, match="development"):
        myst.prepare(str(tmp_path), str(tmp_path / "out"))
