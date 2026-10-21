"""
Tests the speaker of web/assistant/speaker.js, which plays the speech the
service streams, with Node, as the audio thread of an AudioWorklet runs it.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

HARNESS = os.path.join(os.path.dirname(__file__), "synth_harness.mjs")
# The speech's rate, a Piper voice's, and the context's.
SPEECH_SR = 22050
SR = 44100
VOLUME = 0.5
BLOCK = 128
# As in speaker.js.
FADE_SECONDS = 0.02

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def tone(seconds, start=0.0, hz=440):
    """A sine of 0.9 at hz, from start seconds, at the speech's rate."""
    t = start + np.arange(round(seconds * SPEECH_SR)) / SPEECH_SR
    return 0.9 * np.sin(2 * np.pi * hz * t)


def block(seconds):
    """The frame of the block at seconds, where messages arrive."""
    return round(seconds * SR / BLOCK) * BLOCK


def int16(samples):
    return {"int16": np.round(samples * 32767).astype(int).tolist()}


def expected(seconds):
    """The tone as the speaker plays it, at the context's rate."""
    t = np.arange(round(seconds * SR)) / SR
    return VOLUME * 0.9 * np.sin(2 * np.pi * 440 * t)


def play(tmp_path, messages, seconds):
    """
    The samples of the speaker sent the messages, each [frame, data], for
    seconds, what it posted, and the frame it ended at.
    """
    data = {
        "processor": "speaker",
        "module": "speaker.js",
        "options": {"sampleRate": SPEECH_SR, "volume": VOLUME},
        "messages": messages,
    }
    (tmp_path / "speech.json").write_text(json.dumps(data))
    args = [tmp_path / "speech.json", tmp_path / "out.f32", SR, round(seconds * SR)]
    result = subprocess.run(
        ["node", HARNESS, *map(str, args)], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    samples = np.fromfile(tmp_path / "out.f32", dtype=np.float32).reshape(-1, 2)
    return samples[:, 0], report["posted"], report["returnedFalse"]


def test_resamples_to_the_context(tmp_path):
    # Arriving in pieces that split no sample, but not at a sentence.
    audio = tone(1.0)
    pieces = np.split(audio, [1000, 4001])
    messages = [[0, int16(piece)] for piece in pieces] + [[0, "end"]]
    samples, posted, ended = play(tmp_path, messages, 1.5)
    played = samples[:SR]
    error = played - expected(1.0)
    # The 4 point interpolation stays far below the speech, about 60 dB.
    assert np.sqrt(np.mean(error**2)) < 1e-3 * np.sqrt(np.mean(played**2))
    assert np.abs(samples[SR + 2 :]).max() == 0
    # It ends once played, within the block of the last sample.
    assert posted == [[ended - BLOCK, "ended"]]
    assert SR <= ended <= SR + BLOCK


def test_waits_for_the_next_sentence(tmp_path):
    # The second half comes 2 seconds in, as if synthesized slowly.
    audio = tone(1.0)
    half = len(audio) // 2
    later = block(2.0)
    messages = [
        [0, int16(audio[:half])],
        [later, int16(audio[half:])],
        [later, "end"],
    ]
    samples, posted, ended = play(tmp_path, messages, 3.0)
    first = samples[:later]
    waited = np.flatnonzero(first)[-1] + 1
    # It plays all it can, and then silence.
    assert waited == pytest.approx(SR / 2, abs=8)
    assert np.abs(first[waited:]).max() == 0
    # It goes on where it waited, losing nothing.
    played = np.concatenate([first[:waited], samples[later:ended]])
    error = played[:SR] - expected(1.0)
    assert np.sqrt(np.mean(error**2)) < 1e-3 * np.sqrt(np.mean(played**2))
    assert [message for _, message in posted] == ["ended"]


def test_stop_fades_out(tmp_path):
    stop = block(0.5)
    messages = [[0, int16(tone(2.0))], [stop, "stop"]]
    samples, posted, ended = play(tmp_path, messages, 2.0)
    fade = round(FADE_SECONDS * SR)
    full = np.abs(samples[stop - fade : stop]).max()
    halfway = np.abs(samples[stop + fade // 2 - 50 : stop + fade // 2 + 50]).max()
    assert full == pytest.approx(VOLUME * 0.9, rel=0.01)
    assert halfway == pytest.approx(full / 2, rel=0.1)
    assert np.abs(samples[stop + fade :]).max() < 0.01 * full
    assert [message for _, message in posted] == ["ended"]
    assert stop + fade <= ended <= stop + fade + BLOCK


def test_stop_while_waiting_ends_at_once(tmp_path):
    stop = block(1.0)
    messages = [[0, int16(tone(0.5))], [stop, "stop"]]
    samples, posted, ended = play(tmp_path, messages, 2.0)
    assert np.abs(samples[round(0.6 * SR) :]).max() == 0
    assert posted == [[stop, "ended"]]
    assert ended == stop + BLOCK
