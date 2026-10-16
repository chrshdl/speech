"""
Tests the synthesizer of web/assistant/synth.js, which plays a song's notes
on the client, with Node, as the audio thread of an AudioWorklet runs it.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from speech.backend import songs

HARNESS = os.path.join(os.path.dirname(__file__), "synth_harness.mjs")
TWINKLE = os.path.join(
    os.path.dirname(__file__), "..", "examples", "songs", "twinkle.json"
)
SR = 44100
BLOCK = 128
# As in synth.js.
ENVELOPE_SECONDS = 0.025

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def score(voices, tempo=60, partials=(1.0,), beats=None, loop=False, sound=None):
    notes = tuple(songs.notes(voices, tempo, partials, sound))
    length = None
    if beats is not None:
        length = beats * 60 / tempo
        notes = songs.cut(notes, length)
    song = songs.Song("song", "Song", ("song",), notes, loop, length)
    return song.score().model_dump(by_alias=True)


def play(tmp_path, data, seconds, stop=None):
    """The samples of the score played for seconds, what it posted, and when it ended."""
    (tmp_path / "score.json").write_text(json.dumps(data))
    args = [tmp_path / "score.json", tmp_path / "out.f32", SR, round(seconds * SR)]
    if stop is not None:
        args.append(round(stop * SR))
    result = subprocess.run(
        ["node", HARNESS, *map(str, args)], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    samples = np.fromfile(tmp_path / "out.f32", dtype=np.float32)
    return samples, report["posted"], report["returnedFalse"]


def rms(samples):
    return np.sqrt(np.mean(samples**2))


def spectrum_peak(samples):
    magnitudes = np.abs(np.fft.rfft(samples))
    return np.fft.rfftfreq(len(samples), 1 / SR)[magnitudes.argmax()]


def test_plays_the_pitch(tmp_path):
    samples, _, _ = play(tmp_path, score([[["A4", 2]]]), 2)
    assert spectrum_peak(samples) == pytest.approx(440, abs=1)


def test_harmonics(tmp_path):
    # With only the second partial, the octave sounds.
    samples, _, _ = play(tmp_path, score([[["A4", 2]]], partials=(0.0, 1.0)), 2)
    assert spectrum_peak(samples) == pytest.approx(880, abs=1)


def test_detuned_pairs(tmp_path):
    # Each harmonic has a sine 3 Hz plus 1 Hz per harmonic above and below,
    # at 0.4 of its level.
    samples, _, _ = play(tmp_path, score([[["A4", 8]]]), 8)
    magnitudes = np.abs(np.fft.rfft(samples[SR : 7 * SR]))
    hz = np.fft.rfftfreq(6 * SR, 1 / SR)

    def level(frequency):
        return magnitudes[np.abs(hz - frequency).argmin()]

    assert level(443) / level(440) == pytest.approx(0.4, abs=0.02)
    assert level(437) / level(440) == pytest.approx(0.4, abs=0.02)


def test_envelope_rests_and_repeats(tmp_path):
    voices = [[["A4", 1], [None, 1], ["A4", 1]]]
    samples, _, _ = play(tmp_path, score(voices), 3)
    ramp = round(ENVELOPE_SECONDS * SR)
    # It fades in over 25 ms rather than click.
    assert np.abs(samples[:10]).max() < 0.01
    assert rms(samples[: ramp // 2]) < 0.5 * rms(samples[2 * ramp : 4 * ramp])
    # Silent while resting, and the same note again starts from silence.
    assert np.abs(samples[SR : 2 * SR]).max() == 0
    assert np.abs(samples[2 * SR : 2 * SR + 10]).max() < 0.01


def test_slow_attack(tmp_path):
    # Swelling over half a second, a note is still quiet after 25 ms.
    fast, _, _ = play(tmp_path, score([[["A4", 2]]]), 1)
    slow, _, _ = play(tmp_path, score([[["A4", 2]]], sound={"attack": 0.5}), 1)
    window = slice(round(0.02 * SR), round(0.05 * SR))
    assert rms(slow[window]) < 0.2 * rms(fast[window])
    late = slice(round(0.6 * SR), None)
    assert rms(slow[late]) == pytest.approx(rms(fast[late]), rel=0.05)


def test_release_fades_past_the_end(tmp_path):
    # A note of a second, fading over one more, and the song ends then.
    data = score([[["A4", 1]]], sound={"release": 1.0})
    samples, posted, returned_false = play(tmp_path, data, 3)
    tail = slice(round(1.3 * SR), round(1.5 * SR))
    assert 0.01 < rms(samples[tail]) < rms(samples[: SR // 2])
    [(end, message)] = posted
    assert message == "ended" and 1.9 * SR < end < 2.1 * SR
    assert np.abs(samples[returned_false:]).max() == 0


def test_release_carries_over_the_loop(tmp_path):
    # A note ending the song fades on into the next time round, rather than
    # being cut where the song starts over.
    voices = [[[None, 1], ["A4", 1]]]
    data = score(voices, beats=2, loop=True, sound={"release": 0.5})
    samples, _, _ = play(tmp_path, data, 3)
    # The first second of the song is a rest, the first time round silent.
    assert np.abs(samples[: SR - 10]).max() == 0
    assert rms(samples[2 * SR : round(2.2 * SR)]) > 0.01


def test_loud_as_the_tones(tmp_path):
    # The loudest moments of a song are about as loud as the client's
    # tones, 0.3, and the tanh of the mix keeps any below 1.
    with open(TWINKLE) as f:
        data = json.load(f)
    twinkle = score(data["voices"], data["tempo"], tuple(data["partials"]))
    samples, _, _ = play(tmp_path, twinkle, twinkle["seconds"])
    assert 0.2 < np.abs(samples).max() < 0.4
    chords = [[["C4", 4]], [["E4", 4]], [["G4", 4]], [["C5", 4]]]
    samples, _, _ = play(tmp_path, score(chords, partials=(1, 1, 1, 1)), 4)
    assert 0.2 < np.abs(samples).max() < 0.5


def test_ends(tmp_path):
    samples, posted, returned_false = play(tmp_path, score([[["A4", 1]]]), 2)
    # The end, one second in, is reported once, from the block that has it,
    # which is the last.
    assert posted == [[SR // BLOCK * BLOCK, "ended"]]
    assert returned_false == (SR // BLOCK + 1) * BLOCK
    assert np.abs(samples[SR:]).max() == 0


def test_loops(tmp_path):
    # Played over and over, each time as the first, also a rest at the end.
    data = score([[["A4", 1], ["C5", 0.5], [None, 0.5]]], beats=2, loop=True)
    samples, posted, returned_false = play(tmp_path, data, 5)
    assert posted == [] and returned_false is None
    first, second = samples[: 2 * SR], samples[2 * SR : 4 * SR]
    assert np.array_equal(first, second)
    assert np.abs(first).max() > 0.1
    assert np.abs(first[round(1.5 * SR) :]).max() == 0


def test_stops(tmp_path):
    data = score([[["A4", 4]]], beats=4, loop=True)
    samples, _, returned_false = play(tmp_path, data, 4, stop=1)
    # Told to stop in a block, the next block is silent, and the last.
    stopped = -(-SR // BLOCK) * BLOCK
    assert returned_false == stopped + BLOCK
    assert np.abs(samples[: stopped - BLOCK]).max() > 0.1
    assert np.abs(samples[stopped:]).max() == 0


def test_the_same_each_time(tmp_path):
    data = score([[["C4", 1], ["E4", 1]], [["G3", 2]]], partials=(1.0, 0.5, 0.25))
    first, _, _ = play(tmp_path, data, 2)
    again, _, _ = play(tmp_path, data, 2)
    assert np.array_equal(first, again)
