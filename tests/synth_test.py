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

from speech.backend import d50, songs

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


def play_stereo(tmp_path, data, seconds, stop=None):
    """
    The left and right samples of the score played for seconds, what it
    posted, and when it ended.
    """
    (tmp_path / "score.json").write_text(json.dumps(data))
    args = [tmp_path / "score.json", tmp_path / "out.f32", SR, round(seconds * SR)]
    if stop is not None:
        args.append(round(stop * SR))
    result = subprocess.run(
        ["node", HARNESS, *map(str, args)], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    samples = np.fromfile(tmp_path / "out.f32", dtype=np.float32).reshape(-1, 2)
    return samples.T, report["posted"], report["returnedFalse"]


def play(tmp_path, data, seconds, stop=None):
    """The samples of the score played for seconds, what it posted, and when it ended."""
    (left, right), posted, returned_false = play_stereo(tmp_path, data, seconds, stop)
    # Without a patch, the left and right are the same.
    assert np.array_equal(left, right)
    return left, posted, returned_false


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


# The D-50's voice, for notes with a patch.


def patch_score(voices, patch="fantasia", data=None, tempo=60):
    """The score of the voices played with the patch, or with its data instead."""
    notes = tuple(songs.notes(voices, tempo, patch=patch))
    song = songs.Song("song", "Song", ("song",), notes)
    score = song.score().model_dump(by_alias=True)
    if data is not None:
        score["patches"][patch] = data
    return score


def plain(**partial):
    """
    Fantasia made a single synth partial with neither effects nor
    modulation: a square at C4, full cutoff, at full level at once, and
    silent at once when let go, with the partial's parameters changed.
    """
    data = d50.load("fantasia")
    data["patch"].update(key_mode=0, reverb_balance=0, total_volume=100)
    common = data["upper"]["common"]
    common.update(
        structure=0,
        partial_mute=1,
        chorus_balance=0,
        eq_lg=12,
        eq_hg=12,
        pitch_lfo_depth=0,
    )
    for n in (1, 2):
        data["upper"][f"partial{n}"].update(
            wg_coarse=36,
            wg_fine=50,
            wg_keyfollow=11,
            wg_lfo_mode=0,
            wg_waveform=0,
            pw=0,
            pw_velocity=7,
            tvf_cutoff=100,
            tvf_resonance=0,
            tvf_bias_level=7,
            tvf_env_depth=0,
            tva_level=100,
            tva_velocity=50,
            tva_bias_level=12,
            tva_t1=0,
            tva_t2=0,
            tva_t3=0,
            tva_t4=0,
            tva_t5=0,
            tva_l1=100,
            tva_l2=100,
            tva_l3=100,
            tva_sus=100,
            tva_lfo_depth=0,
        )
    data["upper"]["partial1"].update(partial)
    return data


def level(samples, frequency, seconds):
    """The magnitude of the frequency in the samples, of a sine of 1."""
    hz = np.fft.rfftfreq(len(samples), 1 / SR)
    magnitudes = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return magnitudes[np.abs(hz - frequency).argmin()] / (len(samples) / 4)


def decibels(samples, at, seconds=0.02):
    window = samples[round(at * SR) : round((at + seconds) * SR)]
    return 20 * np.log10(rms(window))


def test_patch_square_and_sawtooth_pitch(tmp_path):
    # An LA sawtooth is the square times a cosine of its frequency: at
    # pulse width 0, symmetrical, it sounds an octave above the square, as
    # the D-50's manual says, "pulse width 50% raises the pitch by an
    # octave", but at other pulse widths the square's pitch sounds too.
    def play_partial(**partial):
        data = plain(**partial)
        samples, _, _ = play(tmp_path, patch_score([[["C4", 2]]], data=data), 2)
        return samples[SR // 2 : 3 * SR // 2]

    square = play_partial()
    assert spectrum_peak(square) == pytest.approx(261.63, abs=1)
    sawtooth = play_partial(wg_waveform=1)
    assert spectrum_peak(sawtooth) == pytest.approx(523.25, abs=1)
    assert level(sawtooth, 261.63, 1) < 1e-4
    wide = play_partial(wg_waveform=1, pw=100)
    assert level(wide, 261.63, 1) > 0.1 * level(wide, spectrum_peak(wide), 1)


def test_patch_cutoff_rounds_the_square(tmp_path):
    # The lower the cutoff, the wider the cosines of the square's edges,
    # and the weaker its harmonics; at 22, the LA chip's middle on the
    # D-50's brighter scale, it is all but a sine, and below it fades, as
    # the D-50's manual says. Quiet enough for the limiter not to add
    # harmonics.
    def harmonics(cutoff):
        data = plain(tvf_cutoff=cutoff, tva_level=40)
        samples, _, _ = play(tmp_path, patch_score([[["C4", 1]]], data=data), 1)
        samples = samples[SR // 4 : 3 * SR // 4]
        return [level(samples, h * 261.63, 0.5) for h in (1, 3, 9)]

    fundamental, third, ninth = harmonics(100)
    assert third / fundamental == pytest.approx(1 / 3, rel=0.2)
    assert ninth / fundamental > 0.05
    _, _, ninth = harmonics(40)
    assert ninth / fundamental < 0.01
    sine, third, _ = harmonics(22)
    assert third / sine < 0.02
    faded, _, _ = harmonics(12)
    assert faded < 0.5 * sine


def test_patch_envelope_rates(tmp_path):
    # An LA ramp's increment is the time setting, the D-50's 1.26 times the
    # MT-32's, less 8 * log2 of the distance, a rate of
    # 2^((increment + 24) / 8) / 2^18 of a 1/16 octave at 32 kHz: rising
    # 100 at time 64, as 81, by 24.1 units a second, 9.1 dB, and falling at
    # time 64 by 57.3 units, 21.6 dB a second.
    data = plain(tva_t1=64, tva_t5=64)
    samples, _, _ = play(tmp_path, patch_score([[["C4", 2]]], data=data), 3)
    rise = (decibels(samples, 1.5) - decibels(samples, 0.5)) / 1.0
    assert rise == pytest.approx(24.1 * 6.0206 / 16, rel=0.05)
    fall = (decibels(samples, 2.1) - decibels(samples, 2.6)) / 0.5
    assert fall == pytest.approx(57.3 * 6.0206 / 16, rel=0.05)


def test_patch_ring_modulation(tmp_path):
    # Structure 2 adds the ring modulation of partial 2, a fifth up, to
    # partial 1: their difference sounds, as it does not mixed.
    def play_structure(structure):
        data = plain(tvf_cutoff=50, tva_level=40)
        data["upper"]["common"].update(structure=structure, partial_mute=3)
        data["upper"]["partial2"].update(wg_coarse=43, tvf_cutoff=50, tva_level=40)
        samples, _, _ = play(tmp_path, patch_score([[["C4", 1]]], data=data), 1)
        return samples[SR // 4 : 3 * SR // 4]

    difference = 392.0 - 261.63
    mixed = play_structure(0)
    ringed = play_structure(1)
    assert level(mixed, difference, 0.5) < 1e-4
    assert level(ringed, difference, 0.5) > 0.05 * level(ringed, 261.63, 0.5)


def test_fantasia(tmp_path):
    (left, right), posted, returned_false = play_stereo(
        tmp_path, patch_score([[["C4", 5]]]), 18
    )
    mono = (left + right) / 2
    # As measured on a D-50: the synth sounds at the key, with its
    # harmonics, and the bells strike two octaves and a fourth above it,
    # 5.34 times its frequency, and fade by about 14 dB a second. Over
    # seconds, as the chorus sweeps the notches it makes.
    held = mono[SR : 4 * SR]
    assert spectrum_peak(held) == pytest.approx(261.63, abs=1)
    assert 0.1 < level(held, 2 * 261.63, 3) / level(held, 261.63, 3) < 0.6
    bell = 5.34 * 261.63
    strike = mono[: SR // 4]
    assert level(strike, bell, 0.25) > 0.1 * level(strike, 261.63, 0.25)
    faded = 20 * np.log10(level(strike, bell, 0.25) / level(held, bell, 3))
    assert 8 < faded < 30
    # In stereo, by the chorus and the reverb.
    assert not np.allclose(left[SR : 2 * SR], right[SR : 2 * SR])
    # Let go after 5 seconds, it fades, and the song ends once the reverb
    # has rung out too, and is silent after.
    assert rms(mono[7 * SR : 8 * SR]) < 0.1 * rms(held)
    [(end, message)] = posted
    assert message == "ended" and 8 * SR < end < 17 * SR
    assert np.abs(left[returned_false:]).max() == 0


def test_fantasia_loud_as_the_tones(tmp_path):
    melody = [[[p, 0.5] for p in ("C5", "D5", "E5", "G5", "F5", "E5", "D5", "C5")]]
    chords = [[["C3", 2], ["F3", 2]], [["G3", 2], ["A3", 2]], [["E4", 2], ["C4", 2]]]
    samples, _, _ = play_stereo(tmp_path, patch_score(melody + chords, tempo=120), 3)
    assert 0.2 < np.abs(samples).max() < 0.6


def test_fantasia_the_same_each_time(tmp_path):
    data = patch_score([[["C4", 1], ["E4", 1]], [["G3", 2]]])
    first, _, _ = play_stereo(tmp_path, data, 3)
    again, _, _ = play_stereo(tmp_path, data, 3)
    assert np.array_equal(first, again)


def test_patch_with_additive_voices(tmp_path):
    # A voice of the additive synthesizer sounds beside the patch's.
    notes = tuple(songs.notes([[["A4", 1]]], 60))
    notes += tuple(songs.notes([[["C3", 1]]], 60, patch="fantasia"))
    score = (
        songs.Song("song", "Song", ("song",), notes).score().model_dump(by_alias=True)
    )
    (left, _), _, _ = play_stereo(tmp_path, score, 1)
    window = left[SR // 4 : 3 * SR // 4]
    assert level(window, 440, 0.5) > 0.1 and level(window, 130.81, 0.5) > 0.01


# The keyboard, a patch played live.


def play_keys(tmp_path, messages, seconds, patch="fantasia"):
    """
    The left and right samples of the keyboard sent the messages, each at a
    second, and what it posted, and when it ended.
    """
    data = {
        "processor": "keyboard",
        "options": {"patch": d50.load(patch)},
        "messages": [[round(at * SR), message] for at, message in messages],
    }
    return play_stereo(tmp_path, data, seconds)


def test_keyboard_plays_a_held_key(tmp_path):
    c4 = songs.frequency("C4")
    (left, right), _, returned_false = play_keys(
        tmp_path, [(0.5, {"down": 1, "frequency": c4}), (3, {"up": 1})], 9
    )
    mono = (left + right) / 2
    # Silent before the key, and sounding at its pitch while held, also
    # long after it was pressed.
    assert np.abs(mono[: SR // 2]).max() == 0
    held = mono[round(2 * SR) : 3 * SR]
    assert spectrum_peak(held) == pytest.approx(c4, abs=1)
    assert rms(held) > 0.05
    # Let go, it fades, and once the reverb has rung out it is silent, but
    # goes on for the next key.
    assert rms(mono[round(4.5 * SR) : round(5.5 * SR)]) < 0.1 * rms(held)
    assert np.abs(mono[8 * SR :]).max() == 0
    assert returned_false is None


def test_keyboard_plays_as_many_keys_as_a_d50(tmp_path):
    # Fantasia plays both tones, so 8 keys at once: a 9th takes the place
    # of the first, which fades at once, while the second sounds on.
    notes = ["C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4"]
    messages = [
        (0.1 * n, {"down": n, "frequency": songs.frequency(note)})
        for n, note in enumerate(notes)
    ]
    (left, _), _, _ = play_keys(tmp_path, messages, 3)
    before = left[round(0.5 * SR) : round(0.8 * SR)]
    after = left[round(1.5 * SR) : round(2.5 * SR)]
    c3, d3 = songs.frequency("C3"), songs.frequency("D3")
    assert level(after, c3, 1) < 0.1 * level(before, c3, 0.3)
    assert level(after, d3, 1) > 0.3 * level(before, d3, 0.3)


def test_keyboard_stops(tmp_path):
    messages = [(0, {"down": 1, "frequency": 440.0}), (1, "stop")]
    (left, _), _, returned_false = play_keys(tmp_path, messages, 2)
    assert returned_false == -(-SR // BLOCK) * BLOCK + BLOCK
    assert np.abs(left[returned_false:]).max() == 0
