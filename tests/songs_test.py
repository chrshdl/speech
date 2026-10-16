import asyncio
import dataclasses
import io
import json
import math
import wave

import httpx
import numpy as np
import pytest

from speech.backend import songs
from speech.backend.app import create_app
from speech.intents import IntentRecognizer

SR = songs.SAMPLE_RATE


def song_file(directory, name, **fields):
    data = {
        "title": "Twinkle, Twinkle, Little Star",
        "names": ["twinkle twinkle", "twinkle twinkle little star"],
        "tempo": 120,
        "voices": [[["C4", 1], ["C4", 1], ["G4", 1], ["G4", 1]]],
    } | fields
    path = directory / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    ("pitch", "hz"),
    [("A4", 440.0), ("A5", 880.0), ("C4", 261.63), ("Bb3", 233.08), ("F#4", 369.99)],
)
def test_frequency(pitch, hz):
    assert songs.frequency(pitch) == pytest.approx(hz, abs=0.01)


@pytest.mark.parametrize("pitch", ["H4", "C", "Cb#4", "c4"])
def test_no_pitch(pitch):
    with pytest.raises(ValueError):
        songs.frequency(pitch)


def test_notes():
    # At 120 beats per minute a beat is half a second, and a rest moves on.
    voices = [[["A4", 1], [None, 0.5], ["C5", 2]], [["E4", 4]]]
    notes = songs.notes(voices, 120)
    assert [(n.start, n.seconds) for n in notes] == [(0, 0.5), (0.75, 1), (0, 2)]
    assert notes[1].frequency == pytest.approx(523.25, abs=0.01)


def test_envelope():
    ramp = math.ceil(songs.ENVELOPE_SECONDS * SR)
    env = songs.envelope(4 * ramp, 2 * ramp)
    assert env[0] == 0 and env[ramp // 2] == pytest.approx(0.5, abs=0.01)
    assert env[ramp : 2 * ramp].min() == 1
    # Let go, it falls as fast as it rose.
    assert env[2 * ramp + ramp // 2] == pytest.approx(0.5, abs=0.01)
    assert env[3 * ramp :].max() == 0


def spectrum_peak(samples):
    magnitudes = np.abs(np.fft.rfft(samples))
    return np.fft.rfftfreq(len(samples), 1 / SR)[magnitudes.argmax()]


def test_render_plays_the_pitch():
    samples = songs.render(songs.notes([[["A4", 2]]], 60))
    assert spectrum_peak(samples) == pytest.approx(440, abs=1)
    assert np.abs(samples).max() == pytest.approx(songs.PEAK)
    # A second of a note, and the release after it.
    assert len(samples) == 2 * SR + round(songs.ENVELOPE_SECONDS * SR)


def test_render_harmonics():
    # With only the second partial, the octave sounds.
    samples = songs.render(songs.notes([[["A4", 2]]], 60, (0.0, 1.0)))
    assert spectrum_peak(samples) == pytest.approx(880, abs=1)


def test_render_rests_and_repeats():
    samples = songs.render(songs.notes([[["A4", 1], [None, 1], ["A4", 1]]], 60))
    # Silent while resting, after the release.
    assert np.abs(samples[round(1.03 * SR) : 2 * SR]).max() == 0
    # The same note again starts from silence.
    assert np.abs(samples[2 * SR]) < 1e-3 < np.abs(samples[2 * SR + 2000])


def test_render_is_the_same_each_time():
    notes = songs.notes([[["C4", 1], ["E4", 1]], [["G3", 2]]], 90, (1.0, 0.5, 0.25))
    assert np.array_equal(songs.render(notes), songs.render(notes))


def test_voices_with_their_own_sound():
    voices = [
        [["A4", 2]],
        {"partials": [0.0, 1.0], "volume": 0.5, "notes": [["C4", 1], [None, 1]]},
    ]
    melody, chord = songs.notes(voices, 60, (1.0, 0.2))
    assert (melody.partials, melody.volume) == ((1.0, 0.2), 1.0)
    assert (chord.partials, chord.volume) == ((0.0, 1.0), 0.5)
    # The quieter voice, an octave above C4, is half as loud as the melody.
    samples = songs.render([chord, dataclasses.replace(melody, partials=(1.0,))])
    magnitudes = np.abs(np.fft.rfft(samples[:SR]))
    hz = np.fft.rfftfreq(SR, 1 / SR)

    def level(frequency):
        return magnitudes[np.abs(hz - frequency).argmin()]

    assert level(2 * songs.frequency("C4")) / level(440) == pytest.approx(0.5, rel=0.1)


def test_load(tmp_path):
    song_file(tmp_path, "twinkle")
    (tmp_path / "notes.txt").write_text("not a song")
    [song] = songs.load(tmp_path)
    assert (song.id, song.title) == ("twinkle", "Twinkle, Twinkle, Little Star")
    # Four beats at 120 beats per minute, and the last release.
    assert song.seconds == pytest.approx(2 + songs.ENVELOPE_SECONDS, abs=1e-4)
    with wave.open(io.BytesIO(song.render())) as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, SR)
        assert f.getnframes() / SR == pytest.approx(song.seconds)


@pytest.mark.parametrize(
    "fields",
    [
        {"voices": [[["X4", 1]]]},
        {"voices": [[["C4", 0]]]},
        {"voices": [[[None, 1]]]},
        {"partials": []},
        {"partials": [0.0, 0.0]},
        {"partials": [1.0] * 9},
        {"voices": [{"partials": [], "notes": [["C4", 1]]}]},
        {"voices": [{"partials": [1.0]}]},
        {"names": []},
        {"tempo": None},
    ],
)
def test_bad_song(tmp_path, fields):
    path = song_file(tmp_path, "bad", **fields)
    with pytest.raises(ValueError, match="bad.json"):
        songs.load_song(path)


def test_intent_fits_the_model_labels(tmp_path):
    jukebox = songs.Songs(songs.load(song_file(tmp_path, "twinkle").parent))
    assert jukebox.intent.name == songs.NAME
    assert jukebox.intent.slots == {
        "song": ("twinkle twinkle", "twinkle twinkle little star")
    }
    labels = "abcdefghijklmnopqrstuvwxyz' "
    char_to_int = {c: i for i, c in enumerate(labels)}
    recognizer = IntentRecognizer([jukebox.intent], char_to_int, len(labels), 0.04)
    assert len(recognizer.samples) == len(songs.SAMPLES)


def test_no_songs_no_intent():
    assert songs.Songs([]).intent is None


def test_names_are_unique(tmp_path):
    song_file(tmp_path, "one")
    song_file(tmp_path, "two")
    with pytest.raises(ValueError, match="twinkle twinkle"):
        songs.Songs(songs.load(tmp_path))


def test_songs_render_in_the_background(tmp_path):
    song_file(tmp_path, "twinkle")
    song_file(tmp_path, "star", names=["little star"], tempo=60)
    found = songs.load(tmp_path)
    jukebox = songs.Songs(found)

    async def audio():
        return [await jukebox.audio(song.id) for song in found] + [
            await jukebox.audio("other")
        ]

    star, twinkle, other = asyncio.run(audio())
    assert (star, twinkle) == (found[0].render(), found[1].render())
    assert other is None
    # Both are the same notes, at half the tempo for the star.
    assert len(star) > len(twinkle)


def test_card(tmp_path):
    jukebox = songs.Songs(songs.load(song_file(tmp_path, "twinkle").parent))
    card = asyncio.run(jukebox.card("id", {"song": "twinkle twinkle little star"}))
    assert (card.type, card.id, card.title, card.url) == (
        "SongCard",
        "id",
        "Twinkle, Twinkle, Little Star",
        "/v1/songs/twinkle",
    )
    assert card.seconds == pytest.approx(2.03, abs=0.01)
    assert card.UNTIL_DISMISSED


def test_audio_is_served(tmp_path):
    jukebox = songs.Songs(songs.load(song_file(tmp_path, "twinkle").parent))
    app = create_app(None, songs=jukebox)

    async def get(path):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.get(path)

    response = asyncio.run(get("/v1/songs/twinkle"))
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == asyncio.run(jukebox.audio("twinkle"))
    assert asyncio.run(get("/v1/songs/other")).status_code == 404
    assert asyncio.run(get("/v1/songs/twinkle")).status_code == 200
