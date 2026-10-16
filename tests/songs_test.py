import asyncio
import json

import httpx
import pytest

from speech.backend import songs
from speech.backend.app import create_app
from speech.intents import IntentRecognizer


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


def test_voices_with_their_own_sound():
    voices = [
        [["A4", 2]],
        {"partials": [0.0, 1.0], "volume": 0.5, "notes": [["C4", 1], [None, 1]]},
    ]
    melody, chord = songs.notes(voices, 60, (1.0, 0.2))
    assert (melody.partials, melody.volume) == ((1.0, 0.2), 1.0)
    assert (chord.partials, chord.volume) == ((0.0, 1.0), 0.5)


def test_envelopes(tmp_path):
    # The song's for its voices, a voice's own, and the synthesizer's
    # otherwise.
    voices = [
        [["A4", 1]],
        {"attack": 0.5, "release": 1.5, "notes": [["C4", 1]]},
    ]
    melody, chord = songs.load_song(
        song_file(tmp_path, "soft", voices=voices, release=0.4)
    ).notes
    assert (melody.attack, melody.release) == (songs.ENVELOPE_SECONDS, 0.4)
    assert (chord.attack, chord.release) == (0.5, 1.5)
    [plain] = songs.notes([[["A4", 1]]], 60)
    assert plain.attack == plain.release == songs.ENVELOPE_SECONDS


@pytest.mark.parametrize("fields", [{"attack": 0}, {"release": 11}, {"attack": "slow"}])
def test_bad_envelope(tmp_path, fields):
    with pytest.raises(ValueError, match="bad.json"):
        songs.load_song(song_file(tmp_path, "bad", **fields))


def test_beats_cut_the_notes(tmp_path):
    # Four beats of two at 120 beats per minute: the third note goes on
    # past the second second, and is cut there.
    voices = [[["C4", 1], ["D4", 1], ["E4", 3]]]
    song = songs.load_song(song_file(tmp_path, "cut", voices=voices, beats=4))
    assert [n.seconds for n in song.notes] == [0.5, 0.5, 1.0]
    assert song.seconds == 2
    with pytest.raises(ValueError, match="starts after"):
        songs.load_song(song_file(tmp_path, "late", voices=voices, beats=2))


def test_loop(tmp_path):
    # Played over and over, the song lasts exactly its beats, also a rest
    # at its end.
    voices = [[["A4", 3], [None, 1]]]
    path = song_file(tmp_path, "loop", voices=voices, beats=4, loop=True, tempo=60)
    score = songs.load_song(path).score()
    assert (score.seconds, score.loop) == (4, True)
    [note] = score.notes
    assert (note.start, note.seconds) == (0, 3)


def test_loop_is_true_or_false(tmp_path):
    with pytest.raises(ValueError, match="loop"):
        songs.load_song(song_file(tmp_path, "bad", loop="yes"))


def test_load(tmp_path):
    song_file(tmp_path, "twinkle")
    (tmp_path / "notes.txt").write_text("not a song")
    [song] = songs.load(tmp_path)
    assert (song.id, song.title) == ("twinkle", "Twinkle, Twinkle, Little Star")
    # Four beats at 120 beats per minute.
    assert song.seconds == 2
    score = song.score()
    assert (score.seconds, score.loop, len(score.notes)) == (2, False, 4)
    first = score.notes[0]
    assert (first.start, first.seconds, first.partials, first.volume) == (
        0,
        0.5,
        [1.0],
        1,
    )
    assert first.frequency == pytest.approx(261.63, abs=0.01)


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


def test_card(tmp_path):
    jukebox = songs.Songs(songs.load(song_file(tmp_path, "twinkle").parent))
    card = asyncio.run(jukebox.card("id", {"song": "twinkle twinkle little star"}))
    assert (card.type, card.id, card.title, card.url) == (
        "SongCard",
        "id",
        "Twinkle, Twinkle, Little Star",
        "/v1/songs/twinkle",
    )
    assert card.seconds == 2
    assert card.UNTIL_DISMISSED
    assert not card.loop
    song_file(tmp_path, "lullaby", names=["lullaby"], loop=True)
    jukebox = songs.Songs(songs.load(tmp_path))
    card = asyncio.run(jukebox.card("id", {"song": "lullaby"}))
    assert (card.loop, card.seconds) == (True, 2)


def test_score_is_served(tmp_path):
    jukebox = songs.Songs(songs.load(song_file(tmp_path, "twinkle").parent))
    app = create_app(None, songs=jukebox)

    async def get(path):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.get(path)

    response = asyncio.run(get("/v1/songs/twinkle"))
    assert response.status_code == 200
    score = response.json()
    assert (score["seconds"], score["loop"], len(score["notes"])) == (2, False, 4)
    assert set(score["notes"][0]) == {
        "start",
        "seconds",
        "frequency",
        "partials",
        "volume",
        "attack",
        "release",
    }
    assert asyncio.run(get("/v1/songs/other")).status_code == 404
