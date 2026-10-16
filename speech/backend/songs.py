"""
Plays songs asked for by name, such as "play twinkle twinkle". Each song is
a JSON file in a directory (backend.py --songs), which the service reads
when it starts. It answers with a SongCard, whose client fetches the
song's notes from GET /v1/songs/{songId} and plays them live with the
voice of an additive synthesizer, see web/assistant/synth.js.

A song file:

    {
      "title": "Twinkle, Twinkle, Little Star",
      "names": ["twinkle twinkle", "twinkle twinkle little star"],
      "tempo": 100,
      "partials": [1.0, 0.3, 0.12],
      "voices": [
        [["C4", 1], ["C4", 1], ["G4", 1], ["G4", 1], [null, 1]],
        {"partials": [1.0], "volume": 0.4, "notes": [["C3", 4], ["E3", 2]]}
      ]
    }

names are how the song is asked for, in the model's characters, as in
"play {song}". tempo is in beats per minute. partials, optional, are the
amplitudes of the harmonics, the fundamental first, as the synthesizer's
bubbles set them. Each voice is played from the start, a note after the
other: a pitch, such as "Bb3" or "F#4", or null for a rest, and its length
in beats. A voice can also be an object with its notes and its own
partials, and a volume, 1 by default, such as for quieter chords.
Optionally, beats is how long the song lasts, which cuts notes that last
longer, and loop, if true, plays it over and over until the card is
closed: a looping song lasts exactly its beats, or until its last note
ends. The synthesizer lets every note go before it ends, so nothing
sounds on past the end, and the loop has no seam.
examples/songs/from_midi.py makes a song file from a MIDI file.
"""

import dataclasses
import json
import os
import re

from speech.backend.messages import SongCard, SongNote, SongScore
from speech.intents import Intent

# The intent's name.
NAME = "PlaySong"
SAMPLES = (
    "play {song}",
    "sing {song}",
    "play the song {song}",
    "sing the song {song}",
    "can you play {song}",
    "can you sing {song}",
    "please play {song}",
    "please sing {song}",
    "{song}",
)

# The synthesizer's harmonics, its bubbles.
MAX_PARTIALS = 8

PITCH = re.compile(r"([A-G])(#|b)?(-?\d)")
STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def frequency(pitch):
    """The frequency of a pitch such as "Bb3", equal-tempered at A4 = 440 Hz."""
    match = PITCH.fullmatch(pitch)
    if match is None:
        raise ValueError(f"{pitch!r} is no pitch, such as C4, F#4 or Bb3")
    step, accidental, octave = match.groups()
    midi = 12 * (int(octave) + 1) + STEPS[step] + {"#": 1, "b": -1}.get(accidental, 0)
    return 440.0 * 2 ** ((midi - 69) / 12)


@dataclasses.dataclass(frozen=True)
class Note:
    # In seconds from the song's start.
    start: float
    seconds: float
    frequency: float
    # The amplitudes of the voice's harmonics, and its volume.
    partials: tuple[float, ...] = (1.0,)
    volume: float = 1.0


def check_partials(partials):
    if not 0 < len(partials) <= MAX_PARTIALS or min(partials) < 0 or max(partials) <= 0:
        raise ValueError(
            f"partials needs 1 to {MAX_PARTIALS} amplitudes of 0 or more, one above"
        )
    return tuple(partials)


def notes(voices, tempo, partials=(1.0,)):
    """
    The notes of the voices, with their times in seconds, and partials for
    the voices without their own.
    """
    beat = 60 / tempo
    notes = []
    for voice in voices:
        voice_partials, volume = partials, 1.0
        if isinstance(voice, dict):
            voice_partials = check_partials(voice.get("partials", partials))
            volume = voice.get("volume", 1.0)
            voice = voice["notes"]
        time = 0.0
        for pitch, beats in voice:
            if beats <= 0:
                raise ValueError(f"A note lasts {beats} beats")
            if pitch is not None:
                notes.append(
                    Note(
                        time * beat,
                        beats * beat,
                        frequency(pitch),
                        voice_partials,
                        volume,
                    )
                )
            time += beats
    return notes


def cut(notes, seconds):
    """The notes, cut to end by seconds, raising ValueError for one after it."""
    if any(note.start >= seconds for note in notes):
        raise ValueError(f"A note starts after the song's {seconds:g} seconds")
    return tuple(
        dataclasses.replace(note, seconds=min(note.seconds, seconds - note.start))
        for note in notes
    )


@dataclasses.dataclass(frozen=True)
class Song:
    # The file's name without .json, which the URL of its notes has.
    id: str
    title: str
    names: tuple[str, ...]
    notes: tuple[Note, ...] = dataclasses.field(repr=False)
    loop: bool = False
    # How long the song lasts, None until its last note ends.
    length: float | None = None

    @property
    def seconds(self):
        """How long the song plays, once round for a loop."""
        if self.length is not None:
            return self.length
        return max(note.start + note.seconds for note in self.notes)

    def score(self):
        """The notes the client plays, a SongScore."""
        return SongScore(
            seconds=self.seconds,
            loop=self.loop,
            notes=[
                SongNote(
                    start=note.start,
                    seconds=note.seconds,
                    frequency=note.frequency,
                    partials=list(note.partials),
                    volume=note.volume,
                )
                for note in self.notes
            ],
        )


def load_song(path):
    """Reads the song file at the path, raising ValueError for a bad one."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        partials = check_partials(data.get("partials", [1.0]))
        song_notes = tuple(notes(data["voices"], data["tempo"], partials))
        if not song_notes:
            raise ValueError("A song needs a note")
        length = None
        if "beats" in data:
            length = data["beats"] * 60 / data["tempo"]
            song_notes = cut(song_notes, length)
        loop = data.get("loop", False)
        if not isinstance(loop, bool):
            raise TypeError(f"loop is true or false, not {loop!r}")
        names = tuple(data["names"])
        if not names:
            raise ValueError("A song needs a name to ask for it by")
        return Song(
            id=os.path.splitext(os.path.basename(path))[0],
            title=data["title"],
            names=names,
            notes=song_notes,
            loop=loop,
            length=length,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error!r}") from None


def load(directory):
    """Reads the songs, the .json files in the directory."""
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".json")
    )
    return [load_song(path) for path in paths]


class Songs:
    """Answers a request to play a song with the card that plays it."""

    def __init__(self, songs):
        self.songs = {song.id: song for song in songs}
        self.by_name = {}
        for song in songs:
            for name in song.names:
                if self.by_name.setdefault(name, song) is not song:
                    raise ValueError(f"Two songs are called {name!r}")

    @property
    def intent(self):
        """The intent of asking for one of the songs, None without any."""
        if not self.by_name:
            return None
        return Intent(NAME, SAMPLES, {"song": tuple(self.by_name)})

    def score(self, song_id):
        """The song's notes, a SongScore, None for no such song."""
        song = self.songs.get(song_id)
        return song.score() if song is not None else None

    async def card(self, card_id, slots):
        song = self.by_name[slots["song"]]
        return SongCard(
            id=card_id,
            title=song.title,
            url=f"/v1/songs/{song.id}",
            seconds=round(song.seconds, 2),
            loop=song.loop,
        )

    async def close(self):
        pass
