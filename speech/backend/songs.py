"""
Plays songs asked for by name, such as "play twinkle twinkle". Each song is
a JSON file in a directory (backend.py --songs), which the service reads
when it starts and then renders in the background with the sound of an
additive synthesizer, and serves as a WAV file, for the client to play
with the SongCard that answers the request.

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
ends. As every note is let go before it ends, nothing sounds on past the
end, so the loop has no seam.
examples/songs/from_midi.py makes a song file from a MIDI file.
"""

import asyncio
import concurrent.futures
import dataclasses
import io
import json
import logging
import os
import re
import threading
import time
import wave

import numpy as np

from speech.backend.messages import SongCard
from speech.intents import Intent

log = logging.getLogger(__name__)

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

SAMPLE_RATE = 44100
# The synthesizer's voice: each harmonic a sine with two quieter ones
# detuned by 3 Hz plus 1 Hz per harmonic, which thickens it like a chorus,
# up to 8 harmonics, and a linear attack and release.
DETUNE = 0.4
MAX_PARTIALS = 8
ENVELOPE_SECONDS = 0.025
# The synthesizer mixes its notes through tanh with this much headroom per
# unit of master volume, so chords saturate softly rather than clip.
HEADROOM = 1 / (MAX_PARTIALS * 2.5)
# The loudest sample of a rendered song, as loud as the client's tones.
PEAK = 0.3

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


def envelope(samples, gate):
    """
    The synthesizer's envelope of a key held for gate of the samples: up in
    ENVELOPE_SECONDS, held, and down in as long once let go.
    """
    ramp = ENVELOPE_SECONDS * SAMPLE_RATE
    index = np.arange(samples)
    up = np.minimum(index / ramp, 1.0)
    # Released from the level it reached, which is less than 1 for a note
    # shorter than the attack.
    released = np.minimum(gate / ramp, 1.0) - (index - gate) / ramp
    return np.clip(np.where(index < gate, up, released), 0.0, 1.0)


def partial_wave(frequency, partial, t):
    """
    A harmonic of the synthesizer's voice at the times t, in seconds: the
    partial-th multiple of frequency, from 0, with its detuned pair, at
    random phases that are the same each time, so a song renders the same.
    """
    f = frequency * (partial + 1)
    detune = 3 + partial
    phases = np.random.default_rng([round(frequency * 100), partial]).uniform(0, 1, 3)
    wave = sine(f, phases[0], t)
    wave += DETUNE * sine(f + detune, phases[1], t)
    wave += DETUNE * sine(f - detune, phases[2], t)
    return wave / (1 + 2 * DETUNE)


def sine(frequency, phase, t):
    """
    A sine of the frequency at the times t, from the phase in cycles. The
    cycles are wrapped in float64, as a minute's are millions of radians,
    then the sine is taken in float32, which on a Raspberry Pi 4 is 1.4
    times as fast.
    """
    cycles = frequency * t + phase
    cycles -= np.floor(cycles)
    return np.sin((2 * np.pi * cycles).astype(np.float32))


RELEASE_SAMPLES = round(ENVELOPE_SECONDS * SAMPLE_RATE)


def song_samples(notes, seconds=None):
    """
    How many samples the notes take, for seconds or else until the last
    one ends, with the release after it.
    """
    if seconds is None:
        seconds = max(note.start + note.seconds for note in notes)
    return round(seconds * SAMPLE_RATE) + RELEASE_SAMPLES


def render(notes, loop=False, seconds=None):
    """
    The samples of the notes, played with the synthesizer's voice, each
    with its partials and volume, with a peak of PEAK, for seconds or else
    until the last note ends, and then, unless to loop, a release.
    """
    # Each key is let go a release before its note ends, so the next note,
    # also of the same key, starts from silence, as when played by hand.
    release = RELEASE_SAMPLES
    mix = np.zeros(song_samples(notes, seconds))
    for note in notes:
        start = round(note.start * SAMPLE_RATE)
        samples = round(note.seconds * SAMPLE_RATE)
        gate = max(samples - release, samples // 2)
        # The time of the whole song, so a note's phase continues, as the
        # synthesizer's wavetables loop with its clock.
        t = (start + np.arange(samples)) / SAMPLE_RATE
        voice = sum(
            amplitude * partial_wave(note.frequency, partial, t)
            for partial, amplitude in enumerate(note.partials)
            if amplitude > 0
        )
        mix[start : start + samples] += note.volume * envelope(samples, gate) * voice
    if loop:
        mix = mix[:-release]
    mixed = np.tanh(mix * HEADROOM)
    return mixed * (PEAK / np.abs(mixed).max())


def wav(samples):
    """16 bit mono WAV of the samples, in -1 to 1."""
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((samples * 32767).round().astype("<i2").tobytes())
    return out.getvalue()


@dataclasses.dataclass(frozen=True)
class Song:
    # The file's name without .json, which the URL of its audio has.
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
        samples = song_samples(self.notes, self.length)
        return (samples - RELEASE_SAMPLES * self.loop) / SAMPLE_RATE

    def render(self):
        """The song's WAV file."""
        return wav(render(self.notes, self.loop, self.length))


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


def cut(notes, seconds):
    """The notes, cut to end by seconds, raising ValueError for one after it."""
    if any(note.start >= seconds for note in notes):
        raise ValueError(f"A note starts after the song's {seconds:g} seconds")
    return tuple(
        dataclasses.replace(note, seconds=min(note.seconds, seconds - note.start))
        for note in notes
    )


def load(directory):
    """Reads the songs, the .json files in the directory."""
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".json")
    )
    return [load_song(path) for path in paths]


class Songs:
    """
    Answers a request to play a song with the card that plays it, and
    renders the songs on a thread of its own: on a Raspberry Pi 4 a minute
    of song takes about 6 seconds, which the service's start need not wait
    for.
    """

    def __init__(self, songs):
        self.songs = {song.id: song for song in songs}
        self.by_name = {}
        for song in songs:
            for name in song.names:
                if self.by_name.setdefault(name, song) is not song:
                    raise ValueError(f"Two songs are called {name!r}")
        self.rendered = {song.id: concurrent.futures.Future() for song in songs}
        if songs:
            # A daemon, so stopping the service need not wait for it.
            threading.Thread(target=self._render, daemon=True).start()

    def _render(self):
        for song in self.songs.values():
            started = time.monotonic()
            try:
                self.rendered[song.id].set_result(song.render())
            except Exception as error:
                log.exception("Could not render the song %s", song.id)
                self.rendered[song.id].set_exception(error)
            else:
                log.info(
                    "Rendered the song %s in %.1f seconds",
                    song.id,
                    time.monotonic() - started,
                )

    @property
    def intent(self):
        """The intent of asking for one of the songs, None without any."""
        if not self.by_name:
            return None
        return Intent(NAME, SAMPLES, {"song": tuple(self.by_name)})

    async def audio(self, song_id):
        """The song's WAV file, once rendered, None for no such song."""
        rendered = self.rendered.get(song_id)
        return await asyncio.wrap_future(rendered) if rendered is not None else None

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
