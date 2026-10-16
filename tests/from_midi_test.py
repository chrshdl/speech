import importlib.util
import json
import os
import struct
import subprocess
import sys

from speech.backend import songs

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "examples", "songs", "from_midi.py"
)
spec = importlib.util.spec_from_file_location("from_midi", SCRIPT)
from_midi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(from_midi)

TICKS = 480


def number(value):
    """A MIDI variable-length number."""
    out = [value & 0x7F]
    while value := value >> 7:
        out.insert(0, value & 0x7F | 0x80)
    return bytes(out)


def track(name, notes, running=False):
    """
    A track of the notes, (start, end, key) in beats, with note on and
    note on of velocity 0 for off, optionally in running status.
    """
    events = [(0, b"\xff\x03" + number(len(name)) + name.encode())]
    for start, end, key in notes:
        events.append((start * TICKS, bytes([0x90, key, 80])))
        events.append((end * TICKS, bytes([0x90, key, 0])))
    events.sort(key=lambda e: e[0])
    data, now, status = b"", 0, None
    for ticks, event in events:
        data += number(round(ticks - now))
        now = ticks
        if running and event[0] == status:
            event = event[1:]
        elif event[0] != 0xFF:
            status = event[0]
        data += event
    data += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + data


def midi_file(path):
    tempo = b"\x00\xff\x51\x03" + (500000).to_bytes(3, "big")  # 120 BPM
    meter = b"\x00\xff\x58\x04\x03\x02\x18\x08"  # 3/4
    end = b"\x00\xff\x2f\x00"
    conductor = (
        b"MTrk" + struct.pack(">I", len(tempo + meter + end)) + tempo + meter + end
    )
    melody = track(
        "MELODY",
        # Bar 1: C4 held 1.8 beats, a breath, D4. Bar 2: a rest, E4.
        [(0, 1.8, 60), (2, 3, 62), (4, 6, 64)],
        running=True,
    )
    chords = track("PAD", [(0, 3, 48), (0, 3, 55), (3, 6, 53)])
    header = b"MThd" + struct.pack(">IHHH", 6, 1, 3, TICKS)
    path.write_bytes(header + conductor + melody + chords)
    return path


def test_midi(tmp_path):
    midi = from_midi.Midi(midi_file(tmp_path / "song.mid").read_bytes())
    assert (midi.tempo, midi.beats_per_bar, midi.ticks_per_beat) == (120, 3, TICKS)
    names = [name for name, _ in midi.tracks]
    assert names == ["", "MELODY", "PAD"]
    assert midi.tracks[1][1] == [(0, 864, 60), (960, 1440, 62), (1920, 2880, 64)]


def test_song(tmp_path):
    midi = from_midi.Midi(midi_file(tmp_path / "song.mid").read_bytes())
    song = from_midi.song(
        midi, ["MELODY", "PAD=0.3:1,0.2"], (1, 2), 0.5, None, 2, [1.0]
    )
    assert song["tempo"] == 120
    melody, top, bottom = song["voices"]
    # The breath goes, the rest of bar 2 stays, and all is a tone higher.
    assert melody == [["D4", 2.0], ["E4", 1.0], [None, 1.0], ["F#4", 2.0]]
    # The chord is split into two voices, the top one first, with the pad's
    # sound.
    assert top == {
        "notes": [["A3", 3.0], ["G3", 3.0]],
        "partials": [1.0, 0.2],
        "volume": 0.3,
    }
    assert bottom["notes"] == [["D3", 3.0]]


def test_bars(tmp_path):
    midi = from_midi.Midi(midi_file(tmp_path / "song.mid").read_bytes())
    song = from_midi.song(midi, ["1"], (2, 2), 0.5, 60, 0, [1.0])
    assert (song["tempo"], song["voices"]) == (60, [[[None, 1.0], ["E4", 2.0]]])


def test_writes_a_song_file(tmp_path):
    path = midi_file(tmp_path / "song.mid")
    result = subprocess.run(
        [sys.executable, SCRIPT, path, "--track", "MELODY", "--track", "PAD=0.3"]
        + ["--title", "Test", "--name", "test song", "--tempo", "90"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert (data["title"], data["names"], data["tempo"]) == ("Test", ["test song"], 90)
    assert "beats" not in data and "loop" not in data
    # A note per line.
    assert '      ["C4", 2.0],\n' in result.stdout
    out = tmp_path / "test.json"
    out.write_text(result.stdout)
    song = songs.load_song(out)
    assert len(song.notes) == 6


def test_writes_a_loop_of_bars(tmp_path):
    path = midi_file(tmp_path / "song.mid")
    result = subprocess.run(
        [sys.executable, SCRIPT, path, "--track", "MELODY", "--bars", "1-2", "--loop"]
        + ["--title", "Test", "--name", "test song"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    # Two bars of three beats.
    assert (data["beats"], data["loop"]) == (6, True)
    out = tmp_path / "test.json"
    out.write_text(result.stdout)
    assert songs.load_song(out).seconds == 3


def test_track_sound(tmp_path):
    midi = from_midi.Midi(midi_file(tmp_path / "song.mid").read_bytes())
    song = from_midi.song(midi, ["MELODY=::0.1:0.4"], (1, 2), 0.5, None, 0, [1.0])
    [melody] = song["voices"]
    assert (melody["attack"], melody["release"]) == (0.1, 0.4)
    assert "volume" not in melody and "partials" not in melody
