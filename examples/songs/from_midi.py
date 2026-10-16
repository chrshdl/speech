"""
Makes a song file for speech/backend/songs.py from tracks of a standard
MIDI file, such as one exported from a score:

    python examples/songs/from_midi.py score.mid --track MELODY \\
        --track "WARM PAD=0.35:1,0.15" --bars 5-20 --tempo 72 \\
        --title "My Song" --name "my song" > my-song.json

Each --track is a track's name, or its index from 0, optionally with a
volume after = and partials after :, for a voice of its own sound. The
notes are moved to a grid of --grid beats, and notes of a track that sound
together, such as a chord's, are split into as many voices as it takes.
With --bars, the song lasts exactly those bars, and --loop plays it over
and over.
"""

import argparse
import json
import struct
import sys

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


class Midi:
    def __init__(self, data):
        """Reads a standard MIDI file, of format 0 or 1."""
        if data[:4] != b"MThd":
            raise ValueError("Not a MIDI file")
        length, _, count, division = struct.unpack(">IHHH", data[4:14])
        if division & 0x8000:
            raise ValueError("Time in SMPTE frames is not supported")
        self.ticks_per_beat = division
        self.tempo = 120.0
        self.beats_per_bar = 4
        self.tracks = []
        position = 8 + length
        for _ in range(count):
            kind, size = struct.unpack(">4sI", data[position : position + 8])
            position += 8
            if kind == b"MTrk":
                self.tracks.append(self._track(data[position : position + size]))
            position += size

    def _track(self, data):
        """The track's name and notes, as (start, end, key) in ticks."""
        name, notes, held = "", [], {}
        position, ticks, status = 0, 0, 0

        def number():
            nonlocal position
            value = 0
            while True:
                byte = data[position]
                position += 1
                value = value << 7 | byte & 0x7F
                if not byte & 0x80:
                    return value

        while position < len(data):
            ticks += number()
            if data[position] & 0x80:
                status = data[position]
                position += 1
            if status == 0xFF:
                kind = data[position]
                position += 1
                size = number()
                body = data[position : position + size]
                position += size
                if kind == 0x03:
                    name = body.decode("latin-1")
                elif kind == 0x51 and not self.tracks:
                    self.tempo = 60e6 / int.from_bytes(body, "big")
                elif kind == 0x58 and not self.tracks:
                    self.beats_per_bar = body[0] * 4 / 2 ** body[1]
            elif status in (0xF0, 0xF7):
                position += number()
            else:
                kind = status & 0xF0
                size = 1 if kind in (0xC0, 0xD0) else 2
                key, velocity = data[position], data[position + 1]
                position += size
                if kind == 0x90 and velocity:
                    held.setdefault(key, []).append(ticks)
                elif kind in (0x80, 0x90) and held.get(key):
                    notes.append((held[key].pop(0), ticks, key))
        return name, sorted(notes)


def pitch(key):
    return f"{NAMES[key % 12]}{key // 12 - 1}"


def voices(notes, grid):
    """
    The notes, as (start, end, key) in beats, split into voices whose notes
    follow each other, each a list of [pitch or None, beats]. A gap shorter
    than the grid between notes of a voice is phrasing, as when a player
    breathes, so the note lasts until the next.
    """
    lines = []
    for start, end, key in sorted(notes, key=lambda n: (n[0], -n[2])):
        line = next((line for line in lines if line[-1][1] <= start + grid / 2), None)
        if line is None:
            line = []
            lines.append(line)
        line.append((start, end, key))
    out = []
    for line in lines:
        voice, time = [], 0.0
        for i, (start, end, key) in enumerate(line):
            start = max(round(start / grid) * grid, time)
            if i + 1 < len(line) and line[i + 1][0] - end < grid:
                end = line[i + 1][0]
            end = max(round(end / grid) * grid, start + grid)
            if start > time:
                voice.append([None, start - time])
            voice.append([pitch(key), end - start])
            time = end
        out.append(voice)
    return out


def song(midi, tracks, bars, grid, tempo, transpose, partials):
    first, last = bars
    beats_per_bar = midi.beats_per_bar
    begin, end = (first - 1) * beats_per_bar, last * beats_per_bar
    by_name = {name: i for i, (name, _) in enumerate(midi.tracks)}
    out = []
    for spec in tracks:
        name, _, sound = spec.partition("=")
        volume, _, track_partials = sound.partition(":")
        index = by_name[name] if name in by_name else int(name)
        _, notes = midi.tracks[index]
        notes = [
            (
                s / midi.ticks_per_beat - begin,
                e / midi.ticks_per_beat - begin,
                k + transpose,
            )
            for s, e, k in notes
            if begin <= s / midi.ticks_per_beat < end
        ]
        if not notes:
            raise ValueError(f"Track {name} has no notes in bars {first}-{last}")
        for voice in voices(notes, grid):
            if volume or track_partials:
                voice = {"notes": voice}
                if track_partials:
                    voice["partials"] = [float(p) for p in track_partials.split(",")]
                if volume:
                    voice["volume"] = float(volume)
            out.append(voice)
    return {
        "tempo": tempo or round(midi.tempo, 2),
        "partials": partials,
        "voices": out,
    }


def main(args):
    with open(args.midi, "rb") as f:
        midi = Midi(f.read())
    first, _, last = args.bars.partition("-") if args.bars else ("1", "", "100000")
    first, last = int(first), int(last or first)
    played = song(
        midi,
        args.track,
        (first, last),
        args.grid,
        args.tempo,
        args.transpose,
        [float(p) for p in args.partials.split(",")],
    )
    data = {"title": args.title, "names": args.name, "tempo": played["tempo"]}
    if args.bars:
        # The bars, also a rest at their end, so a loop keeps to the beat.
        data["beats"] = (last - first + 1) * midi.beats_per_bar
    if args.loop:
        data["loop"] = True
    data |= played
    print(dumps(data))


def dumps(data):
    """The song as JSON, a note per line."""
    lines = ["{"]
    lines += [
        f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in data.items() if k != "voices"
    ]
    lines.append('  "voices": [')
    for i, voice in enumerate(data["voices"]):
        notes = voice["notes"] if isinstance(voice, dict) else voice
        body = ",\n".join(f"      {json.dumps(note)}" for note in notes)
        if isinstance(voice, dict):
            sound = "".join(
                f"{json.dumps(k)}: {json.dumps(v)}, "
                for k, v in voice.items()
                if k != "notes"
            )
            text = f'    {{{sound}"notes": [\n{body}\n    ]}}'
        else:
            text = f"    [\n{body}\n    ]"
        lines.append(text + ("," if i < len(data["voices"]) - 1 else ""))
    lines += ["  ]", "}"]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("midi", help="The MIDI file.")
    parser.add_argument(
        "--track",
        action="append",
        required=True,
        help="A track to play, by name or index, as NAME, NAME=VOLUME or "
        "NAME=VOLUME:PARTIALS, such as 'WARM PAD=0.35:1,0.15'.",
    )
    parser.add_argument("--bars", help="The bars to play, such as 5-20, from 1.")
    parser.add_argument("--grid", type=float, default=0.5, help="In beats.")
    parser.add_argument(
        "--tempo", type=float, help="Beats per minute, the file's by default."
    )
    parser.add_argument("--transpose", type=int, default=0, help="In semitones.")
    parser.add_argument(
        "--partials", default="1", help="The song's, such as 1,0.4,0.2."
    )
    parser.add_argument(
        "--loop", action="store_true", help="Play it over and over until closed."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--name", action="append", required=True, help="How it is asked for."
    )
    main(parser.parse_args(sys.argv[1:]))
