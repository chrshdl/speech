import asyncio
import subprocess

import numpy as np
import pytest

from speech.backend import volume
from speech.intents import IntentRecognizer

# The output of `amixer -c usbaudio sget PCM` for a Jabra SPEAK 510.
JABRA = """Simple mixer control 'PCM',0
  Capabilities: pvolume pvolume-joined pswitch pswitch-joined
  Playback channels: Mono
  Limits: Playback 0 - 11
  Mono: Playback 4 [36%] [-20.00dB] [on]
"""


class FakeMixer:
    def __init__(self, value=4, low=0, high=11):
        self.value, self.low, self.high = value, low, high

    def get(self):
        return self.value, self.low, self.high

    def set(self, value):
        self.value = value


def test_turns_a_step_at_a_time():
    mixer = FakeMixer()
    speaker = volume.Volume(mixer)
    assert speaker.turn(1) == 45
    assert mixer.value == 5
    assert speaker.turn(-1) == 36
    # Not to silence, and not past full.
    mixer.value = 1
    assert speaker.turn(-1) == 9
    assert mixer.value == 1
    mixer.value = 11
    assert speaker.turn(1) == 100
    assert mixer.value == 11


def test_a_step_is_a_tenth_of_a_fine_control():
    mixer = FakeMixer(value=100, high=255)
    volume.Volume(mixer).turn(1)
    assert mixer.value == 126


def test_kept_and_set_again(tmp_path):
    state = str(tmp_path / "volume")
    volume.Volume(FakeMixer(), state).turn(1)
    assert (tmp_path / "volume").read_text() == "45"
    mixer = FakeMixer(value=11)
    volume.Volume(mixer, state)
    assert mixer.value == 5


def test_the_card_shows_the_volume():
    mixer = FakeMixer()
    speaker = volume.Volume(mixer)
    card = asyncio.run(speaker.handlers["VolumeDown"].card("abc", {}))
    assert (card.id, card.title, card.percent) == ("abc", "Volume", 27)
    assert mixer.value == 3


def test_mixer_runs_amixer(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, JABRA, "")

    monkeypatch.setattr(volume.subprocess, "run", run)
    mixer = volume.Mixer("usbaudio:PCM")
    assert mixer.get() == (4, 0, 11)
    mixer.set(7)
    assert calls == [
        ["amixer", "-c", "usbaudio", "sget", "PCM"],
        ["amixer", "-c", "usbaudio", "sset", "PCM", "7"],
    ]
    # Without a card, the default card's.
    volume.Mixer("Master").get()
    assert calls[-1] == ["amixer", "sget", "Master"]
    monkeypatch.setattr(
        volume.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "nothing", ""),
    )
    with pytest.raises(ValueError, match="no playback volume"):
        volume.Mixer("Mic").get()


LABELS = "abcdefghijklmnopqrstuvwxyz' "
CHAR_TO_INT = {c: i for i, c in enumerate(LABELS)}
BLANK = len(LABELS)


def spoken(text):
    """Output probabilities that spell the text, with a blank after each label."""
    path = [label for c in text for label in (CHAR_TO_INT[c], BLANK)]
    probs = np.full((len(path), BLANK + 1), 0.03 / BLANK)
    probs[np.arange(len(path)), path] = 0.97
    return probs


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("volume up", "VolumeUp"),
        ("turn it up", "VolumeUp"),
        ("louder", "VolumeUp"),
        ("volume down", "VolumeDown"),
        ("turn down the volume", "VolumeDown"),
        ("quieter", "VolumeDown"),
    ],
)
def test_intents(text, intent):
    recognizer = IntentRecognizer(volume.INTENTS, CHAR_TO_INT, BLANK, 0.04)
    assert recognizer.recognize(spoken(text)).intent.name == intent
