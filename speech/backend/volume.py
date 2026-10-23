"""
Turns the speaker up or down when asked, such as "volume up" or "turn it
down", through a mixer control of ALSA, such as the PCM of a USB
speakerphone, which its own buttons also turn. It never goes below the
lowest step above silence, so an answer can still be heard, and is kept in
a file, so that it is set again when the service starts, as a USB device
can forget it.
"""

import asyncio
import logging
import os
import re
import subprocess
import threading

from speech.backend.messages import VolumeCard
from speech.intents import Intent

log = logging.getLogger(__name__)

# The steps of the volume, at most, from silence to full.
STEPS = 10

UP = Intent(
    "VolumeUp",
    (
        "volume up",
        "turn the volume up",
        "turn up the volume",
        "turn it up",
        "increase the volume",
        "raise the volume",
        "louder",
        "make it louder",
        "speak louder",
    ),
)
DOWN = Intent(
    "VolumeDown",
    (
        "volume down",
        "turn the volume down",
        "turn down the volume",
        "turn it down",
        "decrease the volume",
        "lower the volume",
        "quieter",
        "softer",
        "make it quieter",
        "speak softer",
    ),
)
INTENTS = (UP, DOWN)


class Mixer:
    def __init__(self, control):
        """
        A mixer control of ALSA, through amixer.

        Arguments:
            control (str): The card and the control, such as "usbaudio:PCM",
                or only the control, of the default card.
        """
        card, _, self.name = control.rpartition(":")
        self.card = ["-c", card] if card else []

    def get(self):
        """The control's value and its lowest and highest."""
        out = self._run("sget")
        limits = re.search(r"Limits:.*?(-?\d+) - (-?\d+)", out)
        value = re.search(r"Playback (-?\d+) \[", out)
        if limits is None or value is None:
            raise ValueError(f"{self.name} has no playback volume")
        return int(value.group(1)), int(limits.group(1)), int(limits.group(2))

    def set(self, value):
        self._run("sset", str(value))

    def _run(self, *args):
        return subprocess.run(
            ["amixer", *self.card, *args[:1], self.name, *args[1:]],
            check=True,
            capture_output=True,
            text=True,
        ).stdout


class Volume:
    def __init__(self, mixer, state=None):
        """
        Turns the volume of the mixer, such as a Mixer, whose get() returns
        its value, lowest and highest, and whose set(value) sets it.

        Arguments:
            state (str, optional): The file that keeps the volume in
                percent, which is set again from it now.
        """
        self.mixer = mixer
        self.state = state
        self.lock = threading.Lock()
        if state is not None and os.path.exists(state):
            with open(state) as f:
                level = int(f.read())
            try:
                with self.lock:
                    self._set_percent(level)
                log.info("Set the volume to %d%% again", level)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                log.warning("Could not set the volume again: %s", error)
        # Answers each of INTENTS.
        self.handlers = {UP.name: Step(self, 1), DOWN.name: Step(self, -1)}

    def _set_percent(self, level):
        _, low, high = self.mixer.get()
        self.mixer.set(max(low + round(level / 100 * (high - low)), low + 1))

    def turn(self, steps):
        """
        Turns the volume by so many steps, up or down, and returns it in
        percent, which it keeps.
        """
        with self.lock:
            value, low, high = self.mixer.get()
            step = max(1, round((high - low) / STEPS))
            value = min(max(value + steps * step, low + 1), high)
            self.mixer.set(value)
        level = round((value - low) / (high - low) * 100)
        if self.state is not None:
            # Replaced in one step, so a power cut never leaves half a file.
            with open(self.state + ".tmp", "w") as f:
                f.write(str(level))
            os.replace(self.state + ".tmp", self.state)
        return level


class Step:
    def __init__(self, volume, steps):
        """Answers a request to turn the volume by so many steps."""
        self.volume = volume
        self.steps = steps

    async def card(self, card_id, slots):
        level = await asyncio.to_thread(self.volume.turn, self.steps)
        return VolumeCard(id=card_id, title="Volume", percent=level)

    async def close(self):
        pass
