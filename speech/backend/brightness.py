"""
Sets the brightness of the screen when asked, such as "set the brightness
to fifty percent", through its backlight in /sys/class/backlight. It never
goes below MINIMUM, at which the screen is still readable, and is kept in a
file, so that it is set again when the service starts, also when the
backlight appears only after it.
"""

import logging
import os
import threading
import time

from speech.backend.messages import BrightnessCard
from speech.intents import Intent

log = logging.getLogger(__name__)

BACKLIGHTS = "/sys/class/backlight"
# How long the kept brightness is tried to be set again, while the
# backlight is not there or not writable yet, and how often.
RESTORE_SECONDS = 60.0
RETRY_SECONDS = 1.0
# The lowest brightness in percent, below which the screen looks dark.
MINIMUM = 10

NUMBERS = {
    "zero": 0,
    "five": 5,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "one hundred": 100,
    "a hundred": 100,
}
LEVELS = {
    "full": 100,
    "maximum": 100,
    "max": 100,
    "highest": 100,
    "minimum": MINIMUM,
    "lowest": MINIMUM,
}

INTENT = Intent(
    "SetBrightness",
    (
        "set the brightness to {percent} percent",
        "set brightness to {percent} percent",
        "set the brightness to {percent}",
        "set the screen brightness to {percent} percent",
        "change the brightness to {percent} percent",
        "make the brightness {percent} percent",
        "brightness to {percent} percent",
        "brightness {percent} percent",
        "{level} brightness",
        "set the brightness to {level}",
        "set brightness to {level}",
        "brightness to {level}",
    ),
    {"percent": tuple(NUMBERS), "level": tuple(LEVELS)},
)


def find(name="auto"):
    """
    The directory of the backlight called name in /sys/class/backlight, or
    with "auto", of the only one, raising ValueError for none.
    """
    if name != "auto":
        path = os.path.join(BACKLIGHTS, name)
        if not os.path.isdir(path):
            raise ValueError(f"No backlight {name} in {BACKLIGHTS}")
        return path
    names = sorted(os.listdir(BACKLIGHTS)) if os.path.isdir(BACKLIGHTS) else []
    if len(names) != 1:
        raise ValueError(f"Found {len(names)} backlights in {BACKLIGHTS}, name one")
    return os.path.join(BACKLIGHTS, names[0])


def percent(slots):
    """The brightness in percent that the slots ask for, MINIMUM at least."""
    if "level" in slots:
        asked = LEVELS[slots["level"]]
    else:
        asked = NUMBERS[slots["percent"]]
    return max(asked, MINIMUM)


class Brightness:
    """
    Answers a request to set the brightness with the card that shows it.
    The backlight can appear after the service starts, as the display's
    driver loads, and only then be made writable, so it is looked up each
    time it is set, and the kept brightness is set again as soon as it can
    be, unless the user sets one first.
    """

    def __init__(self, name="auto", state=None, wait=RESTORE_SECONDS):
        """
        Arguments:
            name (str): The backlight in /sys/class/backlight, or "auto"
                for the only one, see find().
            state (str, optional): The file that keeps the brightness, which
                is set again from it, trying for up to wait seconds.
        """
        self.name = name
        self.state = state
        self.lock = threading.Lock()
        # Set once the kept brightness was set again, given up on, or
        # replaced by the user's.
        self.restored = threading.Event()
        level = None
        if state is not None and os.path.exists(state):
            with open(state) as f:
                level = max(min(int(f.read()), 100), MINIMUM)
        if level is None:
            self.restored.set()
        else:
            # A daemon, so stopping the service need not wait for it.
            threading.Thread(
                target=self._restore, args=(level, wait), daemon=True
            ).start()

    def _restore(self, level, wait):
        deadline = time.monotonic() + wait
        while True:
            with self.lock:
                if self.restored.is_set():
                    return
                try:
                    self._write(level)
                except (OSError, ValueError) as error:
                    if time.monotonic() >= deadline:
                        log.warning("Could not set the brightness again: %s", error)
                        self.restored.set()
                        return
                else:
                    log.info("Set the brightness to %d%% again", level)
                    self.restored.set()
                    return
            time.sleep(RETRY_SECONDS)

    def _write(self, level):
        backlight = find(self.name)
        with open(os.path.join(backlight, "max_brightness")) as f:
            maximum = int(f.read())
        # At least 1, as 0 turns some backlights off.
        raw = max(round(level / 100 * maximum), 1)
        with open(os.path.join(backlight, "brightness"), "w") as f:
            f.write(str(raw))

    def set(self, level):
        """Sets the brightness to level percent, and keeps it."""
        with self.lock:
            self._write(level)
            # The user's brightness, rather than the one kept.
            self.restored.set()
        if self.state is not None:
            # Replaced in one step, so a power cut never leaves half a file.
            with open(self.state + ".tmp", "w") as f:
                f.write(str(level))
            os.replace(self.state + ".tmp", self.state)

    async def card(self, card_id, slots):
        level = percent(slots)
        self.set(level)
        return BrightnessCard(id=card_id, title="Brightness", percent=level)

    async def close(self):
        pass
