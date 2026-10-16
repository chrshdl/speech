"""
Sets the brightness of the screen when asked, such as "set the brightness
to fifty percent", through its backlight in /sys/class/backlight. It never
goes below MINIMUM, at which the screen is still readable, and is kept in a
file, so that it is set again when the service starts.
"""

import os

from speech.backend.messages import BrightnessCard
from speech.intents import Intent

BACKLIGHTS = "/sys/class/backlight"
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
    """Answers a request to set the brightness with the card that shows it."""

    def __init__(self, backlight, state=None):
        """
        Arguments:
            backlight (str): A backlight's directory in /sys/class/backlight.
            state (str, optional): The file that keeps the brightness, which
                is set again from it now.
        """
        self.backlight = backlight
        self.state = state
        with open(os.path.join(backlight, "max_brightness")) as f:
            self.max = int(f.read())
        if state is not None and os.path.exists(state):
            with open(state) as f:
                self.set(max(min(int(f.read()), 100), MINIMUM))

    def set(self, level):
        """Sets the brightness to level percent, and keeps it."""
        # At least 1, as 0 turns some backlights off.
        raw = max(round(level / 100 * self.max), 1)
        with open(os.path.join(self.backlight, "brightness"), "w") as f:
            f.write(str(raw))
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
