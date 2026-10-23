"""
Turns the speaker up or down when asked, such as "volume up" or "turn it
down", through a mixer control of ALSA, such as a USB speakerphone's PCM,
and shows the volume when the speakerphone's own keys turn it. It never
goes below the lowest step above silence when asked, so an answer can
still be heard, and is kept in a file, so that it is set again when the
service starts, or the speakerphone is plugged in again, as a USB device
can forget it.

A Jabra SPEAK 510 has 12 levels of its own, which its dots show. Its + and
- step through them themselves, and tell the computer only that they were
pressed, as the keys of a keyboard, so ALSA's driver never hears of the
change: it answers the value it set last, and leaves out setting a value
it takes the device to have. And of the 12 values of its control, 4 dB
apart, it rounds each down onto one of its levels, which are not, so the
values reach only 8 of them, see SPEAKERPHONE_LEVELS. So the service keeps
the speakerphone's level, rather than read the control: it counts the
presses, asked turns to the next level a value reaches, and sets a value
the driver takes the device to have by setting another first. It never
sets the volume in answer to a press: set right after one, the Jabra turns
itself further, several levels within 150 ms, reporting each as a press.
"""

import asyncio
import logging
import os
import re
import select
import struct
import subprocess
import threading

from speech.backend.engine import Busy
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
# Shows the volume, after the speakerphone's keys turned it. Never heard,
# so without samples.
SHOW = Intent("ShowVolume", ())

# Linux's input events, of input-event-codes.h: a key's type, the volume
# keys, and an event's layout, its time, type, code and value, 1 for a
# press.
EV_KEY = 1
KEY_VOLUMEDOWN = 114
KEY_VOLUMEUP = 115
EVENT = struct.Struct("llHHi")
# How long to wait before looking for the keys again, such as after the
# speakerphone was unplugged, and for its sound card to come back.
RETRY_SECONDS = 5.0

# The speakerphone's level, from 1, that each value of its volume control
# sets it to, from the lowest, by its USB id: a Jabra SPEAK 510's dots, with
# its 12 values from -36 to +8 dB.
SPEAKERPHONE_LEVELS = {"0b0e:0422": (1, 1, 1, 1, 2, 2, 3, 4, 6, 7, 10, 12)}


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
    def __init__(self, mixer, state=None, levels=None):
        """
        Turns the volume of the mixer, such as a Mixer, whose get() returns
        its value, lowest and highest, and whose set(value) sets it.

        Arguments:
            state (str, optional): The file that keeps the volume in
                percent, which is set again from it now.
            levels (tuple, optional): The device's level, from 1, that each
                value of the control sets it to, from the lowest, such as a
                speakerphone's of SPEAKERPHONE_LEVELS. By default, each value
                is a level.
        """
        self.mixer = mixer
        self.state = state
        self.levels = levels
        self.lock = threading.Lock()
        # The device's level, from 1.
        self.level = None
        self.restore()
        # Answers each of INTENTS, and SHOW.
        self.handlers = {
            UP.name: Step(self, 1),
            DOWN.name: Step(self, -1),
            SHOW.name: Shown(self),
        }

    def restore(self):
        """
        Sets the volume kept again, or else takes the control's value as the
        device's, and returns whether it could. The driver reads the value
        from the device when it starts, such as when the speakerphone was
        plugged in.
        """
        try:
            with self.lock:
                value, low, levels = self._control()
                self.level = levels[value - low]
                kept = self._kept()
                if kept is not None:
                    level = round(kept / 100 * levels[-1])
                    level = _towards(levels, max(level, _floor(levels)), -1)
                    # The driver's value can be stale, after the keys turned
                    # it while the service was not running.
                    self._set(low, levels, level, force=True)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            log.warning("Could not set the volume again: %s", error)
            return False
        if kept is not None:
            log.info("Set the volume to %d%% again", kept)
        return True

    def turn(self, steps):
        """
        Turns the volume by so many steps, up or down, to the levels the
        control can set, and returns it in percent, which it keeps.
        """
        with self.lock:
            _, low, levels = self._control()
            step = max(1, round((levels[-1] - 1) / STEPS))
            level = min(max(self.level + steps * step, _floor(levels)), levels[-1])
            self._set(low, levels, _towards(levels, level, steps))
            return self._keep(levels)

    def pressed(self, steps):
        """
        The speakerphone's keys turned its volume so many levels, up or
        down, themselves. Returns it in percent, which it keeps.
        """
        with self.lock:
            _, _, levels = self._control()
            self.level = min(max(self.level + steps, 1), levels[-1])
            return self._keep(levels)

    def percent(self):
        with self.lock:
            _, _, levels = self._control()
            return _percent(self.level, levels)

    def _control(self):
        """
        The control's value and lowest, and the level of each value, and
        takes the device's level from the value, if not known yet.
        """
        value, low, high = self.mixer.get()
        levels = self.levels
        if levels is None or len(levels) != high - low + 1:
            levels = tuple(range(1, high - low + 2))
        if self.level is None:
            self.level = levels[value - low]
        return value, low, levels

    def _set(self, low, levels, level, force=False):
        """
        Sets the device to the level, which a value of the control sets. The
        driver leaves out the value it takes the device to have, which the
        keys may have turned since, so then, or with force, it is first set
        to another.
        """
        value = low + levels.index(level)
        if force or (self.mixer.get()[0] == value and self.level != level):
            self.mixer.set(value + 1 if value - low + 1 < len(levels) else value - 1)
        self.mixer.set(value)
        self.level = level

    def _kept(self):
        if self.state is None or not os.path.exists(self.state):
            return None
        with open(self.state) as f:
            return int(f.read())

    def _keep(self, levels):
        """Keeps the volume in percent, and returns it."""
        percent = _percent(self.level, levels)
        if self.state is not None:
            # Replaced in one step, so a power cut never leaves half a file.
            with open(self.state + ".tmp", "w") as f:
                f.write(str(percent))
            os.replace(self.state + ".tmp", self.state)
        return percent


def _percent(level, levels):
    """A level in percent of the highest, so a Jabra's 1 dot of 12 is 8%."""
    return round(level / levels[-1] * 100)


def _floor(levels):
    """
    The lowest level asking turns to: the second value's, as the lowest can
    be silence.
    """
    return levels[min(1, len(levels) - 1)]


def _towards(levels, level, direction):
    """
    The level the control can set nearest the level, at or above it going
    up, at or below it going down.
    """
    if direction > 0:
        return min((each for each in levels if each >= level), default=levels[-1])
    return max((each for each in levels if each <= level), default=levels[0])


def speakerphone_levels(card, root="/"):
    """
    The levels of SPEAKERPHONE_LEVELS of the USB device whose sound card is
    named card, such as "usbaudio", or None.
    """
    try:
        with open(os.path.join(root, "proc/asound", card, "usbid")) as f:
            return SPEAKERPHONE_LEVELS.get(f.read().strip())
    except OSError:
        return None


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


class Shown:
    def __init__(self, volume):
        """Shows the volume, as it is."""
        self.volume = volume

    async def card(self, card_id, slots):
        level = await asyncio.to_thread(self.volume.percent)
        return VolumeCard(id=card_id, title="Volume", percent=level)

    async def close(self):
        pass


def _has_volume_keys(devices, event):
    """
    Whether the input device of the event handler, such as event0, has both
    volume keys, in the kernel's list of input devices.
    """
    for block in devices.split("\n\n"):
        handlers = re.search(r"^H: Handlers=(.*)$", block, re.MULTILINE)
        keys = re.search(r"^B: KEY=(.*)$", block, re.MULTILINE)
        if handlers is None or keys is None or event not in handlers.group(1).split():
            continue
        # Words of the bitmap of keys, of an unsigned long each, the highest
        # first.
        bits = 0
        for word in keys.group(1).split():
            bits = (bits << 64) | int(word, 16)
        return bool(bits >> KEY_VOLUMEDOWN & 1 and bits >> KEY_VOLUMEUP & 1)
    return False


def speakerphone_keys(card, root="/"):
    """
    The /dev/input event devices with both volume keys of the USB device
    whose sound card is named card, such as "usbaudio": a speakerphone's own
    keys, which turn its volume themselves, unlike a keyboard's or a TV
    remote's.
    """
    try:
        index = os.readlink(os.path.join(root, "proc/asound", card))
        interface = os.path.realpath(
            os.path.join(root, "sys/class/sound", index, "device")
        )
        with open(os.path.join(root, "proc/bus/input/devices")) as f:
            devices = f.read()
    except OSError:
        # Not plugged in.
        return []
    usb_device = os.path.dirname(interface) + os.sep
    inputs = os.path.join(root, "sys/class/input")
    paths = []
    for event in sorted(os.listdir(inputs)):
        device = os.path.realpath(os.path.join(inputs, event, "device"))
        if (
            event.startswith("event")
            and device.startswith(usb_device)
            and _has_volume_keys(devices, event)
        ):
            paths.append(os.path.join(root, "dev/input", event))
    return paths


class Keys:
    def __init__(self, volume, answer, find):
        """
        Shows and keeps the volume, a Volume, when the speakerphone's own
        volume keys turned it, of the input devices that find() returns,
        such as speakerphone_keys, from start until stop. When they come
        back, such as when the speakerphone was plugged in again, sets the
        volume kept again.

        Arguments:
            answer: Answers an intent by name as if it was heard, and shows
                its card, such as engine.Recognizer.answer, or raises Busy,
                such as while listening, when no card shows.
        """
        self.volume = volume
        self.answer = answer
        self.find = find
        self.stopped = threading.Event()
        self.thread = None
        self.tasks = set()

    def start(self, loop):
        """
        Listens to the keys. The presses are handled on the loop, one after
        the other.
        """
        self.loop = loop
        self.presses = asyncio.Queue()
        self._run(self._handle())
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        for task in list(self.tasks):
            task.cancel()

    async def _handle(self):
        while True:
            steps = await self.presses.get()
            try:
                await self.press(steps)
            except Exception:
                log.exception("Could not follow a volume key")

    async def press(self, steps):
        """A volume key turned the volume so many steps, up or down."""
        await asyncio.to_thread(self.volume.pressed, steps)
        try:
            await self.answer(SHOW.name)
        except Busy:
            pass

    async def plugged_in(self):
        """Sets the volume kept again, once the sound card is back."""
        while not await asyncio.to_thread(self.volume.restore):
            await asyncio.sleep(RETRY_SECONDS)

    def _run(self, coroutine):
        task = asyncio.ensure_future(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self._done)

    def _done(self, task):
        self.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("Could not follow the volume keys", exc_info=task.exception())

    def _listen(self):
        warned = set()
        gone = False
        while not self.stopped.is_set():
            devices = []
            for path in self.find():
                try:
                    devices.append(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
                except OSError as error:
                    if path not in warned:
                        log.warning(
                            "Cannot read the volume keys of %s: %s", path, error
                        )
                        warned.add(path)
            if devices:
                log.info("Listening to the speakerphone's volume keys")
                if gone:
                    self.loop.call_soon_threadsafe(self._run, self.plugged_in())
                try:
                    gone = self._read(devices)
                finally:
                    for fd in devices:
                        os.close(fd)
            self.stopped.wait(RETRY_SECONDS)

    def _read(self, devices):
        """
        Reads the keys until stopped, or a device goes, such as when
        unplugged, and returns whether one went.
        """
        while not self.stopped.is_set():
            ready, _, _ = select.select(devices, [], [], 0.5)
            for fd in ready:
                try:
                    data = os.read(fd, EVENT.size * 64)
                except OSError:
                    data = b""
                if not data:
                    return True
                for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
                    _, _, kind, code, value = EVENT.unpack_from(data, i)
                    steps = {KEY_VOLUMEUP: 1, KEY_VOLUMEDOWN: -1}.get(code)
                    if kind == EV_KEY and value == 1 and steps is not None:
                        self.loop.call_soon_threadsafe(self.presses.put_nowait, steps)
        return False
