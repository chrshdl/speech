import asyncio
import os
import subprocess

import numpy as np
import pytest

from speech.backend import volume
from speech.backend.engine import Busy
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


class Speakerphone:
    """
    A Jabra SPEAK 510 behind ALSA's driver: its keys turn its own volume,
    and the driver answers the value it set or read last, and leaves out
    setting the value it takes the device to have.
    """

    def __init__(self, device=4, low=0, high=11):
        self.device = self.driver = device
        self.low, self.high = low, high
        self.writes = []

    def get(self):
        return self.driver, self.low, self.high

    def set(self, value):
        if value != self.driver:
            self.device = value
            self.writes.append(value)
        self.driver = value

    def press(self, steps):
        self.device = min(max(self.device + steps, self.low), self.high)

    def plug_in(self, device):
        """Plugged in again, at its own volume, which the driver reads."""
        self.device = self.driver = device


def test_turns_a_step_at_a_time():
    mixer = FakeMixer()
    speaker = volume.Volume(mixer)
    # In percent of the 12 levels, as the Jabra's 12 dots.
    assert speaker.turn(1) == 50
    assert mixer.value == 5
    assert speaker.turn(-1) == 42
    # Not to silence, and not past full.
    mixer = FakeMixer(value=1)
    assert volume.Volume(mixer).turn(-1) == 17
    assert mixer.value == 1
    mixer = FakeMixer(value=11)
    assert volume.Volume(mixer).turn(1) == 100
    assert mixer.value == 11


def test_a_step_is_a_tenth_of_a_fine_control():
    mixer = FakeMixer(value=100, high=255)
    volume.Volume(mixer).turn(1)
    assert mixer.value == 126


def test_kept_and_set_again(tmp_path):
    state = str(tmp_path / "volume")
    volume.Volume(FakeMixer(), state).turn(1)
    assert (tmp_path / "volume").read_text() == "50"
    mixer = FakeMixer(value=11)
    volume.Volume(mixer, state)
    assert mixer.value == 5


def test_the_card_shows_the_volume():
    mixer = FakeMixer()
    speaker = volume.Volume(mixer)
    card = asyncio.run(speaker.handlers["VolumeDown"].card("abc", {}))
    assert (card.id, card.title, card.percent) == ("abc", "Volume", 33)
    assert mixer.value == 3
    shown = asyncio.run(speaker.handlers["ShowVolume"].card("def", {}))
    assert shown.percent == 33


def test_follows_the_speakerphones_keys(tmp_path):
    """
    The speakerphone's keys turn its volume themselves, which is counted,
    never set, and asked to turn it after them, it turns from where they
    left it, though the driver's value is stale.
    """
    state = str(tmp_path / "volume")
    phone = Speakerphone(device=5)
    speaker = volume.Volume(phone, state)
    phone.press(1)
    assert speaker.pressed(1) == 58
    assert (phone.device, phone.driver, phone.writes) == (6, 5, [])
    assert (tmp_path / "volume").read_text() == "58"
    # Down to 5, where the driver takes the device to be.
    assert speaker.turn(-1) == 50
    assert phone.device == 5
    # Down to the lowest of the keys, below the lowest asking turns to.
    for _ in range(7):
        phone.press(-1)
        speaker.pressed(-1)
    assert speaker.percent() == 8 and phone.device == 0


def test_set_again_over_a_stale_driver(tmp_path):
    """
    Started again, or plugged in again, the volume kept is set also where
    the driver takes the device to have it, as after the keys turned it
    while the service was not running.
    """
    state = str(tmp_path / "volume")
    (tmp_path / "volume").write_text("50")
    phone = Speakerphone(device=5)
    phone.press(3)
    volume.Volume(phone, state)
    assert phone.device == 5
    speaker = volume.Volume(phone, state)
    phone.plug_in(device=7)
    assert speaker.restore()
    assert phone.device == 5 and speaker.percent() == 50


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


JABRA_LEVELS = volume.SPEAKERPHONE_LEVELS["0b0e:0422"]


class Jabra(Speakerphone):
    """
    A Jabra SPEAK 510, whose level is its dots: a value of the control sets
    it to the value's level, and its keys step it a level.
    """

    def __init__(self, value):
        super().__init__(device=JABRA_LEVELS[value])
        self.driver = value

    def set(self, value):
        if value != self.driver:
            self.device = JABRA_LEVELS[value]
            self.writes.append(value)
        self.driver = value

    def press(self, steps):
        self.device = min(max(self.device + steps, 1), 12)

    def plug_in(self, value):
        self.driver, self.device = value, JABRA_LEVELS[value]


def test_turns_the_jabra_to_the_levels_it_reaches():
    """
    Asked, the volume goes to the next of the levels the control reaches,
    the dots, though the control has values between.
    """
    phone = Jabra(value=0)
    speaker = volume.Volume(phone, levels=JABRA_LEVELS)
    assert speaker.percent() == 8
    ups = []
    for _ in range(8):
        ups.append(speaker.turn(1))
        assert phone.device == speaker.level
    assert [round(up * 12 / 100) for up in ups] == [2, 3, 4, 6, 7, 10, 12, 12]
    downs = []
    for _ in range(8):
        downs.append(speaker.turn(-1))
        assert phone.device == speaker.level
    assert [round(down * 12 / 100) for down in downs] == [10, 7, 6, 4, 3, 2, 1, 1]


def test_turns_the_jabra_from_where_its_keys_left_it():
    phone = Jabra(value=7)
    speaker = volume.Volume(phone, levels=JABRA_LEVELS)
    assert phone.device == speaker.level == 4
    # Up to a level no value reaches.
    phone.press(1)
    assert speaker.pressed(1) == 42
    # Down to the level of the value the driver takes it to have.
    assert speaker.turn(-1) == 33
    assert phone.device == 4
    phone.press(-1)
    phone.press(-1)
    speaker.pressed(-2)
    assert speaker.turn(1) == 25
    assert phone.device == 3


@pytest.mark.parametrize(("kept", "dots"), [(50, 6), (42, 4), (100, 12), (8, 1)])
def test_sets_the_jabra_again(tmp_path, kept, dots):
    """The level kept, or the nearest below it a value reaches."""
    (tmp_path / "volume").write_text(str(kept))
    phone = Jabra(value=9)
    phone.press(-1)
    speaker = volume.Volume(phone, str(tmp_path / "volume"), JABRA_LEVELS)
    assert phone.device == speaker.level == dots


def test_knows_the_jabras_levels(tmp_path):
    (tmp_path / "proc/asound/usbaudio").mkdir(parents=True)
    (tmp_path / "proc/asound/usbaudio/usbid").write_text("0b0e:0422\n")
    (tmp_path / "proc/asound/other").mkdir(parents=True)
    (tmp_path / "proc/asound/other/usbid").write_text("1234:5678\n")
    root = str(tmp_path)
    assert volume.speakerphone_levels("usbaudio", root) == JABRA_LEVELS
    assert volume.speakerphone_levels("other", root) is None
    assert volume.speakerphone_levels("gone", root) is None


def sysfs(root, devices):
    """
    A tree of /proc and /sys under root, as the Pi's: a Jabra's sound card,
    card1, named usbaudio, on USB device 1-1.1, its keys event0, and HDMI's
    keys event4 and event6, from the Pi's /proc/bus/input/devices.
    """
    (root / "proc/asound").mkdir(parents=True)
    (root / "proc/asound/usbaudio").symlink_to("card1")
    (root / "proc/bus/input").mkdir(parents=True)
    (root / "proc/bus/input/devices").write_text(devices)
    usb = root / "sys/devices/usb1/1-1/1-1.1"
    hdmi = root / "sys/devices/platform/hdmi"
    targets = {
        "sound/card1": usb / "1-1.1:1.0",
        "input/event0": usb / "1-1.1:1.3/input/input0",
        "input/event1": usb / "1-1.1:1.3/input/input1",
        "input/event4": hdmi / "input/input4",
        "input/event6": hdmi / "input/input6",
    }
    for link, target in targets.items():
        target.mkdir(parents=True, exist_ok=True)
        (root / "sys/class" / link).mkdir(parents=True)
        (root / "sys/class" / link / "device").symlink_to(target)
    (root / "dev/input").mkdir(parents=True)


def test_finds_the_speakerphones_keys(tmp_path):
    """
    Only the speakerphone's own volume keys, not HDMI's of a TV remote,
    which do not turn its volume.
    """
    with open(os.path.join(os.path.dirname(__file__), "input_devices.txt")) as f:
        sysfs(tmp_path, f.read())
    keys = volume.speakerphone_keys("usbaudio", root=str(tmp_path))
    assert keys == [str(tmp_path / "dev/input/event0")]
    # Unplugged.
    assert volume.speakerphone_keys("other", root=str(tmp_path)) == []


def key(code, value):
    return volume.EVENT.pack(0, 0, volume.EV_KEY, code, value)


def test_keys_show_the_volume(tmp_path, monkeypatch):
    """
    Each press of the speakerphone's volume keys is counted and shows the
    volume, without a card while busy, and once the speakerphone is
    plugged in again, the volume kept is set again.
    """
    monkeypatch.setattr(volume, "RETRY_SECONDS", 0.05)
    fifo = str(tmp_path / "event0")
    os.mkfifo(fifo)
    phone = Speakerphone(device=4)
    speaker = volume.Volume(phone, str(tmp_path / "volume"))
    shown, busy = [], []

    async def answer(name):
        if busy:
            raise Busy("listening")
        shown.append((await speaker.handlers[name].card("id", {})).percent)

    async def connect():
        while True:
            try:
                return os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                await asyncio.sleep(0.01)

    async def run():
        keys = volume.Keys(speaker, answer, find=lambda: [fifo])
        keys.start(asyncio.get_running_loop())
        writer = await connect()
        # Up, released, another key, down: the speakerphone turns itself.
        phone.press(1)
        phone.press(-1)
        os.write(writer, key(115, 1) + key(115, 0) + key(30, 1) + key(114, 1))
        await asyncio.sleep(0.2)
        busy.append(True)
        phone.press(-1)
        os.write(writer, key(114, 1))
        await asyncio.sleep(0.2)
        busy.clear()
        # Unplugged, and plugged in again at its own volume.
        os.close(writer)
        phone.plug_in(device=9)
        writer = await connect()
        await asyncio.sleep(0.3)
        os.close(writer)
        await asyncio.to_thread(keys.stop)

    asyncio.run(run())
    assert shown == [50, 42]
    # The press while busy counts too.
    assert speaker.percent() == 33
    # Set again to the volume kept, from the 9 it came back at.
    assert phone.device == 3
    assert phone.writes[-1] == 3
