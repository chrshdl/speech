import asyncio
import time

import pytest

from speech.backend import brightness
from speech.intents import IntentRecognizer


def backlight(tmp_path, maximum=31, now=31):
    """A backlight's directory, as /sys/class/backlight has, of the Touch Display 2."""
    path = tmp_path / "10-0045"
    path.mkdir()
    (path / "max_brightness").write_text(f"{maximum}\n")
    (path / "brightness").write_text(f"{now}\n")
    return path


def raw(path):
    return int((path / "brightness").read_text())


@pytest.mark.parametrize(
    ("slots", "percent"),
    [
        ({"percent": "fifty"}, 50),
        ({"percent": "one hundred"}, 100),
        ({"percent": "ten"}, 10),
        # Darker would look all dark.
        ({"percent": "zero"}, 10),
        ({"percent": "five"}, 10),
        ({"level": "full"}, 100),
        ({"level": "minimum"}, 10),
    ],
)
def test_percent(slots, percent):
    assert brightness.percent(slots) == percent


@pytest.fixture
def backlights(tmp_path, monkeypatch):
    """A /sys/class/backlight of its own."""
    path = tmp_path / "backlights"
    path.mkdir()
    monkeypatch.setattr(brightness, "BACKLIGHTS", str(path))
    monkeypatch.setattr(brightness, "RETRY_SECONDS", 0.01)
    return path


def test_card_sets_the_backlight(backlights):
    path = backlight(backlights)
    screen = brightness.Brightness("auto")
    card = asyncio.run(screen.card("id", {"percent": "fifty"}))
    assert (card.type, card.id, card.title, card.percent) == (
        "BrightnessCard",
        "id",
        "Brightness",
        50,
    )
    # 50% of 31 steps.
    assert raw(path) == 16
    asyncio.run(screen.card("id", {"percent": "zero"}))
    assert raw(path) == 3


def test_never_off(backlights):
    # With few steps the minimum still lights the screen.
    path = backlight(backlights, maximum=7)
    brightness.Brightness("10-0045").set(brightness.MINIMUM)
    assert raw(path) == 1


def test_kept_and_set_again(backlights, tmp_path):
    path = backlight(backlights)
    state = tmp_path / "brightness"
    screen = brightness.Brightness("auto", str(state))
    asyncio.run(screen.card("id", {"percent": "seventy"}))
    assert state.read_text() == "70"
    # After a restart, as the backlight starts at full brightness.
    (path / "brightness").write_text("31")
    screen = brightness.Brightness("auto", str(state))
    assert screen.restored.wait(2)
    assert raw(path) == 22


def test_kept_brightness_is_clamped(backlights, tmp_path):
    path = backlight(backlights)
    state = tmp_path / "brightness"
    state.write_text("0")
    assert brightness.Brightness("auto", str(state)).restored.wait(2)
    assert raw(path) == 3


def test_set_again_once_the_backlight_appears(backlights, tmp_path):
    # The display's driver can load after the service starts.
    state = tmp_path / "brightness"
    state.write_text("50")
    screen = brightness.Brightness("auto", str(state))
    time.sleep(0.05)
    assert not screen.restored.is_set()
    path = backlight(backlights)
    assert screen.restored.wait(2)
    assert raw(path) == 16


def test_gives_up_without_a_backlight(backlights, tmp_path):
    state = tmp_path / "brightness"
    state.write_text("50")
    screen = brightness.Brightness("auto", str(state), wait=0.05)
    assert screen.restored.wait(2)


def test_the_users_brightness_wins(backlights, tmp_path):
    # Set while the kept one waits for the backlight, it is not replaced.
    state = tmp_path / "brightness"
    state.write_text("50")
    screen = brightness.Brightness("auto", str(state))
    path = backlight(backlights)
    screen.set(90)
    time.sleep(0.05)
    assert raw(path) == 28 and state.read_text() == "90"


def test_find(tmp_path, monkeypatch):
    monkeypatch.setattr(brightness, "BACKLIGHTS", str(tmp_path))
    with pytest.raises(ValueError, match="Found 0"):
        brightness.find()
    path = backlight(tmp_path)
    assert brightness.find() == str(path)
    assert brightness.find("10-0045") == str(path)
    with pytest.raises(ValueError, match="No backlight"):
        brightness.find("other")
    (tmp_path / "other").mkdir()
    with pytest.raises(ValueError, match="Found 2"):
        brightness.find()


def test_intent_fits_the_model_labels():
    labels = "abcdefghijklmnopqrstuvwxyz' "
    char_to_int = {c: i for i, c in enumerate(labels)}
    recognizer = IntentRecognizer([brightness.INTENT], char_to_int, len(labels), 0.04)
    assert len(recognizer.samples) == len(brightness.INTENT.samples)
