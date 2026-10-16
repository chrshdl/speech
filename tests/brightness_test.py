import asyncio

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


def test_card_sets_the_backlight(tmp_path):
    path = backlight(tmp_path)
    screen = brightness.Brightness(str(path))
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


def test_never_off(tmp_path):
    # With few steps the minimum still lights the screen.
    path = backlight(tmp_path, maximum=7)
    brightness.Brightness(str(path)).set(brightness.MINIMUM)
    assert raw(path) == 1


def test_kept_and_set_again(tmp_path):
    path = backlight(tmp_path)
    state = tmp_path / "state" / "brightness"
    state.parent.mkdir()
    screen = brightness.Brightness(str(path), str(state))
    asyncio.run(screen.card("id", {"percent": "seventy"}))
    assert state.read_text() == "70"
    # After a restart, as the backlight starts at full brightness.
    (path / "brightness").write_text("31")
    brightness.Brightness(str(path), str(state))
    assert raw(path) == 22


def test_kept_brightness_is_clamped(tmp_path):
    path = backlight(tmp_path)
    state = tmp_path / "brightness"
    state.write_text("0")
    brightness.Brightness(str(path), str(state))
    assert raw(path) == 3


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
