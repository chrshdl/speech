import asyncio
import string
import types

import pytest

from speech.backend import game
from speech.intents import IntentRecognizer

CHAR_TO_INT = {c: i for i, c in enumerate(" '" + string.ascii_lowercase)}


@pytest.mark.parametrize(
    ("said", "level"),
    [(None, 1), ("one", 1), ("three", 3), ("five", 5)],
)
def test_level(said, level):
    assert game.level(said) == level


def test_card():
    card = asyncio.run(game.Game().card("id", {"level": "two"}))
    assert (card.type, card.id, card.title, card.level) == (
        "GameCard",
        "id",
        "Superball",
        2,
    )
    assert card.UNTIL_DISMISSED


def test_intents_use_the_model_labels():
    # A model's labels: letters, the space and the apostrophe.
    model = types.SimpleNamespace(
        char_to_int=CHAR_TO_INT, blank=len(CHAR_TO_INT), frame_seconds=0.04
    )
    IntentRecognizer([game.INTENT], model.char_to_int, model.blank, 0.04)
    commands = game.commands(model)
    assert commands.threshold == game.THRESHOLD
