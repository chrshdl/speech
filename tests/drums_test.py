import asyncio

from speech.backend import drums
from speech.intents import IntentRecognizer


def test_intent_fits_the_model_labels():
    labels = "abcdefghijklmnopqrstuvwxyz' "
    char_to_int = {c: i for i, c in enumerate(labels)}
    recognizer = IntentRecognizer([drums.INTENT], char_to_int, len(labels), 0.04)
    assert len(recognizer.samples) == len(drums.INTENT.samples)


def test_card():
    card = asyncio.run(drums.Drums().card("42", {}))
    assert (card.type, card.id, card.title) == ("DrumsCard", "42", "Drums")
    assert card.UNTIL_DISMISSED
