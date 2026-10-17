import asyncio

import httpx
import pytest

from speech.backend import d50, keyboard
from speech.backend.app import create_app
from speech.intents import IntentRecognizer


def test_intent_fits_the_model_labels():
    labels = "abcdefghijklmnopqrstuvwxyz' "
    char_to_int = {c: i for i, c in enumerate(labels)}
    recognizer = IntentRecognizer([keyboard.INTENT], char_to_int, len(labels), 0.04)
    assert len(recognizer.samples) == len(keyboard.INTENT.samples)


def test_card():
    card = asyncio.run(keyboard.Keyboard().card("42", {}))
    assert (card.type, card.id, card.title) == ("KeyboardCard", "42", "Fantasia")
    assert (card.patch, card.url) == ("fantasia", "/v1/patches/fantasia")
    assert card.UNTIL_DISMISSED


def test_no_such_patch():
    with pytest.raises(ValueError, match="No D-50 patch"):
        keyboard.Keyboard("missing")


def test_patch_is_served():
    app = create_app(None)

    async def get(path):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.get(path)

    response = asyncio.run(get("/v1/patches/fantasia"))
    assert response.status_code == 200
    assert response.json() == d50.load("fantasia")
    for path in ("/v1/patches/missing", "/v1/patches/..%2Fsongs"):
        assert asyncio.run(get(path)).status_code == 404
