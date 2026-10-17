"""
The synthesizer's keyboard: asked to "play the synthesizer", the service
shows a KeyboardCard, whose client shows a keyboard that plays a Roland
D-50 patch, Fantasia, from GET /v1/patches/{patch}, see
web/assistant/keyboard.js and web/assistant/synth.js.
"""

from speech.backend import d50
from speech.backend.messages import KeyboardCard
from speech.intents import Intent

# The patch the keyboard plays, of speech/backend/patches.
PATCH = "fantasia"

INTENT = Intent(
    "PlayKeyboard",
    (
        "play the synthesizer",
        "play synthesizer",
        "open the synthesizer",
        "the synthesizer",
        "synthesizer",
        "play the synth",
        "play the keyboard",
        "play keyboard",
        "open the keyboard",
        "keyboard",
    ),
)


class Keyboard:
    """Answers a request to play the synthesizer with the card of its keyboard."""

    def __init__(self, patch=PATCH):
        # Read now, so a bad patch fails when the service starts.
        self.patch = patch
        self.title = d50.load(patch)["name"]

    async def card(self, card_id, slots):
        return KeyboardCard(
            id=card_id,
            title=self.title,
            patch=self.patch,
            url=f"/v1/patches/{self.patch}",
        )

    async def close(self):
        pass
