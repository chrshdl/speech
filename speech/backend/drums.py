"""
The drums: asked to "play the drums", the service shows a DrumsCard, whose
client shows a drum kit to tap, which plays recordings of each drum, see
web/assistant/drums.js.
"""

from speech.backend.messages import DrumsCard
from speech.intents import Intent

INTENT = Intent(
    "PlayDrums",
    (
        "play the drums",
        "play drums",
        "play the drum",
        "play the drum kit",
        "open the drums",
        "the drums",
        "drums",
        "drum kit",
    ),
)


class Drums:
    """Answers a request to play the drums with the card of its kit."""

    async def card(self, card_id, slots):
        return DrumsCard(id=card_id, title="Drums")

    async def close(self):
        pass
