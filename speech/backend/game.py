"""
Superball, a game steered by voice: asked to "play superball", the service
shows a GameCard, whose client lets flowers fall toward the unicorn and
steers it with the commands of a COMMANDS capture, "left" and "right".
"""

from speech.backend.messages import GameCard
from speech.intents import Intent

TITLE = "Superball"
# The levels as they are said.
LEVELS = ("one", "two", "three", "four", "five")

INTENT = Intent(
    "PlayGame",
    (
        "play superball",
        "play super ball",
        "let's play superball",
        "let's play super ball",
        "start superball",
        "superball",
        "super ball",
        "play superball level {level}",
        "play super ball level {level}",
        "let's play superball level {level}",
    ),
    {"level": LEVELS},
)

# The commands that steer the unicorn, by the names CommandRecognized
# reports.
COMMANDS = [
    Intent("LEFT", ("left", "go left", "move left", "turn left")),
    Intent("RIGHT", ("right", "go right", "move right", "turn right")),
]


def level(said):
    """The level of the word said for it, 1 if none."""
    return LEVELS.index(said) + 1 if said in LEVELS else 1


class Game:
    """Answers a request to play with the card that starts the game."""

    async def card(self, card_id, slots):
        return GameCard(id=card_id, title=TITLE, level=level(slots.get("level")))

    async def close(self):
        pass
