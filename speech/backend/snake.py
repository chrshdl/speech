"""
Snake, a game steered by voice: asked to "play snake", the service shows a
SnakeCard, whose client lets the snake crawl to the apples, growing with
each it eats, with more walls at each level, and steers it with the
commands of a COMMANDS capture, "up", "down", "left" and "right", see
game.py.
"""

from speech.backend.game import LEVELS, level
from speech.backend.messages import SnakeCard
from speech.intents import Intent

TITLE = "Snake"

INTENT = Intent(
    "PlaySnake",
    (
        "play snake",
        "play the snake",
        "play the snake game",
        "let's play snake",
        "start snake",
        "snake",
        "the snake game",
        "snake game",
        "play snake level {level}",
        "let's play snake level {level}",
        "play the snake game level {level}",
    ),
    {"level": LEVELS},
)


class Snake:
    """Answers a request to play snake with the card that starts the game."""

    async def card(self, card_id, slots):
        return SnakeCard(id=card_id, title=TITLE, level=level(slots.get("level")))

    async def close(self):
        pass
