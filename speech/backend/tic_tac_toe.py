"""
Tic Tac Toe: asked to "play tic tac toe", the service shows a TicTacToeCard,
whose client shows a board of 3 by 3 on which two players take turns to
draw their X or O with a finger, round after round, see
web/assistant/tic_tac_toe.js.
"""

from speech.backend.messages import TicTacToeCard
from speech.intents import Intent

TITLE = "Tic Tac Toe"

INTENT = Intent(
    "PlayTicTacToe",
    (
        "play tic tac toe",
        "let's play tic tac toe",
        "play a game of tic tac toe",
        "start tic tac toe",
        "tic tac toe",
        "the tic tac toe game",
        "tic tac toe game",
        "play noughts and crosses",
        "noughts and crosses",
    ),
)


class TicTacToe:
    """Answers a request to play tic tac toe with the card of its board."""

    async def card(self, card_id, slots):
        return TicTacToeCard(id=card_id, title=TITLE)

    async def close(self):
        pass
