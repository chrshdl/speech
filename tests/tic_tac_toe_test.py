import asyncio
import itertools
import json
import os
import shutil
import subprocess

import pytest

from speech.backend import tic_tac_toe
from speech.intents import IntentRecognizer


def test_intent_fits_the_model_labels():
    labels = "abcdefghijklmnopqrstuvwxyz' "
    char_to_int = {c: i for i, c in enumerate(labels)}
    recognizer = IntentRecognizer([tic_tac_toe.INTENT], char_to_int, len(labels), 0.04)
    assert len(recognizer.samples) == len(tic_tac_toe.INTENT.samples)


def test_card():
    card = asyncio.run(tic_tac_toe.TicTacToe().card("42", {}))
    assert (card.type, card.id, card.title) == ("TicTacToeCard", "42", "Tic Tac Toe")
    assert card.UNTIL_DISMISSED


# Who wins each board, as web/assistant/tic_tac_toe.js finds it.
GAME = os.path.join(
    os.path.dirname(__file__), "..", "web", "assistant", "tic_tac_toe.js"
)
HARNESS = """
const fs = require("node:fs");
const vm = require("node:vm");
const { ticTacToeWinner } = vm.runInNewContext(
  `${fs.readFileSync(process.argv[1], "utf8")}\\n({ ticTacToeWinner })`,
);
const boards = JSON.parse(fs.readFileSync(0, "utf8"));
console.log(JSON.stringify(boards.map((cells) => ticTacToeWinner(cells))));
"""


def lines():
    rows = [[3 * r, 3 * r + 1, 3 * r + 2] for r in range(3)]
    columns = [[c, c + 3, c + 6] for c in range(3)]
    return rows + columns + [[0, 4, 8], [2, 4, 6]]


def test_winner_of_every_board():
    if shutil.which("node") is None:
        pytest.skip("needs Node")
    boards = [list(cells) for cells in itertools.product(["X", "O", None], repeat=9)]
    run = subprocess.run(
        ["node", "-e", HARNESS, GAME],
        input=json.dumps(boards),
        capture_output=True,
        text=True,
        check=True,
    )
    for cells, result in zip(boards, json.loads(run.stdout), strict=True):
        won = [
            line
            for line in lines()
            if cells[line[0]] and len({cells[i] for i in line}) == 1
        ]
        if won:
            # Of two lines, such as of a board played on after a win, the first.
            assert result == {"mark": cells[won[0][0]], "line": won[0]}
        elif all(cells):
            assert result == {"tie": True}
        else:
            assert result is None
