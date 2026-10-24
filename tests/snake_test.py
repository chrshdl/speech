import asyncio
import itertools
import json
import os
import shutil
import subprocess

import pytest

from speech.backend import game, snake
from speech.intents import IntentRecognizer


def test_intent_fits_the_model_labels():
    labels = "abcdefghijklmnopqrstuvwxyz' "
    char_to_int = {c: i for i, c in enumerate(labels)}
    recognizer = IntentRecognizer([snake.INTENT], char_to_int, len(labels), 0.04)
    assert len(recognizer.samples) == len(snake.INTENT.samples)


def test_card():
    card = asyncio.run(snake.Snake().card("42", {}))
    assert (card.type, card.id, card.title, card.level) == (
        "SnakeCard",
        "42",
        "Snake",
        1,
    )
    assert card.UNTIL_DISMISSED


def test_card_at_level():
    card = asyncio.run(snake.Snake().card("42", {"level": "three"}))
    assert card.level == 3


def test_commands_steer_the_snake():
    names = {intent.name for intent in game.COMMANDS}
    assert names == {"LEFT", "RIGHT", "UP", "DOWN"}
    # A command is never among the other words it competes with.
    samples = {sample for intent in game.COMMANDS for sample in intent.samples}
    assert not samples & set(game.OTHERS)


# The snake's mazes, as web/assistant/snake.js lays them, see snake_harness.mjs.
HARNESS = os.path.join(os.path.dirname(__file__), "snake_harness.mjs")


@pytest.fixture(scope="module")
def gardens():
    if shutil.which("node") is None:
        pytest.skip("needs Node")
    run = subprocess.run(["node", HARNESS], capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def reachable(start, walls, columns, rows):
    """The cells reachable from start, also across the sides."""
    seen = {start}
    todo = [start]
    while todo:
        x, y = todo.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = ((x + dx) % columns, (y + dy) % rows)
            if cell not in walls and cell not in seen:
                seen.add(cell)
                todo.append(cell)
    return seen


def test_mazes_leave_every_cell_reachable(gardens):
    for garden in gardens:
        columns, rows = garden["columns"], garden["rows"]
        for walls in garden["levels"]:
            walls = {tuple(cell) for cell in walls}
            free = {(x, y) for x in range(columns) for y in range(rows)} - walls
            head = tuple(garden["start"][0])
            assert reachable(head, walls, columns, rows) == free


def test_mazes_grow(gardens):
    for garden in gardens:
        levels = [{tuple(cell) for cell in walls} for walls in garden["levels"]]
        assert not levels[0]
        for before, after in itertools.pairwise(levels):
            assert before < after


def test_snake_starts_clear_of_walls(gardens):
    # It lies on free cells, heading right, with room to turn ahead.
    for garden in gardens:
        columns = garden["columns"]
        (x, y), *_ = garden["start"]
        clear = [*map(tuple, garden["start"])]
        clear += [((x + i) % columns, y) for i in range(1, 4)]
        for walls in garden["levels"]:
            assert not set(clear) & {tuple(cell) for cell in walls}
