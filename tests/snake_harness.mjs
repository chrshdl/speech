// Prints the walls of web/assistant/snake.js's mazes, for tests/snake_test.py:
//
//   node snake_harness.mjs
//
// prints, as JSON, for each garden, its columns and rows, the snake's start
// as its game lays it, head first, and the walls of each level.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const file = path.join(path.dirname(new URL(import.meta.url).pathname), "..", "web", "assistant", "snake.js");
// Only the mazes run, so the page's globals are not needed.
const game = vm.runInNewContext(
  `${fs.readFileSync(file, "utf8")}\n({ mazeWalls, MAZES, WIDE_GARDEN, TALL_GARDEN, START_LENGTH })`,
);
const gardens = [game.WIDE_GARDEN, game.TALL_GARDEN].map(({ columns, rows }) => {
  const x = Math.floor(columns / 2);
  const y = Math.floor(rows / 2);
  return {
    columns,
    rows,
    start: Array.from({ length: game.START_LENGTH }, (_, i) => [x - i, y]),
    levels: game.MAZES.map((_, i) => game.mazeWalls(i + 1, columns, rows).map(({ x, y }) => [x, y])),
  };
});
console.log(JSON.stringify(gardens));
