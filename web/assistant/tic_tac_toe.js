// Tic Tac Toe: two players take turns to draw their mark with a finger on a
// board of 3 by 3, X first in the first round, and then who did not start
// the round before. The ink stays as it was drawn, in the player's color,
// and once the mark is done it is drawn to fit its cell. Three in a row win
// the round, which a rainbow strikes through, and the board clears for the
// next, counting the rounds each won, until the board is closed. The game
// only tells when a mark is made and a round ends; app.js plays the tones.

// The board's grid, of cells of TTT_CELL, in the drawing's units.
const TTT_CELL = 300;
const TTT_BOARD = 3 * TTT_CELL;
// The rows, columns and diagonals, by the cells' indices, row by row.
const TTT_LINES = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
];
// The strokes of a mark: an X is done with its second, an O with its first.
// A mark of fewer strokes is done once the player paused MARK_PAUSE_MS
// after lifting the finger, such as an X drawn in one stroke.
const MARK_STROKES = { X: 2, O: 1 };
const MARK_PAUSE_MS = 2000;
// A stroke shorter than this, such as a tap, draws nothing.
const MIN_INK = 0.2 * TTT_CELL;
// The points of a stroke at least this far apart, as it is drawn.
const INK_STEP = 4;
// The ink's width, in the drawing's units.
const INK_WIDTH = 26;
// A mark done is drawn this large in its cell, scaled at most within
// MARK_SCALE, so a small or a large one keeps some of its size, over
// TIDY_MS.
const MARK_SIZE = 0.62 * TTT_CELL;
const MARK_SCALE = [0.6, 1.6];
const TIDY_MS = 260;
// How long a stroke that draws nothing, or lands on a taken cell, fades.
const FADE_MS = 400;
// After a round ends, the next starts after ROUND_END_MS, or at a tap, once
// NEXT_TAP_MS have passed, so the tap that ended it does not.
const ROUND_END_MS = 5000;
const NEXT_TAP_MS = 1200;
// The rainbow strikes through the three, from a little beyond the first to
// a little beyond the last, after STRIKE_MS of which the result shows.
const STRIKE_BEYOND = 0.38 * TTT_CELL;
const STRIKE_MS = 500;
// How long how to play shows, unless a stroke begins sooner.
const HINT_MS = 4000;

// Who won, of the cells, row by row, each "X", "O" or null: the mark and
// the line of three, or a tie when every cell is taken, or else null.
function ticTacToeWinner(cells) {
  for (const line of TTT_LINES) {
    const [a, b, c] = line.map((i) => cells[i]);
    if (a && a === b && a === c) {
      return { mark: a, line };
    }
  }
  return cells.every(Boolean) ? { tie: true } : null;
}

class TicTacToe {
  // The element of the game, with the score, the board and the banner.
  // onMark is called with the player's mark when a mark is done, and onEnd
  // with the result of ticTacToeWinner when a round ends.
  constructor({ root, onMark, onEnd }) {
    this.root = root;
    this.onMark = onMark;
    this.onEnd = onEnd;
    this.ui = {
      board: root.querySelector(".ttt-board"),
      marks: root.querySelector(".ttt-marks"),
      ink: root.querySelector(".ttt-ink-layer"),
      strike: root.querySelector(".ttt-strike"),
      rainbow: root.querySelector("#ttt-rainbow"),
      players: Object.fromEntries(
        [...root.querySelectorAll(".ttt-player")].map((element) => [element.dataset.mark, element]),
      ),
      banner: root.querySelector(".game-banner"),
      title: root.querySelector(".game-banner-title"),
      text: root.querySelector(".game-banner-text"),
    };
    this.score = { X: 0, O: 0 };
    // Who starts the next round.
    this.starter = "X";
    // "playing", "over", after a round, or "gone".
    this.phase = "playing";
    // The stroke being drawn: its pointer, points and element.
    this.stroke = null;
    // The mark being drawn, not yet done: its cell, strokes and timer.
    this.pending = null;
    this.timers = new Set();
    this.frames = new Set();
    // The timer that hides how to play, while it shows.
    this.hint = null;
    // The timers of the round's end: showing the result, and the next round.
    this.ending = [];
    this.listeners = {
      pointerdown: (event) => this.down(event),
      pointermove: (event) => this.move(event),
      pointerup: (event) => this.up(event),
      // On the Raspberry Pi's touch display, WPE WebKit ends each touch on
      // the board with a pointercancel just before its pointerup, so a
      // cancel ends the stroke as lifting does, as in keyboard.js.
      pointercancel: (event) => this.up(event),
      // Touches are drawn as pointers, so the touches themselves do nothing
      // else, such as scroll, or zoom on a double tap.
      touchstart: (event) => event.preventDefault(),
      touchmove: (event) => event.preventDefault(),
      touchend: (event) => event.preventDefault(),
    };
    for (const [name, listener] of Object.entries(this.listeners)) {
      this.ui.board.addEventListener(name, listener, { passive: false });
    }
  }

  // Shows the board, empty, and starts the first round.
  start() {
    this.root.hidden = false;
    document.body.classList.add("tictactoe");
    this.newRound();
    // How to play, until the first stroke, or for a while.
    this.showBanner("Tic Tac Toe!", "Take turns to draw your X or O");
    this.hint = this.later(() => this.hideHint(), HINT_MS);
  }

  // Ends the game at once.
  destroy() {
    if (this.phase === "gone") {
      return;
    }
    this.phase = "gone";
    for (const [name, listener] of Object.entries(this.listeners)) {
      this.ui.board.removeEventListener(name, listener);
    }
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.frames.forEach(cancelAnimationFrame);
    this.frames.clear();
    this.ui.marks.replaceChildren();
    this.ui.ink.replaceChildren();
    this.ui.strike.classList.remove("struck");
    this.hideBanner();
    this.root.hidden = true;
    document.body.classList.remove("tictactoe");
  }

  newRound() {
    this.ending.forEach((timer) => this.cancel(timer));
    this.ending = [];
    this.cells = Array(9).fill(null);
    this.turn = this.starter;
    this.starter = this.starter === "X" ? "O" : "X";
    this.stroke = null;
    this.pending = null;
    this.phase = "playing";
    this.ui.marks.replaceChildren();
    this.ui.ink.replaceChildren();
    this.ui.strike.classList.remove("struck");
    this.hideBanner();
    this.updateHud();
  }

  // Drawing.

  // Where the pointer is on the board, in the drawing's units.
  at(event) {
    const box = this.ui.board.getBoundingClientRect();
    return {
      x: ((event.clientX - box.left) / box.width) * TTT_BOARD,
      y: ((event.clientY - box.top) / box.height) * TTT_BOARD,
    };
  }

  down(event) {
    if (event.pointerType === "mouse" && event.button !== 0) {
      return;
    }
    event.preventDefault();
    if (this.phase === "over") {
      if (performance.now() - this.ended >= NEXT_TAP_MS) {
        this.newRound();
      }
      return;
    }
    // One finger draws at a time.
    if (this.phase !== "playing" || this.stroke) {
      return;
    }
    this.hideHint();
    // The mark's next stroke may take its time.
    if (this.pending) {
      this.cancel(this.pending.timer);
    }
    this.ui.board.setPointerCapture?.(event.pointerId);
    const group = document.createElementNS(SVG_NS, "g");
    group.classList.add("ttt-stroke", this.turn.toLowerCase());
    const path = document.createElementNS(SVG_NS, "path");
    path.classList.add("ttt-ink");
    group.append(path);
    this.ui.ink.append(group);
    const point = this.at(event);
    this.stroke = { id: event.pointerId, points: [point], length: 0, group, path };
    this.drawStroke();
  }

  move(event) {
    const { stroke } = this;
    if (!stroke || event.pointerId !== stroke.id) {
      return;
    }
    // Every point the pointer passed since the last event, where the
    // browser tells them, for a smooth line.
    const events = event.getCoalescedEvents?.() ?? [];
    for (const each of events.length ? events : [event]) {
      const point = this.at(each);
      const last = stroke.points.at(-1);
      const step = Math.hypot(point.x - last.x, point.y - last.y);
      if (step >= INK_STEP) {
        stroke.points.push(point);
        stroke.length += step;
      }
    }
    this.drawStroke();
  }

  up(event) {
    const { stroke } = this;
    if (!stroke || event.pointerId !== stroke.id) {
      return;
    }
    this.stroke = null;
    if (stroke.length < MIN_INK) {
      this.fade(stroke.group);
      this.waitForMark();
      return;
    }
    const cell = this.cellOf(stroke.points);
    if (this.cells[cell]) {
      this.fade(stroke.group, true);
      this.waitForMark();
      return;
    }
    if (this.pending && this.pending.cell !== cell) {
      // Begun anew in another cell.
      this.pending.strokes.forEach(({ group }) => this.fade(group));
      this.pending = null;
    }
    this.pending ??= { cell, strokes: [], timer: null };
    this.pending.strokes.push(stroke);
    if (this.pending.strokes.length >= MARK_STROKES[this.turn]) {
      this.markDone();
    } else {
      this.waitForMark();
    }
  }

  // The mark drawn so far is done once the player paused.
  waitForMark() {
    const { pending } = this;
    if (!pending) {
      return;
    }
    this.cancel(pending.timer);
    pending.timer = this.later(() => {
      if (this.pending === pending && !this.stroke) {
        this.markDone();
      }
    }, MARK_PAUSE_MS);
  }

  // The cell under the middle of the points' box, where a stroke drawn
  // across a line mostly is.
  cellOf(points) {
    const box = bounds(points);
    const column = clampCell(Math.floor((box.x + box.width / 2) / TTT_CELL));
    const row = clampCell(Math.floor((box.y + box.height / 2) / TTT_CELL));
    return row * 3 + column;
  }

  // Smooths the stroke through the midpoints between its points.
  drawStroke() {
    const { points, path } = this.stroke;
    const [first] = points;
    let d = `M${first.x.toFixed(1)} ${first.y.toFixed(1)}`;
    if (points.length === 1) {
      d += "h0.1";
    }
    for (let i = 1; i < points.length - 1; i++) {
      const { x, y } = points[i];
      const next = points[i + 1];
      d += `Q${x.toFixed(1)} ${y.toFixed(1)} ${((x + next.x) / 2).toFixed(1)} ${((y + next.y) / 2).toFixed(1)}`;
    }
    if (points.length > 1) {
      const last = points.at(-1);
      d += `L${last.x.toFixed(1)} ${last.y.toFixed(1)}`;
    }
    path.setAttribute("d", d);
  }

  // Fades the stroke away, shaking it first for a taken cell.
  fade(group, refused = false) {
    group.classList.add(refused ? "refused" : "fading");
    this.later(() => group.remove(), FADE_MS);
  }

  // The mark drawn is the player's, drawn to fit its cell, and the turn
  // passes, unless the round ended.
  markDone() {
    const { cell, strokes, timer } = this.pending;
    this.cancel(timer);
    this.pending = null;
    const mark = this.turn;
    this.cells[cell] = mark;
    // The mark bounces around its cell, for three in a row, and within it
    // the ink is drawn to fit, apart, as the bounce's transform-origin
    // would also move the ink's transform.
    const group = document.createElementNS(SVG_NS, "g");
    group.classList.add("ttt-mark", mark.toLowerCase());
    group.dataset.cell = cell;
    const ink = document.createElementNS(SVG_NS, "g");
    for (const stroke of strokes) {
      stroke.group.classList.remove("ttt-stroke");
      ink.append(stroke.group);
    }
    group.append(ink);
    this.ui.marks.append(group);
    const [x, y] = cellCenter(cell);
    group.style.transformOrigin = `${x}px ${y}px`;
    this.tidy(ink, cell, strokes);
    this.onMark(mark);

    const result = ticTacToeWinner(this.cells);
    if (result) {
      this.endRound(result);
    } else {
      this.turn = mark === "X" ? "O" : "X";
      this.updateHud();
    }
  }

  // Moves and scales the mark into the middle of its cell, keeping the ink
  // as wide as it was drawn.
  tidy(group, cell, strokes) {
    const box = bounds(strokes.flatMap(({ points }) => points));
    const from = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    const [toX, toY] = cellCenter(cell);
    const to = { x: toX, y: toY };
    const size = Math.max(box.width, box.height, 1);
    const scale = Math.min(Math.max(MARK_SIZE / size, MARK_SCALE[0]), MARK_SCALE[1]);
    const start = performance.now();
    const step = (now) => {
      const t = Math.min((now - start) / TIDY_MS, 1);
      const eased = 1 - (1 - t) ** 3;
      const k = 1 + (scale - 1) * eased;
      const x = from.x + (to.x - from.x) * eased;
      const y = from.y + (to.y - from.y) * eased;
      group.setAttribute("transform", `translate(${x} ${y}) scale(${k}) translate(${-from.x} ${-from.y})`);
      group.style.setProperty("--ink", `${INK_WIDTH / k}px`);
      if (t < 1 && this.phase !== "gone") {
        frame = requestAnimationFrame(step);
        this.frames.add(frame);
      }
    };
    let frame = requestAnimationFrame(step);
    this.frames.add(frame);
  }

  // Three in a row are struck through, and bounce, or every cell is taken;
  // the result shows once struck.
  endRound(result) {
    this.phase = "over";
    this.ended = performance.now();
    this.ui.ink.replaceChildren();
    let title = "It's a tie!";
    if (result.mark) {
      this.score[result.mark] += 1;
      title = `${result.mark} wins!`;
      this.strike(result.line);
      for (const cell of result.line) {
        this.ui.marks.querySelector(`[data-cell="${cell}"]`)?.classList.add("won");
      }
    }
    this.updateHud(result.mark);
    this.onEnd(result);
    this.ending = [
      this.later(
        () => this.showBanner(title, "Tap to play again", result.mark ? result.mark.toLowerCase() : "tie"),
        result.mark ? STRIKE_MS : 0,
      ),
      this.later(() => this.newRound(), ROUND_END_MS),
    ];
  }

  strike([first, , last]) {
    const [x1, y1] = cellCenter(first);
    const [x2, y2] = cellCenter(last);
    const length = Math.hypot(x2 - x1, y2 - y1);
    const [ux, uy] = [(x2 - x1) / length, (y2 - y1) / length];
    const ends = {
      x1: x1 - ux * STRIKE_BEYOND,
      y1: y1 - uy * STRIKE_BEYOND,
      x2: x2 + ux * STRIKE_BEYOND,
      y2: y2 + uy * STRIKE_BEYOND,
    };
    for (const [name, value] of Object.entries(ends)) {
      this.ui.strike.setAttribute(name, value);
      // The rainbow runs along the line.
      this.ui.rainbow.setAttribute(name, value);
    }
    this.ui.strike.classList.remove("struck");
    void this.ui.strike.getBoundingClientRect();
    this.ui.strike.classList.add("struck");
  }

  // The score, and whose turn it is, or who just won.
  updateHud(won = null) {
    for (const [mark, element] of Object.entries(this.ui.players)) {
      element.querySelector(".ttt-player-score").textContent = this.score[mark];
      element.classList.toggle("turn", this.phase === "playing" && this.turn === mark);
      element.classList.toggle("won", won === mark);
    }
    this.ui.board.dataset.turn = this.phase === "playing" ? this.turn : "";
  }

  showBanner(title, text, kind = "") {
    this.ui.title.textContent = title;
    this.ui.text.textContent = text;
    this.ui.text.hidden = !text;
    // Restart its animation, also for a banner of the same kind.
    this.ui.banner.className = "game-banner";
    void this.ui.banner.offsetWidth;
    this.ui.banner.className = `game-banner ${kind}`.trim();
    this.ui.banner.hidden = false;
  }

  hideBanner() {
    this.ui.banner.hidden = true;
  }

  // Timers that end with the game. Returns the timer.
  later(callback, ms) {
    const timer = setTimeout(() => {
      this.timers.delete(timer);
      callback();
    }, ms);
    this.timers.add(timer);
    return timer;
  }

  cancel(timer) {
    clearTimeout(timer);
    this.timers.delete(timer);
  }

  hideHint() {
    if (this.hint !== null) {
      this.cancel(this.hint);
      this.hint = null;
      this.hideBanner();
    }
  }
}

// The box around the points.
function bounds(points) {
  const xs = points.map(({ x }) => x);
  const ys = points.map(({ y }) => y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y };
}

// The middle of the cell, in the drawing's units.
function cellCenter(cell) {
  return [((cell % 3) + 0.5) * TTT_CELL, (Math.floor(cell / 3) + 0.5) * TTT_CELL];
}

function clampCell(index) {
  return Math.min(Math.max(index, 0), 2);
}
