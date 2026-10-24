// Snake: the snake crawls through the garden a cell at a time, and the player
// turns it by voice, "up", "down", "left" and "right", or with the arrow keys
// or buttons, to the apple. Each apple it eats makes it a bead longer, and
// every 3 apples the next level adds walls, as the Nokia's Snake II's mazes.
// It crawls out of one side of the garden and back in at the other, so the
// game ends only when it bites itself or bumps into a wall, and then shows
// the apples it ate.

// The garden's cells, across and down, for wide and for tall screens. The
// game takes the one that draws the cells larger.
const WIDE_GARDEN = { columns: 13, rows: 9 };
const TALL_GARDEN = { columns: 9, rows: 12 };
// A cell's size in the garden's drawing.
const CELL = 100;
// The seconds the snake takes to crawl a cell. Saying a command and seeing
// the snake turn takes about two of them, see COMMAND_SECONDS in game.js.
const CRAWL_SECONDS = 0.6;
const START_LENGTH = 3;
// Where each command turns the snake.
const TURNS = { UP: [0, -1], DOWN: [0, 1], LEFT: [-1, 0], RIGHT: [1, 0] };
// The turns said ahead, which the snake takes one a cell, so two said
// quickly, such as "up" and "left", both turn it.
const AHEAD = 2;
// A turn heard in the first part of the crawl to the next cell turns the
// snake at once, at the cell it is leaving, as if it had come in time, and
// its head swerves to the new way over SWERVE_MS. A command is heard about
// 0.4 seconds after it was said, and the snake would otherwise wait for the
// next cell, up to a crawl longer. Heard later, it turns at the next cell.
const TURN_AT_ONCE = 0.5;
const SWERVE_MS = 120;
// The beads' colors, from behind the head, the unicorn mane's.
const BEADS = ["#ffa94d", "#ffd93d", "#6bd88f", "#5cc0ff", "#a98bff", "#ff7ab8"];
// The radius of the beads, from behind the head to the tail, in the
// drawing's units. The head, #snake-head-face, is 58.
const BEAD_RADIUS = { neck: 55, tail: 40 };
// How long the score shows when the game ends, in ms.
const SNAKE_SCORE_MS = 6000;
// The apples to eat for the next level, and how long the snake waits when it
// comes, in ms, so the player sees the new walls and can turn in time.
const APPLES_PER_LEVEL = 3;
const LEVEL_PAUSE_MS = 2600;
// The cells ahead of the head where no wall comes up, as the snake would
// bump into it before a command could turn it. It comes up once the head
// passed.
const CLEAR_AHEAD = 2;

// Each level's new walls, on a garden W cells wide and H high, W >= H, by
// their cells in its upper left quarter, which are mirrored into the others.
// A tall garden takes them turned, down for across. Every free cell can
// still be reached, and a gate in the box leads into a lane.
const quarter = (cells) => (W, H) =>
  cells(W, H).flatMap(([x, y]) => [
    [x, y],
    [W - 1 - x, y],
    [x, H - 1 - y],
    [W - 1 - x, H - 1 - y],
  ]);
const span = (from, to) => Array.from({ length: to - from + 1 }, (_, i) => from + i);
// The cells of a box, along the top and the left side of the quarter, but
// the gates, the cells at most gate from the middle of each side.
const box = (gate) =>
  quarter((W, H) => [
    ...span(0, Math.ceil(W / 2) - 1)
      .filter((x) => Math.abs(x - (W - 1) / 2) > gate)
      .map((x) => [x, 0]),
    ...span(0, Math.ceil(H / 2) - 1)
      .filter((y) => Math.abs(y - (H - 1) / 2) > gate)
      .map((y) => [0, y]),
  ]);
const MAZES = [
  // An open garden.
  () => [],
  // Corners.
  quarter(() => [
    [1, 1],
    [2, 1],
    [1, 2],
  ]),
  // Rails, down either side of the middle, open across it.
  quarter((W, H) => span(2, Math.ceil(H / 2) - 2).map((y) => [Math.floor(W / 3), y])),
  // A box, with gates of three cells.
  box(1.5),
  // Narrower gates, of one cell, or two on a side of an even length.
  box(0.5),
];

// The walls of the levels up to level, on a garden of columns and rows.
function mazeWalls(level, columns, rows) {
  const tall = rows > columns;
  const [W, H] = tall ? [rows, columns] : [columns, rows];
  const walls = new Map();
  for (const maze of MAZES.slice(0, level)) {
    for (const [x, y] of maze(W, H)) {
      const cell = tall ? { x: y, y: x } : { x, y };
      walls.set(`${cell.x},${cell.y}`, cell);
    }
  }
  return [...walls.values()];
}

class Snake {
  // The element of the game, with the garden, the score, the buttons and
  // the banners. onOver is called when the game is over and needs no more
  // commands, onEnd once its score was shown.
  constructor({ root, onOver, onEnd }) {
    this.root = root;
    this.onOver = onOver;
    this.onEnd = onEnd;
    this.ui = {
      field: root.querySelector(".snake-field"),
      garden: root.querySelector(".snake-garden"),
      lawn: root.querySelector(".snake-lawn"),
      ground: root.querySelector(".snake-ground"),
      walls: root.querySelector(".snake-walls"),
      apple: root.querySelector(".snake-apple"),
      body: root.querySelector(".snake-body"),
      heads: root.querySelector(".snake-heads"),
      head: root.querySelector("#snake-head"),
      points: root.querySelector(".snake-points"),
      level: root.querySelector(".snake-level"),
      score: root.querySelector(".snake-score-value"),
      banner: root.querySelector(".game-banner"),
      title: root.querySelector(".game-banner-title"),
      text: root.querySelector(".game-banner-text"),
      turns: Object.fromEntries(
        [...root.querySelectorAll(".snake-turn")].map((button) => [button.dataset.turn, button]),
      ),
    };
    // "intro", "playing", "level", while it waits for the next, "over" or
    // "gone".
    this.phase = "intro";
    this.score = 0;
    // The walls, by their cells as "x,y", each with its element, and
    // whether it is still to come up, where the snake was.
    this.walls = new Map();
    // Whether the next level comes once the snake reaches its cell.
    this.levelUp = false;
    // The apple's cell, none between levels.
    this.apple = null;
    // The cells the snake crawled from to the next, while it can still turn
    // there at once, see TURN_AT_ONCE.
    this.retake = null;
    // How far the head was drawn from where it is, as it turned at once, and
    // when, see SWERVE_MS.
    this.swerve = null;
    this.frame = null;
    this.timers = new Set();
    this.resizing = new ResizeObserver(() => this.layOut());
  }

  // Shows the garden at the level, from 1, with its walls, the snake and the
  // first apple, counts down and starts.
  async start(level = 1) {
    this.level = Math.min(Math.max(level, 1), MAZES.length);
    this.root.hidden = false;
    document.body.classList.add("snake");
    // The cells are as large as fit, in the garden that fits them larger.
    const box = this.ui.field.getBoundingClientRect();
    const fit = ({ columns, rows }) => Math.min(box.width / columns, box.height / rows);
    const { columns, rows } = fit(TALL_GARDEN) > fit(WIDE_GARDEN) ? TALL_GARDEN : WIDE_GARDEN;
    this.columns = columns;
    this.rows = rows;
    this.ui.lawn.setAttribute("viewBox", `0 0 ${columns * CELL} ${rows * CELL}`);
    this.ui.ground.setAttribute("width", columns * CELL);
    this.ui.ground.setAttribute("height", rows * CELL);
    this.layOut();
    this.resizing.observe(this.ui.field);

    // The snake lies in the middle, heading right, its head first.
    const x = Math.floor(columns / 2);
    const y = Math.floor(rows / 2);
    this.cells = Array.from({ length: START_LENGTH }, (_, i) => ({ x: x - i, y }));
    // Where each bead crawls from, to its cell.
    this.from = this.cells.map((cell) => ({ ...cell }));
    this.heading = [1, 0];
    this.turns = [];
    this.crawled = 0;
    this.addWalls(mazeWalls(this.level, columns, rows));
    this.placeApple();
    this.updateHud();
    this.draw();

    this.showBanner("Snake!", "Say “up”, “down”, “left” or “right” to lead the snake to the apples");
    await this.sleep(3200);
    for (const count of ["3", "2", "1", "Go!"]) {
      if (this.phase === "gone") {
        return;
      }
      this.showBanner(count, "", "count");
      await this.sleep(700);
    }
    this.hideBanner();
    if (this.phase !== "intro") {
      return;
    }
    this.phase = "playing";
    this.last = performance.now();
    this.frame = requestAnimationFrame((now) => this.tick(now));
  }

  // Turns the snake, for "UP", "DOWN", "LEFT" or "RIGHT", at the next cell,
  // or after the turns said before. It cannot turn back into itself.
  command(name) {
    const turn = TURNS[name];
    if (!turn || !this.listening) {
      return;
    }
    const [dx, dy] = this.turns.at(-1) ?? this.heading;
    const back = turn[0] === -dx && turn[1] === -dy;
    if (back) {
      this.press(name, "refused");
      return;
    }
    if ((turn[0] !== dx || turn[1] !== dy) && !this.turnAtOnce(turn)) {
      if (this.turns.length === AHEAD) {
        this.turns.shift();
      }
      this.turns.push(turn);
    }
    this.press(name, "pressed");
  }

  // Turns the snake at the cell it is leaving, when the turn comes early in
  // its crawl, see TURN_AT_ONCE, and returns whether it did. Not after it
  // turned or ate there, nor where the turn would bump into a wall or into
  // itself, which waits for the next cell.
  turnAtOnce(turn) {
    const cells = this.retake;
    if (this.phase !== "playing" || !cells || this.turns.length || this.crawled >= TURN_AT_ONCE) {
      return false;
    }
    const move = this.ahead(cells, turn);
    if (move.bumps) {
      return false;
    }
    const [x, y] = this.beadAt(0);
    this.retake = null;
    this.heading = turn;
    this.move(cells, move);
    const [toX, toY] = this.beadAt(0);
    this.swerve = { x: x - toX, y: y - toY, start: performance.now() };
    return true;
  }

  // Ends the game, such as when the player stops listening, showing the
  // score first.
  quit() {
    if (this.phase === "intro") {
      this.destroy();
      this.onEnd();
    } else if (this.phase === "playing" || this.phase === "level") {
      this.gameOver();
    }
  }

  // Ends the game at once.
  destroy() {
    if (this.phase === "gone") {
      return;
    }
    this.phase = "gone";
    cancelAnimationFrame(this.frame);
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.resizing.disconnect();
    for (const layer of [this.ui.walls, this.ui.body, this.ui.heads, this.ui.points]) {
      layer.replaceChildren();
    }
    this.walls.clear();
    this.ui.garden.classList.remove("bitten");
    this.hideBanner();
    this.root.hidden = true;
    document.body.classList.remove("snake");
  }

  // Whether the game still needs to hear commands.
  get listening() {
    return this.phase === "intro" || this.phase === "playing" || this.phase === "level";
  }

  // The garden as large as fits in its field, with whole pixels per cell, so
  // the lawn's checks stay sharp.
  layOut() {
    const box = this.ui.field.getBoundingClientRect();
    const garden = getComputedStyle(this.ui.garden);
    const frame = parseFloat(garden.borderLeftWidth) + parseFloat(garden.borderRightWidth);
    const cell = Math.max(
      Math.floor(Math.min((box.width - frame) / this.columns, (box.height - frame) / this.rows)),
      1,
    );
    this.ui.garden.style.width = `${cell * this.columns + frame}px`;
    this.ui.garden.style.height = `${cell * this.rows + frame}px`;
  }

  // The game.

  tick(now) {
    if (this.phase !== "playing") {
      return;
    }
    // A hidden tab pauses the frames, so never jump far ahead.
    this.crawled += Math.min((now - this.last) / 1000, 0.1) / CRAWL_SECONDS;
    this.last = now;
    while (this.crawled >= 1 && this.phase === "playing") {
      this.crawled -= 1;
      this.crawl();
    }
    if (this.phase === "playing") {
      this.draw();
      this.frame = requestAnimationFrame((time) => this.tick(time));
    }
  }

  // The snake crawls a cell, turning first if it was told to, and eats the
  // apple there, or bites itself or bumps into a wall. When the next level
  // comes, it waits for it instead.
  crawl() {
    if (this.levelUp) {
      this.nextLevel();
      return;
    }
    const straight = !this.turns.length;
    if (!straight) {
      this.heading = this.turns.shift();
    }
    const cells = this.cells;
    const move = this.ahead(cells, this.heading);
    this.retake = null;
    if (move.bumps) {
      this.crawled = 0;
      this.from = cells.map((cell) => ({ ...cell }));
      this.draw();
      this.gameOver();
      return;
    }
    this.move(cells, move);
    // A straight crawl can still turn at once, unless it ate.
    if (straight && !move.eats) {
      this.retake = cells;
    }
  }

  // Where the snake crawls from its cells, heading [dx, dy]: the head's next
  // cell, whether it eats the apple there, the beads that follow it, and
  // whether it bumps into a wall or into itself.
  ahead(cells, [dx, dy]) {
    const [head] = cells;
    const next = this.wrap({ x: head.x + dx, y: head.y + dy });
    const eats = this.apple !== null && next.x === this.apple.x && next.y === this.apple.y;
    // The tail leaves its cell as the head comes, unless the snake grows.
    const body = eats ? cells : cells.slice(0, -1);
    const bumps = this.isWall(next) || body.some((cell) => cell.x === next.x && cell.y === next.y);
    return { next, eats, body, bumps };
  }

  // The snake crawls from its cells as ahead found.
  move(cells, { next, eats, body }) {
    this.from = cells.map((cell) => ({ ...cell }));
    this.cells = [next, ...body];
    if (eats) {
      // The new bead is the old tail, which stays where it was.
      this.from.push({ ...this.cells.at(-1) });
      this.eat();
    }
    this.raiseWalls();
  }

  wrap({ x, y }) {
    return { x: (x + this.columns) % this.columns, y: (y + this.rows) % this.rows };
  }

  eat() {
    this.score += 1;
    this.point(this.apple);
    if (this.score % APPLES_PER_LEVEL === 0 && this.level < MAZES.length) {
      // The next level comes as the snake reaches the apple's cell, and
      // brings the next apple, which its walls would otherwise cover.
      this.levelUp = true;
      this.apple = null;
      this.ui.apple.setAttribute("visibility", "hidden");
    } else if (!this.placeApple()) {
      // The snake fills the garden.
      this.gameOver(true);
    }
    this.updateHud();
  }

  // The next level: the snake waits on its cell while the level's walls come
  // up, and then crawls on, to a new apple. Commands said meanwhile turn it
  // when it crawls on.
  nextLevel() {
    this.levelUp = false;
    this.level += 1;
    this.phase = "level";
    this.crawled = 0;
    this.from = this.cells.map((cell) => ({ ...cell }));
    this.draw();
    this.addWalls(mazeWalls(this.level, this.columns, this.rows));
    this.updateHud();
    this.showBanner(`Level ${this.level}!`, "Watch out for the walls");
    this.later(() => {
      if (this.phase !== "level") {
        return;
      }
      this.hideBanner();
      if (!this.placeApple()) {
        this.gameOver(true);
        return;
      }
      this.phase = "playing";
      this.last = performance.now();
      this.frame = requestAnimationFrame((now) => this.tick(now));
    }, LEVEL_PAUSE_MS);
  }

  // Adds the walls of the cells it has not yet, each still to come up.
  addWalls(cells) {
    for (const cell of cells) {
      const key = `${cell.x},${cell.y}`;
      if (this.walls.has(key)) {
        continue;
      }
      // The stone grows from the middle of its cell, as the apple does.
      const place = document.createElementNS(SVG_NS, "g");
      place.setAttribute("transform", `translate(${(cell.x + 0.5) * CELL} ${(cell.y + 0.5) * CELL})`);
      const element = document.createElementNS(SVG_NS, "g");
      element.classList.add("snake-wall", "pending");
      const stone = document.createElementNS(SVG_NS, "use");
      stone.setAttribute("href", "#snake-stone");
      element.append(stone);
      place.append(element);
      this.ui.walls.append(place);
      this.walls.set(key, { element, pending: true });
    }
    this.raiseWalls();
  }

  // The walls still to come up do, but where the snake lies, or is about to
  // crawl.
  raiseWalls() {
    const near = new Set(this.cells.map(({ x, y }) => `${x},${y}`));
    const [dx, dy] = this.turns[0] ?? this.heading;
    let [ahead] = this.cells;
    for (let i = 0; i < CLEAR_AHEAD; i++) {
      ahead = this.wrap({ x: ahead.x + dx, y: ahead.y + dy });
      near.add(`${ahead.x},${ahead.y}`);
    }
    for (const [key, wall] of this.walls) {
      if (wall.pending && !near.has(key)) {
        wall.pending = false;
        wall.element.classList.replace("pending", "up");
      }
    }
  }

  isWall({ x, y }) {
    const wall = this.walls.get(`${x},${y}`);
    return wall !== undefined && !wall.pending;
  }

  // A new apple on a cell neither the snake nor a wall takes, also one still
  // to come up, or none, false, where they take all.
  placeApple() {
    const taken = new Set([...this.cells.map(({ x, y }) => `${x},${y}`), ...this.walls.keys()]);
    const free = [];
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.columns; x++) {
        if (!taken.has(`${x},${y}`)) {
          free.push({ x, y });
        }
      }
    }
    if (!free.length) {
      this.apple = null;
      this.ui.apple.setAttribute("visibility", "hidden");
      return false;
    }
    this.apple = free[Math.floor(Math.random() * free.length)];
    const { apple } = this.ui;
    apple.setAttribute("transform", `translate(${(this.apple.x + 0.5) * CELL} ${(this.apple.y + 0.5) * CELL})`);
    apple.removeAttribute("visibility");
    // It grows again, also when it is the same element.
    apple.classList.remove("grown");
    void apple.getBoundingClientRect();
    apple.classList.add("grown");
    return true;
  }

  // A "+1" rises where the apple was eaten.
  point({ x, y }) {
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", (x + 0.5) * CELL);
    text.setAttribute("y", Math.max((y + 0.3) * CELL, 60));
    text.setAttribute("text-anchor", "middle");
    text.classList.add("snake-point");
    text.textContent = "+1";
    this.ui.points.append(text);
    this.later(() => text.remove(), 900);
  }

  // The snake bit itself or bumped into a wall, or fills the garden, when won.
  gameOver(won = false) {
    this.phase = "over";
    this.turns = [];
    this.onOver();
    cancelAnimationFrame(this.frame);
    this.ui.garden.classList.add("bitten");
    const apples = this.score === 1 ? "apple" : "apples";
    this.showBanner(
      won ? "You did it!" : "Great game!",
      won ? "The snake fills the whole garden" : `The snake ate ${this.score} ${apples}`,
      "score",
    );
    this.later(() => {
      this.destroy();
      this.onEnd();
    }, SNAKE_SCORE_MS);
  }

  // Drawing.

  // Draws each bead on its way from its last cell to its cell, the head
  // last, over the beads. A bead crawling out of one side of the garden is
  // drawn there and coming in at the other.
  draw() {
    const beads = [];
    const n = this.cells.length;
    for (let i = n - 1; i >= 0; i--) {
      let [x, y] = this.beadAt(i);
      if (i === 0 && this.swerve) {
        // Eased out, from where the head was drawn before it turned.
        const left = 1 - (performance.now() - this.swerve.start) / SWERVE_MS;
        if (left > 0) {
          x += this.swerve.x * left * left;
          y += this.swerve.y * left * left;
        } else {
          this.swerve = null;
        }
      }
      const k = n > 2 ? (i - 1) / (n - 2) : 0;
      const radius = BEAD_RADIUS.neck + (BEAD_RADIUS.tail - BEAD_RADIUS.neck) * k * k;
      for (const [cx, cy] of this.copies(x, y)) {
        beads.push({ i, cx, cy, radius });
      }
    }
    // The beads, in a pool of circles that grows as the snake does.
    const body = beads.filter(({ i }) => i > 0);
    const circles = this.ui.body.children;
    while (circles.length < body.length) {
      this.ui.body.append(document.createElementNS(SVG_NS, "circle"));
    }
    body.forEach(({ i, cx, cy, radius }, j) => {
      const circle = circles[j];
      circle.setAttribute("cx", cx);
      circle.setAttribute("cy", cy);
      circle.setAttribute("r", radius);
      circle.setAttribute("fill", BEADS[(i - 1) % BEADS.length]);
      circle.removeAttribute("visibility");
    });
    for (let j = body.length; j < circles.length; j++) {
      circles[j].setAttribute("visibility", "hidden");
    }
    // The head, facing the way it crawls, twice as it crosses a side.
    const heads = beads.filter(({ i }) => i === 0);
    while (this.ui.heads.children.length < heads.length) {
      const head = this.ui.head.cloneNode(true);
      head.removeAttribute("id");
      this.ui.heads.append(head);
    }
    const angle = Math.atan2(this.heading[1], this.heading[0]) * (180 / Math.PI);
    [...this.ui.heads.children].forEach((head, j) => {
      if (j < heads.length) {
        head.setAttribute("transform", `translate(${heads[j].cx} ${heads[j].cy}) rotate(${angle})`);
        head.removeAttribute("visibility");
      } else {
        head.setAttribute("visibility", "hidden");
      }
    });
  }

  // The cell, x and y, where bead i is on its way from its last cell to its
  // cell, the short way round, past a side of the garden.
  beadAt(i) {
    const to = this.cells[i];
    const from = this.from[i];
    const dx = Math.abs(to.x - from.x) > 1 ? -Math.sign(to.x - from.x) : to.x - from.x;
    const dy = Math.abs(to.y - from.y) > 1 ? -Math.sign(to.y - from.y) : to.y - from.y;
    return [from.x + dx * this.crawled, from.y + dy * this.crawled];
  }

  // Where to draw a bead at the cell x, y: there, and also at the other
  // side of the garden while it crawls out of one.
  copies(x, y) {
    const wrap = (value, count) => ((value % count) + count) % count;
    const along = (value, count) => {
      const center = (wrap(value, count) + 0.5) * CELL;
      const at = [center];
      if (center < CELL / 2) {
        at.push(center + count * CELL);
      } else if (center > (count - 0.5) * CELL) {
        at.push(center - count * CELL);
      }
      return at;
    };
    return along(x, this.columns).flatMap((cx) => along(y, this.rows).map((cy) => [cx, cy]));
  }

  // The button of a command heard lights up, or shakes for a turn back.
  press(name, kind) {
    const button = this.ui.turns[name];
    if (!button) {
      return;
    }
    button.classList.remove("pressed", "refused");
    // Restart the animation when the same command comes again.
    void button.offsetWidth;
    button.classList.add(kind);
  }

  updateHud() {
    this.ui.level.textContent = `Level ${this.level}`;
    this.ui.score.textContent = this.score;
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

  // Timers that end with the game.
  later(callback, ms) {
    const timer = setTimeout(() => {
      this.timers.delete(timer);
      callback();
    }, ms);
    this.timers.add(timer);
  }

  sleep(ms) {
    return new Promise((resolve) => this.later(resolve, ms));
  }
}
