// Superball: flowers fall toward the unicorn, which the player steers left
// and right by voice, or with the arrow keys or buttons, to dodge them.
// Each flower dodged is a point, every 10 points the next level makes the
// flowers fall faster, and three flowers end the game.

const SVG_NS = "http://www.w3.org/2000/svg";

// The meadow's drawing is 1600 by 900, cut to fill the screen.
const SCENE = { width: 1600, height: 900 };
// Where the unicorn stands on the meadow, and its size: the middle of its
// hooves, in the meadow's units.
const UNICORN = { x: 1132.6, y: 739.4, width: 391, height: 396 };

const LANES = 5;
// For each level: the seconds a flower takes to fall to the unicorn, the
// seconds between flowers, and how often one falls toward the unicorn.
const LEVELS = [
  { fall: 6.5, every: 2.6, aim: 0.5 },
  { fall: 5.2, every: 2.2, aim: 0.55 },
  { fall: 4.2, every: 1.85, aim: 0.6 },
  { fall: 3.4, every: 1.55, aim: 0.65 },
  { fall: 2.7, every: 1.3, aim: 0.7 },
];
const POINTS_PER_LEVEL = 10;
const HEARTS = 3;
// After a flower hits, the unicorn blinks and cannot be hit again.
const HURT_SECONDS = 1.4;
// The flower drawing's radius, and its scale beside a lane of this width.
const FLOWER = { radius: 20, scale: 2, lane: 230 };
const COLORS = ["#ff7ab8", "#b18cff", "#ffa94d", "#5cc0ff", "#ffffff", "#ff6b6b"];
// How long the unicorn takes to trot to the game and back, in ms.
const TROT_MS = 800;
// How long the score shows when the game ends, in ms.
const SCORE_MS = 6000;

const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

class Superball {
  // The elements: the one the meadow fills, the unicorn on the meadow, a
  // group to draw the flowers in, and the overlay with the score and the
  // buttons.
  // onOver is called when the game is over and needs no more commands,
  // onEnd once its score was shown.
  constructor({ view, unicorn, layer, overlay, onOver, onEnd }) {
    this.view = view;
    this.unicorn = unicorn;
    this.layer = layer;
    this.overlay = overlay;
    this.onOver = onOver;
    this.onEnd = onEnd;
    this.ui = {
      level: overlay.querySelector(".game-level"),
      score: overlay.querySelector(".game-score-value"),
      hearts: [...overlay.querySelectorAll(".game-heart")],
      banner: overlay.querySelector(".game-banner"),
      title: overlay.querySelector(".game-banner-title"),
      text: overlay.querySelector(".game-banner-text"),
      left: overlay.querySelector(".game-steer.left"),
      right: overlay.querySelector(".game-steer.right"),
    };
    // "intro", "playing", "over" or "gone".
    this.phase = "intro";
    this.flowers = [];
    this.score = 0;
    this.hearts = HEARTS;
    this.lane = Math.floor(LANES / 2);
    this.facing = -1; // The unicorn looks left, as drawn.
    this.hurtUntil = 0;
    this.frame = null;
    this.timers = new Set();
    // Counts the trots, so a new one stops the one before.
    this.trots = 0;
    this.trotting = false;
  }

  // Trots the unicorn to the game, counts down and starts.
  async start(level) {
    this.level = Math.min(Math.max(level, 1), LEVELS.length);
    this.overlay.hidden = false;
    document.body.classList.add("playing");
    this.updateHud();
    this.measure();
    this.x = this.laneX(this.lane);
    await this.trot(true);
    if (this.phase === "gone") {
      return;
    }
    // The unicorn can move already, but no flowers fall yet.
    this.last = performance.now();
    this.frame = requestAnimationFrame((now) => this.tick(now));
    this.showBanner("Superball!", "Say “left” or “right” to dodge the flowers");
    await this.sleep(2600);
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
    this.nextFlower = 0;
  }

  // Moves one lane, for "LEFT" or "RIGHT".
  command(name) {
    if (this.phase !== "intro" && this.phase !== "playing") {
      return;
    }
    const step = name === "LEFT" ? -1 : name === "RIGHT" ? 1 : 0;
    if (!step) {
      return;
    }
    this.lane = Math.min(Math.max(this.lane + step, 0), LANES - 1);
    this.facing = step;
    const button = step < 0 ? this.ui.left : this.ui.right;
    button.classList.remove("pressed");
    // Restart the animation when the same command comes again.
    void button.offsetWidth;
    button.classList.add("pressed");
  }

  // Ends the game, such as when the player stops listening, showing the
  // score first.
  quit() {
    if (this.phase === "intro") {
      this.destroy();
      this.onEnd();
    } else if (this.phase === "playing") {
      this.gameOver();
    }
  }

  // Ends the game at once, and the unicorn trots back.
  destroy() {
    if (this.phase === "gone") {
      return;
    }
    this.phase = "gone";
    cancelAnimationFrame(this.frame);
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.layer.replaceChildren();
    this.overlay.hidden = true;
    this.hideBanner();
    document.body.classList.remove("playing");
    this.unicorn.classList.remove("hurt");
    this.trot(false);
  }

  get active() {
    return this.phase !== "gone";
  }

  // Whether the game still needs to hear commands.
  get listening() {
    return this.phase === "intro" || this.phase === "playing";
  }

  // Geometry.

  // The part of the meadow on screen, which the lanes fit in, and the size
  // of one of the meadow's units on screen.
  measure() {
    const box = this.view.getBoundingClientRect();
    const scale = Math.max(box.width / SCENE.width, box.height / SCENE.height);
    this.scale = scale;
    const width = box.width / scale;
    const height = box.height / scale;
    this.top = SCENE.height - height;
    // Narrow screens, such as phones upright, fit narrower lanes with all
    // in them smaller.
    this.spacing = Math.min(260, width * 0.18);
    this.size = (0.95 * this.spacing) / UNICORN.width;
    this.flowerScale = FLOWER.scale * Math.min(1, this.spacing / FLOWER.lane);
    this.radius = FLOWER.radius * this.flowerScale;
    this.ground = SCENE.height - Math.min(36, height * 0.04);
  }

  laneX(lane) {
    return SCENE.width / 2 + (lane - (LANES - 1) / 2) * this.spacing;
  }

  // Draws the unicorn at x, at its game size, facing where it last moved.
  place(x = this.x, y = this.ground, size = this.size, facing = this.facing) {
    const flip = facing > 0 ? -1 : 1;
    const dx = (x - flip * size * UNICORN.x) * this.scale;
    const dy = (y - size * UNICORN.y) * this.scale;
    this.unicorn.style.transform = `translate(${dx}px, ${dy}px) scale(${flip * size}, ${size})`;
  }

  // Trots the unicorn from where it stands on the meadow to the game, or
  // back, facing where it goes.
  trot(into) {
    const trot = ++this.trots;
    this.trotting = true;
    return new Promise((resolve) => {
      const start = performance.now();
      const step = (now) => {
        if (trot !== this.trots) {
          resolve();
          return;
        }
        const t = Math.min((now - start) / TROT_MS, 1);
        const k = ease(into ? t : 1 - t);
        this.measure();
        const target = into ? this.laneX(this.lane) : this.x;
        const x = UNICORN.x + (target - UNICORN.x) * k;
        const y = UNICORN.y + (this.ground - UNICORN.y) * k;
        const size = 1 + (this.size - 1) * k;
        const facing = Math.sign(into ? target - UNICORN.x : UNICORN.x - target) || -1;
        this.place(x, y, size, t < 1 ? facing : into ? this.facing : -1);
        if (t < 1) {
          requestAnimationFrame(step);
          return;
        }
        this.trotting = false;
        if (into) {
          this.x = target;
        } else {
          this.unicorn.style.removeProperty("transform");
        }
        resolve();
      };
      requestAnimationFrame(step);
    });
  }

  // The game.

  tick(now) {
    if (this.phase === "gone") {
      return;
    }
    // A hidden tab pauses the frames, so never jump far ahead.
    const dt = Math.min((now - this.last) / 1000, 0.05);
    this.last = now;
    this.measure();

    // The unicorn glides to its lane.
    this.x += (this.laneX(this.lane) - this.x) * (1 - Math.exp(-dt * 18));
    this.place();

    if (this.phase === "playing") {
      this.nextFlower -= dt;
      if (this.nextFlower <= 0) {
        this.drop();
        this.nextFlower = LEVELS[this.level - 1].every;
      }
    }
    for (const flower of this.flowers) {
      this.fall(flower, dt, now);
    }
    this.flowers = this.flowers.filter((flower) => flower.element.isConnected);
    this.frame = requestAnimationFrame((time) => this.tick(time));
  }

  // Drops a flower, often toward the unicorn, never onto another that has
  // only just started to fall.
  drop() {
    const { aim } = LEVELS[this.level - 1];
    const fresh = new Set(
      this.flowers.filter((f) => f.y < this.top + 260).map((f) => f.lane),
    );
    let lanes = [...Array(LANES).keys()].filter((lane) => !fresh.has(lane));
    if (lanes.length === 0) {
      return;
    }
    if (lanes.includes(this.lane) && Math.random() < aim) {
      lanes = [this.lane];
    }
    const lane = lanes[Math.floor(Math.random() * lanes.length)];
    const element = document.createElementNS(SVG_NS, "use");
    element.setAttribute("href", "#flower");
    element.setAttribute("fill", COLORS[Math.floor(Math.random() * COLORS.length)]);
    element.classList.add("falling-flower");
    this.layer.append(element);
    this.flowers.push({
      element,
      lane,
      y: this.top - this.radius * 1.5,
      turn: Math.random() * 360,
      spin: (Math.random() < 0.5 ? -1 : 1) * (40 + Math.random() * 50),
      state: "falling",
    });
  }

  fall(flower, dt, now) {
    const { fall } = LEVELS[this.level - 1];
    const speed = (this.ground - this.top) / fall;
    const x = this.laneX(flower.lane);
    if (flower.state === "falling") {
      flower.y += speed * dt;
      flower.turn += flower.spin * dt;
      const height = UNICORN.height * this.size;
      const body = this.ground - height * 0.78;
      if (
        flower.y + this.radius * 0.6 > body &&
        flower.y < this.ground - 10 &&
        Math.abs(x - this.x) < this.spacing * 0.5 &&
        this.phase === "playing" &&
        now >= this.hurtUntil
      ) {
        this.hit(flower, now);
      } else if (flower.y - this.radius > this.ground - height * 0.35) {
        flower.state = "landed";
        if (this.phase === "playing") {
          this.dodged(x, flower.y);
        }
      }
    } else if (flower.state === "landed") {
      // It sinks into the grass.
      flower.y += speed * dt * 0.3;
      if (flower.y > this.ground + this.radius) {
        flower.element.remove();
      }
    }
    flower.element.setAttribute(
      "transform",
      `translate(${x} ${flower.y}) rotate(${flower.turn}) scale(${this.flowerScale})`,
    );
  }

  hit(flower, now) {
    flower.state = "popped";
    flower.element.classList.add("pop");
    const element = flower.element;
    this.later(() => element.remove(), 500);
    this.hearts -= 1;
    this.hurtUntil = now + HURT_SECONDS * 1000;
    this.unicorn.classList.remove("hurt");
    void this.unicorn.getBoundingClientRect();
    this.unicorn.classList.add("hurt");
    this.later(() => this.unicorn.classList.remove("hurt"), HURT_SECONDS * 1000);
    this.updateHud();
    if (this.hearts === 0) {
      this.gameOver();
    }
  }

  dodged(x, y) {
    this.score += 1;
    this.sparkle(x, y);
    if (this.score % POINTS_PER_LEVEL === 0 && this.level < LEVELS.length) {
      this.level += 1;
      this.showBanner(`Level ${this.level}!`, "The flowers fall faster", "flash");
      this.later(() => this.phase === "playing" && this.hideBanner(), 1800);
    }
    this.updateHud();
  }

  // A "+1" rises where a flower was dodged.
  sparkle(x, y) {
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", x);
    text.setAttribute("y", Math.min(y, this.ground - 40));
    text.setAttribute("text-anchor", "middle");
    text.classList.add("game-point");
    text.textContent = "+1";
    this.layer.append(text);
    this.later(() => text.remove(), 900);
  }

  gameOver() {
    this.phase = "over";
    this.onOver();
    // The flowers still falling fade away.
    for (const flower of this.flowers) {
      if (flower.state === "falling") {
        flower.state = "popped";
        flower.element.classList.add("fade");
        const { element } = flower;
        this.later(() => element.remove(), 600);
      }
    }
    const flowers = this.score === 1 ? "flower" : "flowers";
    this.showBanner("Great game!", `You dodged ${this.score} ${flowers}`, "score");
    this.later(() => {
      this.destroy();
      this.onEnd();
    }, SCORE_MS);
  }

  // The score and banners.

  updateHud() {
    this.ui.level.textContent = `Level ${this.level}`;
    this.ui.score.textContent = this.score;
    this.ui.hearts.forEach((heart, i) => heart.classList.toggle("lost", i >= this.hearts));
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
