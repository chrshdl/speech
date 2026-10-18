// The synthesizer's keyboard: an octave of keys, C to C, from C4 at first and
// shifted an octave down or up by the buttons of app.js, played by touch,
// also with several fingers at once, a finger sliding over the keys playing
// each key it reaches, as on a synthesizer's ribbon of keys. It only draws
// the keys and tells what is played; app.js plays it with synth.js.
//
// On the Raspberry Pi's touch display, WPE WebKit sometimes loses a finger
// of a chord lifting: it never ends, and its key would sound on for good.
// So once no touch is left, or touches start with no other down, every key
// still down from before is let go. Not by isPrimary: WPE WebKit takes the
// first touch point of the event as the primary one, by the controller's
// slot, so of a chord changing, a finger landing in a lower slot than one
// held let go of the held one's key.

// The lowest key, a C, at first, and as low and as high as it shifts: the
// keys reach from C2 to C7.
const FIRST_KEY = 60; // C4
const LOWEST_FIRST_KEY = 36; // C2
const HIGHEST_FIRST_KEY = 84; // C6
// The white keys of an octave, from C, and where each black key sits, as
// the white keys it lies between.
const WHITE_STEPS = [0, 2, 4, 5, 7, 9, 11];
// The colors of the white keys' stripes, a rainbow over each octave.
const KEY_COLORS = ["#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#4dabf7", "#9775fa", "#f783ac"];
// How far down a white key the black keys reach.
const BLACK_DEPTH = 0.6;

class Keys {
  // The element to draw the keys in, and what is called when a key is
  // pressed, with an id of the press and the key's frequency, and when that
  // press lets go.
  constructor({ element, onDown, onUp }) {
    this.element = element;
    this.onDown = onDown;
    this.onUp = onUp;
    // The presses sounding, by pointer: the key and the press's id.
    this.presses = new Map();
    // The pointers pressed since the last touch started: WebKit sends a
    // touch's pointerdown before its touchstart.
    this.starting = new Set();
    this.nextId = 1;
    this.first = FIRST_KEY;
    this.draw();
    this.listeners = {
      pointerdown: (event) => this.down(event),
      pointermove: (event) => this.move(event),
      pointerup: (event) => this.up(event),
      pointercancel: (event) => this.up(event),
      // Touches are played as pointers, so the touches themselves do
      // nothing else, such as the mouse and click WebKit makes of a tap.
      touchstart: (event) => this.touchesStart(event),
      touchmove: (event) => event.preventDefault(),
      touchend: (event) => this.touchesLeft(event),
      touchcancel: (event) => this.touchesLeft(event),
    };
    for (const [name, listener] of Object.entries(this.listeners)) {
      element.addEventListener(name, listener, { passive: false });
    }
  }

  // The name of the lowest key, such as C4.
  get name() {
    return `C${this.first / 12 - 1}`;
  }

  get canShiftDown() {
    return this.first > LOWEST_FIRST_KEY;
  }

  get canShiftUp() {
    return this.first < HIGHEST_FIRST_KEY;
  }

  // Shifts the keys an octave down, by -1, or up, by 1, letting go of every
  // key still down.
  shift(octaves) {
    const first = Math.min(Math.max(this.first + 12 * octaves, LOWEST_FIRST_KEY), HIGHEST_FIRST_KEY);
    if (first === this.first) {
      return;
    }
    this.releaseAll();
    this.first = first;
    this.draw();
  }

  // Draws the octave of keys from the lowest.
  draw() {
    const last = this.first + 12;
    this.keys = new Map();
    this.whites = [];
    for (let key = this.first; key <= last; key++) {
      if (WHITE_STEPS.includes((key - this.first) % 12)) {
        this.whites.push(key);
      }
    }
    // The stylesheet sizes the board for these eight white keys.
    this.element.replaceChildren();
    for (const key of this.whites) {
      const index = this.whites.indexOf(key);
      const div = document.createElement("div");
      div.className = "keys-key white";
      div.style.setProperty("--i", index);
      div.style.setProperty("--color", KEY_COLORS[index % 7]);
      this.element.append(div);
      this.keys.set(key, div);
    }
    for (let key = this.first; key <= last; key++) {
      if (this.keys.has(key)) {
        continue;
      }
      const div = document.createElement("div");
      div.className = "keys-key black";
      // Between the white key below it and the one above.
      div.style.setProperty("--i", this.whites.indexOf(key - 1) + 1);
      this.element.append(div);
      this.keys.set(key, div);
    }
  }

  // The key at a point of the screen, null off the keys: a black key in
  // the upper part of the keys, else the white key under it.
  keyAt(x, y) {
    // The keys fill the board inside its border, less a gap all round.
    const rect = this.element.getBoundingClientRect();
    const gap = parseFloat(getComputedStyle(this.element).paddingLeft);
    const left = rect.left + this.element.clientLeft + gap;
    const top = rect.top + this.element.clientTop + gap;
    const across = (x - left) / (this.element.clientWidth - 2 * gap);
    const down = (y - top) / (this.element.clientHeight - 2 * gap);
    if (across < 0 || across >= 1 || down < 0 || down >= 1) {
      return null;
    }
    const white = across * this.whites.length;
    const index = Math.floor(white);
    if (down < BLACK_DEPTH) {
      // A black key is a third of a white key's width each side of the
      // line between two white keys.
      const line = Math.round(white);
      if (Math.abs(white - line) < 0.3 && line > 0 && line < this.whites.length) {
        const black = this.whites[line - 1] + 1;
        if (black !== this.whites[line]) {
          return black;
        }
      }
    }
    return this.whites[index];
  }

  down(event) {
    event.preventDefault();
    if (event.pointerType === "touch") {
      this.starting.add(event.pointerId);
    }
    // A pointer that presses again lets go of what it pressed.
    this.release(event.pointerId);
    // Moves go on to this element also once the finger left it.
    this.element.setPointerCapture?.(event.pointerId);
    this.press(event.pointerId, this.keyAt(event.clientX, event.clientY));
  }

  // Touches started: with no other down, any press but theirs was lost.
  touchesStart(event) {
    event.preventDefault();
    if (event.touches.length === event.changedTouches.length) {
      for (const pointer of [...this.presses.keys()]) {
        if (!this.starting.has(pointer)) {
          this.release(pointer);
          this.presses.delete(pointer);
        }
      }
    }
    this.starting.clear();
  }

  // A touch ended: with none left, every key is let go.
  touchesLeft(event) {
    event.preventDefault();
    if (event.touches.length === 0) {
      this.releaseAll();
    }
  }

  releaseAll() {
    for (const pointer of [...this.presses.keys()]) {
      this.release(pointer);
    }
    this.presses.clear();
  }

  move(event) {
    const press = this.presses.get(event.pointerId);
    if (press === undefined) {
      return;
    }
    const key = this.keyAt(event.clientX, event.clientY);
    if (press === null ? key !== null : key !== press.key) {
      this.release(event.pointerId);
      this.press(event.pointerId, key);
    }
  }

  up(event) {
    if (this.presses.has(event.pointerId)) {
      this.release(event.pointerId);
      this.presses.delete(event.pointerId);
    }
  }

  // A pointer on a key, or off the keys while still down, null.
  press(pointer, key) {
    if (key === null) {
      this.presses.set(pointer, null);
      return;
    }
    const id = this.nextId++;
    this.presses.set(pointer, { key, id });
    this.keys.get(key).classList.add("down");
    this.onDown(id, 440 * 2 ** ((key - 69) / 12));
  }

  release(pointer) {
    const press = this.presses.get(pointer);
    if (!press) {
      return;
    }
    this.presses.set(pointer, null);
    this.onUp(press.id);
    // Still down while another finger holds it.
    const held = [...this.presses.values()].some((other) => other && other.key === press.key);
    if (!held) {
      this.keys.get(press.key).classList.remove("down");
    }
  }

  // Lets go of every key, and stops listening.
  destroy() {
    this.releaseAll();
    for (const [name, listener] of Object.entries(this.listeners)) {
      this.element.removeEventListener(name, listener);
    }
    this.element.replaceChildren();
  }
}
