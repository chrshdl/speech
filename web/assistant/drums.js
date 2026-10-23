// The drums: a kit drawn in index.html, played by touch, also with several
// fingers at once, a drum sounding as a finger lands on it. Its sounds are
// recordings, in web/assistant/drums, decoded once when the page loads into
// buffers, which a hit plays at once. The kit only tells what is hit; app.js
// plays it.

// How long a drum shows it was hit, in ms.
const HIT_MS = 140;
// The kit's grid, as wide as tall screens, where style.css moves the drums
// into rows below each other, the bass drum at the bottom. The kit takes the
// one that draws it larger.
const WIDE_KIT = { width: 1000, height: 600 };
const TALL_KIT = { width: 600, height: 940 };
// The drums, each with its recording, drums/<name>.wav.
const DRUM_SOUNDS = ["kick", "snare", "tom1", "tom2", "floor", "hihat", "crash", "ride"];

// Each drum's recording decoded for the context, as a map from the drum's
// name to an AudioBuffer. Decoding resamples it to the context's rate, such
// as the 48 kHz of the Raspberry Pi's speakerphone, once, rather than each
// time it plays.
async function decodeDrums(context) {
  const decoded = await Promise.all(
    DRUM_SOUNDS.map(async (name) => {
      const url = `drums/${name}.wav`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${url}: ${response.status}`);
      }
      return [name, await context.decodeAudioData(await response.arrayBuffer())];
    }),
  );
  return new Map(decoded);
}

class DrumKit {
  // The element of the kit, whose drums have data-drum with their sound's
  // name, and what is called with that name when a drum is hit, and when a
  // pointer lifts.
  constructor({ element, onHit, onLift }) {
    this.element = element;
    this.onHit = onHit;
    this.onLift = onLift;
    // The timers that end each drum's showing it was hit.
    this.lit = new Map();
    const lift = () => this.onLift();
    this.listeners = {
      pointerdown: (event) => this.down(event),
      pointerup: lift,
      pointercancel: lift,
      // Touches are played as pointers, so the touches themselves do
      // nothing else, such as the mouse and click WebKit makes of a tap, or
      // zoom on a double tap.
      touchstart: (event) => event.preventDefault(),
      touchmove: (event) => event.preventDefault(),
      touchend: (event) => event.preventDefault(),
    };
    for (const [name, listener] of Object.entries(this.listeners)) {
      element.addEventListener(name, listener, { passive: false });
    }
    this.resizing = new ResizeObserver(() => this.layOut());
    this.resizing.observe(element);
  }

  layOut() {
    const { clientWidth: width, clientHeight: height } = this.element;
    const scale = (kit) => Math.min(width / kit.width, height / kit.height);
    const tall = scale(TALL_KIT) > scale(WIDE_KIT);
    const kit = tall ? TALL_KIT : WIDE_KIT;
    this.element.classList.toggle("tall", tall);
    this.element.setAttribute("viewBox", `0 0 ${kit.width} ${kit.height}`);
  }

  down(event) {
    if (event.pointerType === "mouse" && event.button !== 0) {
      return;
    }
    event.preventDefault();
    const drum = event.target.closest("[data-drum]");
    if (drum && this.element.contains(drum)) {
      this.hit(drum);
    }
  }

  // Sounds the drum, and shows it was hit, for longer when hit again.
  hit(drum) {
    this.onHit(drum.dataset.drum);
    clearTimeout(this.lit.get(drum));
    drum.classList.add("hit");
    this.lit.set(
      drum,
      setTimeout(() => {
        drum.classList.remove("hit");
        this.lit.delete(drum);
      }, HIT_MS),
    );
  }

  // Stops listening, and shows no drum hit.
  destroy() {
    this.resizing.disconnect();
    for (const [name, listener] of Object.entries(this.listeners)) {
      this.element.removeEventListener(name, listener);
    }
    for (const [drum, timer] of this.lit) {
      clearTimeout(timer);
      drum.classList.remove("hit");
    }
    this.lit.clear();
  }
}
