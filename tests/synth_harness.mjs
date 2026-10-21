// Runs web/assistant/synth.js as the audio thread of an AudioWorklet does,
// for tests/synth_test.py:
//
//   node synth_harness.mjs score.json out.f32 RATE FRAMES [STOP_FRAME]
//
// plays the score from the JSON file for at most FRAMES frames at RATE,
// sending the processor "stop" at STOP_FRAME, writes the samples to out.f32
// as float32, left and right in turn, as the client's node has two
// channels, and prints what the processor posted, with the frame it
// posted at, and the frame process() returned false at, if it did.
//
// A JSON file with a processor runs that one instead, such as the
// keyboard: {"processor": "keyboard", "options": {...}, "messages": [[frame,
// data], ...]}, sending each message before the block at its frame, and
// with "module", from another file of web/assistant, such as speaker.js. A
// message {"int16": [...]} is sent as an ArrayBuffer of those samples.
import fs from "node:fs";
import path from "node:path";

const [scoreFile, outFile, rate, length, stopFrame] = process.argv.slice(2);
const frames = Number(length);
const BLOCK = 128;
const posted = [];
let frame = 0;
const processors = new Map();

// The globals of an AudioWorkletGlobalScope, in Node's own: in a vm
// context instead, each lookup of a global such as Math is many times slower.
Object.assign(globalThis, {
  sampleRate: Number(rate),
  AudioWorkletProcessor: class {
    constructor() {
      this.port = { onmessage: null, postMessage: (data) => posted.push([frame, data]) };
    }
  },
  registerProcessor: (name, Processor) => {
    processors.set(name, Processor);
  },
});
const data = JSON.parse(fs.readFileSync(scoreFile, "utf8"));
const { processor = "song", options = data, messages = [], module = "synth.js" } = data.processor ? data : {};
const file = path.join(path.dirname(new URL(import.meta.url).pathname), "..", "web", "assistant", module);
await import(new URL(`file://${file}`));

const Processor = processors.get(processor);
const song = new Processor({ processorOptions: options });
const samples = new Float32Array(2 * frames);
let returnedFalse = null;
while (frame < frames) {
  if (stopFrame !== undefined && frame >= Number(stopFrame)) {
    song.port.onmessage({ data: "stop" });
  }
  while (messages.length && messages[0][0] <= frame) {
    const message = messages.shift()[1];
    song.port.onmessage({ data: message.int16 ? new Int16Array(message.int16).buffer : message });
  }
  const out = [new Float32Array(BLOCK), new Float32Array(BLOCK)];
  const more = song.process([], [out]);
  for (let i = 0; i < Math.min(BLOCK, frames - frame); i++) {
    samples[2 * (frame + i)] = out[0][i];
    samples[2 * (frame + i) + 1] = out[1][i];
  }
  frame += BLOCK;
  if (!more) {
    returnedFalse = frame;
    break;
  }
}
fs.writeFileSync(outFile, Buffer.from(samples.buffer));
console.log(JSON.stringify({ posted, returnedFalse }));
