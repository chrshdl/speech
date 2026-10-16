// Runs web/assistant/synth.js as the audio thread of an AudioWorklet does,
// for tests/synth_test.py:
//
//   node synth_harness.mjs score.json out.f32 RATE FRAMES [STOP_FRAME]
//
// plays the score from the JSON file for at most FRAMES frames at RATE,
// sending the processor "stop" at STOP_FRAME, writes the samples to out.f32
// as float32, and prints what the processor posted, with the frame it
// posted at, and the frame process() returned false at, if it did.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const [scoreFile, outFile, rate, frames, stopFrame] = process.argv.slice(2);
const BLOCK = 128;
const posted = [];
let frame = 0;
let processor = null;

const scope = {
  sampleRate: Number(rate),
  AudioWorkletProcessor: class {
    constructor() {
      this.port = { onmessage: null, postMessage: (data) => posted.push([frame, data]) };
    }
  },
  registerProcessor: (name, Processor) => {
    processor = (options) => new Processor(options);
  },
};
const synth = path.join(path.dirname(new URL(import.meta.url).pathname), "..", "web", "assistant", "synth.js");
vm.runInNewContext(fs.readFileSync(synth, "utf8"), scope);

const song = processor({ processorOptions: JSON.parse(fs.readFileSync(scoreFile, "utf8")) });
const samples = new Float32Array(Number(frames));
let returnedFalse = null;
while (frame < samples.length) {
  if (stopFrame !== undefined && frame >= Number(stopFrame)) {
    song.port.onmessage({ data: "stop" });
  }
  const out = new Float32Array(BLOCK);
  const more = song.process([], [[out]]);
  samples.set(out.subarray(0, Math.min(BLOCK, samples.length - frame)), frame);
  frame += BLOCK;
  if (!more) {
    returnedFalse = frame;
    break;
  }
}
fs.writeFileSync(outFile, Buffer.from(samples.buffer));
console.log(JSON.stringify({ posted, returnedFalse }));
