// The voice of the additive synthesizer, live on the audio thread, as an
// AudioWorklet: a SongCard's notes, from GET /v1/songs/{songId}, played as
// the synthesizer's C++ engine plays its keys. Each harmonic of a note is
// a sine with two quieter ones detuned by 3 Hz plus 1 Hz per harmonic,
// which thickens it like a chorus, read from a table as the engine reads
// its wavetables, and each note fades in and out linearly, by default as
// fast as the engine's keys do, or as slowly as its voice says. The voices
// are mixed through tanh, as the engine mixes, so loud chords saturate softly.

const DETUNE = 0.4; // The level of each detuned sine.
// How long before its end a note's key is let go, so the next note, also of
// the same key, starts afresh, as when played by hand; and the synthesizer's
// attack and release.
const ENVELOPE_SECONDS = 0.025;
const TABLE_SIZE = 4096;
// The level of a song's loudest moments, as loud as the client's tones.
const PEAK = 0.3;
// How loud a mix gets, at most, against all its harmonics in phase. With
// random phases, the peaks of songs were 0.40 and 0.41 of that.
const PEAK_FRACTION = 0.4;
// What tanh gets at most, all harmonics in phase, where it bends a peak by
// 8%: a song's peaks, at PEAK_FRACTION of it, by 1%, so it changes the
// sound as little as the synthesizer's headroom does.
const DRIVE = 0.5;

const TABLE = new Float32Array(TABLE_SIZE + 1);
for (let i = 0; i <= TABLE_SIZE; i++) {
  TABLE[i] = Math.sin((2 * Math.PI * i) / TABLE_SIZE);
}

// The same random phases each time for a frequency's harmonic, so a song
// sounds the same each time: mulberry32, seeded with them.
function phases(frequency, partial) {
  let seed = (Math.round(frequency * 100) * 31 + partial) >>> 0;
  const random = () => {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return [random(), random(), random()];
}

// A note, ready to play: its frames from the song's start, its envelope's,
// and its oscillators, each a frequency in cycles per frame, a level and a
// phase in cycles, in typed arrays for the loops of the audio thread.
function voice(note, rate) {
  const attack = (note.attack ?? ENVELOPE_SECONDS) * rate;
  const release = (note.release ?? ENVELOPE_SECONDS) * rate;
  const frames = Math.round(note.seconds * rate);
  const steps = [];
  const levels = [];
  const starts = [];
  note.partials.forEach((amplitude, partial) => {
    if (amplitude <= 0) {
      return;
    }
    const frequency = note.frequency * (partial + 1);
    const detune = 3 + partial;
    const level = (note.volume * amplitude) / (1 + 2 * DETUNE);
    steps.push(frequency / rate, (frequency + detune) / rate, (frequency - detune) / rate);
    levels.push(level, level * DETUNE, level * DETUNE);
    starts.push(...phases(note.frequency, partial));
  });
  // Let go just before it ends, and then fading for its release, also on
  // past its end, into the next note.
  const gate = Math.max(frames - ENVELOPE_SECONDS * rate, frames / 2);
  return {
    start: Math.round(note.start * rate),
    // How long it sounds, with its release.
    frames: Math.ceil(gate + release),
    attack,
    release,
    gate,
    // The level it was let go at, less than 1 for a note shorter than the attack.
    held: Math.min(gate / attack, 1),
    steps: Float64Array.from(steps),
    levels: Float32Array.from(levels),
    phases: Float64Array.from(starts),
  };
}

// How loud the notes could get, at most: every harmonic sounding at once
// at the busiest moment, also of notes still fading.
function loudest(notes) {
  let most = 0;
  for (const { start } of notes) {
    let sum = 0;
    for (const note of notes) {
      const fading = Math.max((note.release ?? ENVELOPE_SECONDS) - ENVELOPE_SECONDS, 0);
      if (note.start <= start && start < note.start + note.seconds + fading) {
        sum += note.volume * note.partials.reduce((a, b) => a + b, 0);
      }
    }
    most = Math.max(most, sum);
  }
  return most;
}

class Song extends AudioWorkletProcessor {
  constructor({ processorOptions: score }) {
    super();
    this.voices = score.notes.map((note) => voice(note, sampleRate)).sort((a, b) => a.start - b.start);
    this.frames = Math.round(score.seconds * sampleRate);
    this.loop = score.loop;
    this.drive = DRIVE / Math.max(loudest(score.notes), 1e-9);
    this.gain = PEAK / (PEAK_FRACTION * DRIVE);
    // The song's position, the next voice to start, and those sounding,
    // each with its own frame and phases.
    this.frame = 0;
    this.next = 0;
    this.sounding = [];
    this.stopped = false;
    this.port.onmessage = ({ data }) => {
      if (data === "stop") {
        this.stopped = true;
      }
    };
  }

  process(inputs, outputs) {
    if (this.stopped) {
      return false;
    }
    const channels = outputs[0];
    const out = channels[0];
    out.fill(0);
    // The block in parts, each up to where a note starts or the song ends.
    let at = 0;
    while (at < out.length) {
      if (this.frame >= this.frames) {
        if (this.loop) {
          // The song starts over, while the notes still fading sound on.
          this.frame = 0;
          this.next = 0;
        } else if (!this.sounding.length) {
          this.port.postMessage("ended");
          this.finish(out, at, channels);
          return false;
        }
      }
      while (this.next < this.voices.length && this.voices[this.next].start === this.frame) {
        const started = this.voices[this.next++];
        this.sounding.push({ voice: started, at: 0, phases: started.phases.slice() });
      }
      let end = out.length;
      if (this.frame < this.frames) {
        end = Math.min(end, at + this.frames - this.frame);
      }
      if (this.next < this.voices.length) {
        end = Math.min(end, at + this.voices[this.next].start - this.frame);
      }
      for (const playing of this.sounding) {
        this.play(playing, out, at, end);
      }
      this.sounding = this.sounding.filter((playing) => playing.at < playing.voice.frames);
      this.frame += end - at;
      at = end;
    }
    this.finish(out, out.length, channels);
    return true;
  }

  // Mixes the block's frames from to end through tanh, as the
  // synthesizer does, to all channels.
  finish(out, end, channels) {
    for (let i = 0; i < end; i++) {
      out[i] = this.gain * Math.tanh(out[i] * this.drive);
    }
    for (let c = 1; c < channels.length; c++) {
      channels[c].set(out);
    }
  }

  // Adds a sounding note to the out frames from to end, or until it ends.
  play(playing, out, from, end) {
    const { voice, phases } = playing;
    const count = Math.min(end - from, voice.frames - playing.at);
    const wave = this.wave(count);
    for (let k = 0; k < phases.length; k++) {
      const step = voice.steps[k];
      const level = voice.levels[k];
      let phase = phases[k];
      for (let j = 0; j < count; j++) {
        const index = phase * TABLE_SIZE;
        const i = index | 0;
        const a = TABLE[i];
        wave[j] += level * (a + (index - i) * (TABLE[i + 1] - a));
        phase += step;
        // A step is less than a cycle, also below 0 for a detuned sine
        // below 3 Hz.
        if (phase >= 1) {
          phase -= 1;
        } else if (phase < 0) {
          phase += 1;
        }
      }
      phases[k] = phase;
    }
    const { attack, release, gate, held } = voice;
    for (let j = 0; j < count; j++) {
      const at = playing.at + j;
      const envelope = at < gate ? Math.min(at / attack, 1) : Math.max(held - (at - gate) / release, 0);
      out[from + j] += envelope * wave[j];
    }
    playing.at += count;
  }

  // A buffer of silence for a note's frames of a block.
  wave(count) {
    if (!this.scratch || this.scratch.length < count) {
      this.scratch = new Float32Array(Math.max(count, 128));
    }
    const wave = this.scratch.subarray(0, count);
    wave.fill(0);
    return wave;
  }
}

registerProcessor("song", Song);
