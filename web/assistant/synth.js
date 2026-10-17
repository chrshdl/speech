// The voices of a SongCard's notes, live on the audio thread, as an
// AudioWorklet: from GET /v1/songs/{songId}, a note with a patch plays with
// the voice of a Roland D-50, see below, in stereo, and the others with that
// of the additive synthesizer, as the synthesizer's C++ engine plays its
// keys. Each harmonic of a note is a sine with two quieter ones detuned by
// 3 Hz plus 1 Hz per harmonic, which thickens it like a chorus, read from a
// table as the engine reads its wavetables, and each note fades in and out
// linearly, by default as fast as the engine's keys do, or as slowly as its
// voice says. The voices are mixed through tanh, as the engine mixes, so
// loud chords saturate softly.

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

// The voice of a Roland D-50, for a note with a patch: its linear
// arithmetic (LA) synthesis, from the patch's own parameters, the values the
// D-50 stores, see speech/backend/patches. A patch has two tones, upper and
// lower, each of two partials, which its key mode plays together (Dual), one
// (Whole) or by the key (Split).
//
// The partials are made as Roland's LA chip makes them, after Munt's
// emulation of it (github.com/munt/munt, mt32emu/src: LA32FloatWaveGenerator,
// LA32Ramp, TVA, TVF and TVP), measured on the MT-32, the D-50's smaller
// sibling with the same partials. The chip has no filter: a synth partial is
// a square wave whose edges are cosine segments, the narrower the higher the
// TVF's cutoff, with a decaying sine on each half for the resonance, and a
// sawtooth is that square times a cosine of its frequency, which is why a
// sawtooth at pulse width 0 sounds an octave higher than the square, as the
// D-50's manual says, and its init tone's sawtooth is at C3, its square at
// C4. A PCM partial plays a sample. The envelopes are the chip's ramps, in
// 1/16 octaves of amplitude and of cutoff, whose segments each take a time
// however far they go.
//
// Where the D-50 differs from the MT-32, it was measured on recordings of
// it: single notes of Fantasia (Andrey B. Jardim's "Roland D-50 Fantasia
// SF2", CC BY, musical-artifacts.com/artifacts/557) and its init tones
// (synthmania.com's waveform demos). Its envelope times are TIME_SCALE times
// longer, its cutoff CUTOFF_OFFSET brighter, its pulse widths all
// asymmetrical, and its LFO at rate 75 makes a 5 Hz vibrato, which starts
// 0.65 seconds after the key at delay 50.
//
// The D-50's own PCM samples are Roland's, so its waves here are stand-ins
// for those the patches use, see PCM_WAVES. Not measured and chosen by ear
// are the depths of the LFOs, the pitch envelope's times, and the D-50's
// equalizer, chorus and reverb, which follow the parameters' descriptions
// in its manual.

// The LA chip's sample rate, which Munt's measurements are in.
const LA_RATE = 32000;
// How hard a song's keys are played, of 127.
const VELOCITY = 100;
// The envelopes and LFOs move on every so many frames, about 3,000 times a
// second, as the D-50's processor moves them; the amplitude in between.
const CONTROL_FRAMES = 16;
// The pitch, in 1/4096 octaves, of an LA partial's middle C: a square's, and
// the wave of a sawtooth, which sounds an octave higher, see above.
const MIDDLE_C_PITCH = 37133;
const MAX_PITCH = 59392;
// How many times longer the D-50's envelope times are than the MT-32's, whose
// segments take about 4 ms * 2^(time / 8): Fantasia's synth swells to its
// level in about 0.2 seconds at time 36, and its bells, at time 69 from
// level 100 to 0, fade by 14 dB a second, measured. So time 50 is half a
// second, and 100 a minute and a half.
const TIME_SCALE = 1.26;
// How much brighter the D-50's cutoff is than the MT-32's, in 1/16 octaves:
// Fantasia's synth, at cutoff 52, has its second and third harmonics about
// 12 and 20 dB below the first.
const CUTOFF_OFFSET = 28;
// The amplitude of an LA partial at full level, of a pair's mix, as Munt
// scales the chip's samples.
const PAIR_LEVEL = 0.25;
// The voices of the patches are mixed through tanh, below LA_LIMIT, as loud
// as the client's tones, PEAK, for a few notes of Fantasia at once.
const LA_GAIN = 0.9;
const LA_LIMIT = 0.6;
// Below which a patch's effects have rung out, after its last note.
const SILENT = 1e-5;
// How deep the LFOs modulate at depth 100: the pitch in 1/4096 octaves, as
// the MT-32's vibrato, the pulse width of 255, the cutoff and the amplitude
// in their 1/16 octaves.
const VIBRATO_UNITS = 200;
const PW_LFO_UNITS = 128;
const CUTOFF_LFO_UNITS = 32;
const TREMOLO_UNITS = 16;

// The tables of Munt's TVP, TVF, TVA and Tables, which match the chip's.
// The multipliers of the pitch key follow, by 8192: -1, -1/2, -1/4, 0, 1/8,
// 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1, 5/4, 3/2, 2, s1 and s2.
const PITCH_KEYFOLLOW = [
  -8192, -4096, -2048, 0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 16384, 8198, 8226,
];
// The same, and the TVF's, by 21.
const CUTOFF_KEYFOLLOW = [-21, -10, -5, 0, 2, 5, 8, 10, 13, 16, 18, 21, 26, 32, 42, 21, 21];
const TVF_BIAS = [85, 42, 21, 16, 10, 5, 2, 0, -2, -5, -10, -16, -21, -74, -85];
const TVA_BIAS = [255, 187, 137, 100, 74, 54, 40, 29, 21, 15, 10, 5, 0];

// How much a level of 0 to 100 lowers the amplitude.
function levelToAmpSubtraction(level) {
  return Math.min(Math.floor((2 - Math.log10(level + 1)) * 128 + 1), 255);
}

// A D-50 envelope time as the MT-32's, of up to 127.
function scaledTime(time) {
  return Math.min(Math.round(time * TIME_SCALE), 127);
}

// The increment of a ramp that goes a distance in about the same time
// whatever the distance, less a time setting.
function envLogarithmicTime(distance) {
  return distance === 0 ? 64 : Math.ceil(64 + Math.log2(distance) * 8);
}

// An LA chip ramp, an envelope's: it goes to a target at a rate set by an
// increment of 7 bits, ascending or descending by its 8th, and is reached
// when it gets there. An increment of 0 holds it where it is.
class Ramp {
  constructor() {
    this.value = 0;
    this.target = 0;
    this.rate = 0;
    this.descending = false;
    this.reached = false;
  }

  start(target, increment) {
    const steps = increment & 0x7f;
    this.target = target;
    // In units per second.
    this.rate = steps === 0 ? 0 : (2 ** ((steps + 24) / 8) * LA_RATE) / 2 ** 18;
    this.descending = (increment & 0x80) !== 0;
    this.reached = false;
  }

  advance(seconds) {
    if (this.rate === 0 || this.reached) {
      return;
    }
    // Going the other way than to the target, it jumps there, as when
    // starting a note's envelope "as quickly as possible".
    if (this.descending) {
      this.value -= this.rate * seconds;
      if (this.value <= this.target) {
        this.value = this.target;
        this.reached = true;
      }
    } else {
      this.value += this.rate * seconds;
      if (this.value >= this.target) {
        this.value = this.target;
        this.reached = true;
      }
    }
  }
}

// The phases of the TVA and TVF envelopes: when one is the target, the
// level of the point before it is targeted, within its time.
const PHASE_BASIC = 0;
const PHASE_ATTACK = 1;
const PHASE_4 = 4;
const PHASE_SUSTAIN = 5;
const PHASE_RELEASE = 6;
const PHASE_DEAD = 7;

// A partial's TVA, its amplitude in 1/16 octaves, as Munt's TVA.
class TVA {
  constructor(p, key, velocity, gain) {
    this.p = p;
    this.times = [p.tva_t1, p.tva_t2, p.tva_t3, p.tva_t4, p.tva_t5].map(scaledTime);
    this.levels = [p.tva_l1, p.tva_l2, p.tva_l3, p.tva_sus];
    this.ramp = new Ramp();
    this.held = true;
    this.velocity = velocity;
    this.keyTimeSubtraction = p.tva_time_kf ? (key - 60) >> (5 - p.tva_time_kf) : 0;
    this.basic = basicAmp(p, key, velocity, gain);
    if (this.times[0] === 0) {
      this.phase = PHASE_ATTACK;
      this.startRamp(this.basic + this.levels[0], 0x80 | 127, PHASE_ATTACK);
    } else {
      this.startRamp(this.basic, 0x80 | 127, PHASE_BASIC);
    }
  }

  startRamp(target, increment, phase) {
    this.target = target;
    this.phase = phase;
    this.ramp.start(target, increment);
  }

  get dead() {
    return this.phase >= PHASE_DEAD;
  }

  release() {
    this.held = false;
    if (this.phase >= PHASE_RELEASE) {
      return;
    }
    this.startRamp(0, this.times[4] === 0 ? 1 : 0x80 | Math.max(128 - this.times[4], 1), PHASE_RELEASE);
  }

  advance(seconds) {
    this.ramp.advance(seconds);
    // A ramp of time 0 is reached at once, and the next starts.
    for (let i = 0; this.ramp.reached && i < 8 && !this.dead; i++) {
      this.nextPhase();
    }
  }

  nextPhase() {
    let phase = this.phase + 1;
    if (phase === PHASE_DEAD) {
      this.phase = phase;
      return;
    }
    const levels = this.levels;
    // Once the levels still to come are all 0, it goes to silence.
    let zero = false;
    if (levels[3] === 0) {
      zero =
        phase === PHASE_4 ||
        (levels[2] === 0 &&
          (phase === 3 || (levels[1] === 0 && (phase === 2 || (levels[0] === 0 && phase === PHASE_ATTACK)))));
    }
    let target = 0;
    let increment = 0;
    const index = this.phase;
    if (!zero) {
      target = this.basic;
      if (phase === PHASE_SUSTAIN || phase === PHASE_RELEASE) {
        if (levels[3] === 0) {
          this.phase = PHASE_DEAD;
          return;
        }
        if (!this.held) {
          phase = PHASE_RELEASE;
          target = 0;
          increment = this.times[4] === 0 ? 1 : 0x80 | Math.max(128 - this.times[4], 1);
        } else {
          target += levels[3];
        }
      } else {
        target += levels[index];
      }
    }
    if ((phase !== PHASE_SUSTAIN && phase !== PHASE_RELEASE) || zero) {
      let time = this.times[index];
      if (phase === PHASE_ATTACK) {
        time -= (this.velocity - 64) >> (6 - this.p.tva_velocity_t1);
        if (time <= 0 && this.times[index] !== 0) {
          time = 1;
        }
      } else {
        time -= this.keyTimeSubtraction;
      }
      if (time > 0) {
        let delta = target - this.target;
        if (delta <= 0) {
          if (delta === 0) {
            delta = -1;
            target--;
            if (target < 0) {
              delta = 1;
              target = -target;
            }
          }
          increment = Math.max(envLogarithmicTime(-delta & 0xff) - time, 1) | 0x80;
        } else {
          increment = Math.max(envLogarithmicTime(delta & 0xff) - time, 1);
        }
      } else {
        increment = target >= this.target ? 0x80 | 127 : 127;
      }
    }
    this.startRamp(Math.max(Math.min(target, 255), 0), increment, phase);
  }
}

// A partial's amplitude before its envelope, of 155, in 1/16 octaves.
function basicAmp(p, key, velocity, gain) {
  let amp = 155 - gain;
  let bias = 0;
  if ((p.tva_bias_point & 0x40) === 0) {
    bias = Math.max(p.tva_bias_point + 33 - key, 0);
  } else {
    bias = Math.max(key - (p.tva_bias_point - 31), 0);
  }
  amp -= Math.min((bias * TVA_BIAS[p.tva_bias_level]) >> 5, 255);
  amp -= levelToAmpSubtraction(p.tva_level);
  const sensitivity = p.tva_velocity - 50;
  amp -= Math.abs(sensitivity) - (((sensitivity * (velocity - 64)) << 2) >> 8);
  amp = Math.min(Math.max(amp, 0), 155);
  return Math.max(amp - (p.tvf_resonance >> 1), 0);
}

// A synth partial's TVF, the cutoff it adds to its base, in 1/16 octaves,
// as Munt's TVF, also to the D-50's end level once let go.
class TVF {
  constructor(p, key, velocity) {
    this.p = p;
    this.times = [p.tvf_t1, p.tvf_t2, p.tvf_t3, p.tvf_t4, p.tvf_t5].map(scaledTime);
    this.levels = [p.tvf_l1, p.tvf_l2, p.tvf_l3, p.tvf_sus];
    this.ramp = new Ramp();
    this.held = true;
    let mult = ((velocity * p.tvf_env_velocity) >> 6) + 109 - p.tvf_env_velocity;
    mult += (key - 60) >> (4 - p.tvf_env_depth_kf);
    this.levelMult = Math.min((Math.max(mult, 0) * p.tvf_env_depth) >> 6, 255);
    this.keyTimeSubtraction = p.tvf_env_time_kf ? (key - 60) >> (5 - p.tvf_env_time_kf) : 0;
    const target = (this.levelMult * this.levels[0]) >> 8;
    const time = this.times[0] - this.keyTimeSubtraction;
    this.target = target;
    this.phase = PHASE_ATTACK;
    this.ramp.start(target, time <= 0 ? 0x80 | 127 : Math.max(envLogarithmicTime(target) - time, 1));
  }

  startRamp(target, increment, phase) {
    this.target = target;
    this.phase = phase;
    this.ramp.start(target, increment);
  }

  release() {
    this.held = false;
    if (this.phase >= PHASE_RELEASE) {
      return;
    }
    const target = this.p.tvf_end ? (this.levelMult * 100) >> 8 : 0;
    if (this.times[4] === 0) {
      this.ramp.value = target;
      this.startRamp(target, 0, PHASE_RELEASE);
    } else {
      this.startRamp(target, Math.max(128 - this.times[4], 1) | (target < this.ramp.value ? 0x80 : 0), PHASE_RELEASE);
    }
  }

  advance(seconds) {
    this.ramp.advance(seconds);
    for (let i = 0; this.ramp.reached && i < 8 && this.phase < PHASE_SUSTAIN; i++) {
      this.nextPhase();
    }
  }

  nextPhase() {
    const phase = this.phase + 1;
    if (phase === PHASE_SUSTAIN) {
      if (!this.held) {
        this.phase = phase;
        this.release();
      } else {
        this.startRamp((this.levelMult * this.levels[3]) >> 8, 0, phase);
      }
      return;
    }
    const index = this.phase;
    const time = this.times[index] - this.keyTimeSubtraction;
    let target = (this.levelMult * this.levels[index]) >> 8;
    let increment;
    if (time > 0) {
      let delta = target - this.target;
      if (delta === 0) {
        if (target === 0) {
          delta = 1;
          target = 1;
        } else {
          delta = -1;
          target--;
        }
      }
      increment = Math.max(envLogarithmicTime(Math.abs(delta)) - time, 1) | (delta < 0 ? 0x80 : 0);
    } else {
      increment = target >= this.target ? 0x80 | 127 : 127;
    }
    this.startRamp(target, increment, phase);
  }
}

// A synth partial's base cutoff, of 255 in 1/16 octaves, whose middle 128
// is cutoff 50, as Munt's calcBaseCutoff.
function baseCutoff(p, basePitch, key) {
  let cutoff = CUTOFF_KEYFOLLOW[p.tvf_keyfollow] - CUTOFF_KEYFOLLOW[p.wg_keyfollow];
  cutoff *= key - 60;
  const point = p.tvf_bias_point;
  if ((point & 0x40) === 0) {
    const bias = point + 33 - key;
    if (bias > 0) {
      cutoff -= bias * TVF_BIAS[p.tvf_bias_level];
    }
  } else {
    const bias = point - 31 - key;
    if (bias < 0) {
      cutoff += bias * TVF_BIAS[p.tvf_bias_level];
    }
  }
  cutoff += ((p.tvf_cutoff + CUTOFF_OFFSET) << 4) - 800;
  if (cutoff >= 0) {
    // Not above the pitch.
    const over = (basePitch >> 4) + cutoff - 3584;
    if (over > 0) {
      cutoff -= over;
    }
  } else if (cutoff < -2048) {
    cutoff = -2048;
  }
  return Math.min((cutoff + 2056) >> 4, 255);
}

// The cutoff above which the cosines narrow, and resonance sounds, the
// highest, and above which the resonance no longer swells.
const MIDDLE_CUTOFF = 128;
const RESONANCE_CUTOFF = 144;
const MAX_CUTOFF = 240;
const RESONANCE_DECAY = [31, 16, 12, 8, 5, 3, 2, 1];

// A cycle of the cosine, and 2^-x for x up to EXP_OCTAVES, in tables, as the
// LA chip reads its log-sine and exponent tables.
const COS_STEPS = 4096;
const COS = new Float32Array(COS_STEPS + 1);
for (let i = 0; i <= COS_STEPS; i++) {
  COS[i] = Math.cos((2 * Math.PI * i) / COS_STEPS);
}
const EXP_OCTAVES = 24;
const EXP_STEPS = 256;
const EXP = new Float32Array(EXP_OCTAVES * EXP_STEPS + 1);
for (let i = 0; i < EXP.length; i++) {
  EXP[i] = 2 ** (-i / EXP_STEPS);
}

// The cosine of an angle in cycles.
function cos(cycles) {
  let x = (cycles - Math.floor(cycles)) * COS_STEPS;
  const i = x | 0;
  const a = COS[i];
  return a + (x - i) * (COS[i + 1] - a);
}

// 2^-x, 0 for x past the table.
function exp2Negative(x) {
  if (x >= EXP_OCTAVES) {
    return 0;
  }
  const at = x * EXP_STEPS;
  const i = at | 0;
  const a = EXP[i];
  return a + (at - i) * (EXP[i + 1] - a);
}

// The shape of a synth partial's wave, as Munt's LA32FloatWaveGenerator,
// for a wave of a length in frames, a cutoff of 255, a pulse width of 255
// and resonance 1 to 31: the length of its cosine segments, of its high
// part, and its resonance's amplitude and decay, which hold for a control
// tick, into shape.
function waveShape(shape, length, cutoff, pulseWidth, resonance, sawtooth) {
  cutoff = Math.min(cutoff, MAX_CUTOFF);
  let cosine = 0.5 * length;
  if (cutoff > MIDDLE_CUTOFF) {
    cosine *= 2 ** ((cutoff - MIDDLE_CUTOFF) / -16);
  }
  const pulse = (pulseWidth > 128 ? 2 ** ((128 - pulseWidth) / 64 - 1) : 0.5) * length;
  shape.length = length;
  shape.cosine = cosine;
  shape.high = Math.max(pulse - cosine, 0);
  shape.sawtooth = sawtooth;
  shape.attenuation = cutoff < MIDDLE_CUTOFF ? 2 ** (-0.125 * (MIDDLE_CUTOFF - cutoff)) : 1;
  shape.resonant = cutoff >= MIDDLE_CUTOFF;
  let amp = 2 ** (1 - (32 - resonance) / 4);
  if (cutoff < RESONANCE_CUTOFF) {
    amp *= Math.sin((Math.PI * (cutoff - MIDDLE_CUTOFF)) / 32);
  }
  shape.resonanceAmp = amp;
  shape.decay = RESONANCE_DECAY[resonance >> 2];
  return shape;
}

// A sample of a synth partial's wave at a position in it, of its shape:
// a square whose edges are cosine segments, below the middle cutoff the
// quieter, above it with a decaying sine on each half, times a cosine for
// a sawtooth.
function synthSample(position, shape) {
  const { length, cosine, high } = shape;
  // Starting in the middle of the first cosine segment.
  let relative = position + 0.5 * cosine;
  if (relative > length) {
    relative -= length;
  }
  let sample;
  if (relative < cosine) {
    sample = -cos(relative / (2 * cosine));
  } else if (relative < cosine + high) {
    sample = 1;
  } else if (relative < 2 * cosine + high) {
    sample = cos((relative - (cosine + high)) / (2 * cosine));
  } else {
    sample = -1;
  }
  if (!shape.resonant) {
    sample *= shape.attenuation;
  } else {
    let decay = shape.decay;
    let sign = 1;
    relative = position;
    if (!(relative < cosine + high)) {
      sign = -1;
      relative -= cosine + high;
      decay += 0.25;
    }
    // sin(pi x) = cos(pi x - pi / 2), in cycles.
    const sine = sign * cos(relative / (2 * cosine) - 0.25);
    let fade = exp2Negative(0.125 * decay * (relative / cosine));
    relative = position;
    if (!(position < length - 0.5 * cosine)) {
      relative -= length;
    } else if (!(position < high + 0.5 * cosine)) {
      relative -= cosine + high;
    }
    // Windows so the wave does not break where the sine starts and ends.
    if (relative < 0.5 * cosine) {
      const sync = cos(relative / (2 * cosine) - 0.25);
      fade *= relative < 0 ? sync * sync : sync;
    }
    sample += sine * shape.resonanceAmp * fade;
  }
  if (shape.sawtooth) {
    sample *= cos(position / length);
  }
  return sample;
}

// Stand-ins for the D-50's PCM waves, numbered 1 to 100 as on its panel,
// made rather than Roland's samples. Roland set a patch's coarse and fine
// pitch to its samples' own, which are not published, so root is the pitch
// a stand-in sounds at, in cents from the key, at C4 and fine 0, chosen so
// that the factory patches sound in tune.
const PCM_WAVES = {
  // Bells, struck once: in Fantasia at F#5 +27 cents, its first mode two
  // octaves and a fourth above the key, 5.34 times its frequency, measured.
  13: { root: 1073, make: bells },
  // Spectrum 2, a loop: in Fantasia at G2 +11 cents, an octave below the key.
  69: { root: 489, make: spectrum },
};

// mulberry32, for the waves and the random LFO to be the same each time.
function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A struck bell: the inharmonic modes of a bar, as measured in Fantasia,
// each with its level and the seconds it takes to fade to 1/e, at LA_RATE,
// its first mode at F7, where Fantasia plays it, so it lasts as long there,
// as a sample made at its pitch does.
function bells() {
  const modes = [
    [1, 1, 6],
    [2.68, 0.3, 3],
    [5.14, 0.04, 0.3],
  ];
  const frequency = 2793.83;
  const length = Math.round(6 * LA_RATE);
  // And a frame of silence after, to interpolate the last frame to.
  const data = new Float32Array(length + 1);
  const next = random(13);
  for (const [ratio, level, seconds] of modes) {
    const step = (2 * Math.PI * frequency * ratio) / LA_RATE;
    const phase = 2 * Math.PI * next();
    for (let i = 0; i < length; i++) {
      data[i] += level * Math.exp(-i / (seconds * LA_RATE)) * Math.sin(phase + step * i);
    }
  }
  return normalized({ data, frequency, length, loop: false });
}

// A spectrum, looped, as the D-50's Spectrum waves are: an airy band of
// partials close together, none at its pitch, as measured in Spectrum 1.
// Fantasia's sounds faintly from 2 to 6 times its pitch, at a fifth of a
// full sample's level, also measured. The loop is 8 cycles of its pitch, so
// its partials are an eighth of it apart, at random levels and phases, the
// same each time.
function spectrum() {
  const cycles = 8;
  const length = 2048 * cycles;
  const data = new Float32Array(length + 1);
  const next = random(69);
  for (let h = 2 * cycles; h <= 6 * cycles; h++) {
    const level = 0.5 + next();
    const phase = 2 * Math.PI * next();
    const step = (2 * Math.PI * h) / length;
    for (let i = 0; i <= length; i++) {
      data[i] += level * Math.sin(phase + step * i);
    }
  }
  return normalized({ data, frequency: (cycles * LA_RATE) / length, length, loop: true }, 0.2);
}

// The wave with its peak at the level, of a full sample's.
function normalized(wave, level = 1) {
  let peak = 0;
  for (const sample of wave.data) {
    peak = Math.max(peak, Math.abs(sample));
  }
  wave.data = wave.data.map((sample) => (level * sample) / peak);
  return wave;
}

const pcmWaves = new Map();
function pcmWave(number) {
  if (!pcmWaves.has(number)) {
    const stand = PCM_WAVES[number];
    pcmWaves.set(number, stand ? { ...stand.make(), root: stand.root } : null);
  }
  return pcmWaves.get(number);
}

// The rate of an LFO in Hz: 5 Hz at 75, measured, doubling every 8 as the
// MT-32's vibrato does.
function lfoHz(rate) {
  return 5 * 2 ** ((rate - 75) / 8);
}

// How many seconds after a key an LFO starts, for its delay, and then takes
// half as long again to swell: at 50, 0.65 seconds, measured.
function lfoDelaySeconds(delay) {
  return 2.6 * (delay / 100) ** 2;
}

// An LFO of a tone: triangle, sawtooth, square or random, from -1 to 1.
class LFO {
  constructor(wave, rate, seed) {
    this.wave = wave;
    this.hz = lfoHz(rate);
    this.phase = 0;
    this.next = random(seed);
    this.held = 0;
    this.value = 0;
  }

  reset() {
    this.phase = 0;
  }

  advance(seconds) {
    this.phase += this.hz * seconds;
    if (this.phase >= 1) {
      this.phase -= Math.floor(this.phase);
      this.held = 2 * this.next() - 1;
    }
    const p = this.phase;
    if (this.wave === 0) {
      this.value = p < 0.25 ? 4 * p : p < 0.75 ? 2 - 4 * p : 4 * p - 4;
    } else if (this.wave === 1) {
      this.value = 1 - 2 * p;
    } else if (this.wave === 2) {
      this.value = p < 0.5 ? 1 : -1;
    } else {
      this.value = this.held;
    }
  }
}

// The value of an LFO select, +1, -1, +2, -2, +3 or -3, of the LFOs' values.
function selected(values, select) {
  return (select & 1 ? -1 : 1) * values[select >> 1];
}

// A partial of a key: its pitch, wave, TVF and TVA.
class Partial {
  constructor(p, common, tone, key, cents, velocity, gain) {
    this.p = p;
    this.common = common;
    this.tone = tone;
    const pcm = tone.pcm(p);
    this.pcm = pcm ? pcmWave(p.wg_pcm + 1) : null;
    // A PCM wave without a stand-in is silent.
    this.silent = pcm && !this.pcm;
    let pitch = Math.round(((key - 60) * 4096) / 12) + (cents * 4096) / 1200;
    pitch = (pitch * PITCH_KEYFOLLOW[p.wg_keyfollow]) / 8192;
    pitch += ((p.wg_coarse - 36) * 4096) / 12 + ((p.wg_fine - 50) * 4096) / 1200;
    pitch += tone.pitchShift;
    pitch += MIDDLE_C_PITCH + ((this.pcm?.root ?? 0) * 4096) / 1200;
    this.basePitch = Math.min(Math.max(Math.round(pitch), 0), MAX_PITCH);
    this.tva = new TVA(p, key, velocity, gain);
    if (!pcm) {
      this.tvf = new TVF(p, key, velocity);
      this.baseCutoff = baseCutoff(p, this.basePitch, key);
      // The D-50's pulse widths are all asymmetrical but 0, the LA chip's
      // from 128 of 255 up.
      this.pulseWidth = (velocity - 64) * (p.pw_velocity - 7) + 128 + Math.round((p.pw * 127) / 100);
      this.resonance = p.tvf_resonance + 1;
      this.sawtooth = p.wg_waveform === 1;
    }
    // The wave's position, in frames of a synth wave, or of the sample.
    this.position = 0;
    this.frequency = 0;
    this.cutoff = 0;
    this.shape = {
      length: 1,
      cosine: 0.5,
      high: 0,
      sawtooth: false,
      attenuation: 1,
      resonant: false,
      resonanceAmp: 0,
      decay: 0,
    };
    this.amp = 0;
    this.step = 0;
  }

  get dead() {
    return this.silent || this.tva.dead || (this.pcm && !this.pcm.loop && this.position >= this.pcm.length);
  }

  release() {
    this.tva.release();
    this.tvf?.release();
  }

  // Moves the envelopes on by the frames, and sets where the amplitude
  // goes over them, with the pitch offset and the values of the LFOs.
  control(frames, rate, pitchOffset, lfos) {
    const p = this.p;
    const seconds = frames / rate;
    this.tva.advance(seconds);
    let pitch = this.basePitch + pitchOffset;
    if (p.wg_lfo_mode === 1 || p.wg_lfo_mode === 2) {
      const depth = (this.common.pitch_lfo_depth * VIBRATO_UNITS) / 100;
      pitch += (p.wg_lfo_mode === 1 ? 1 : -1) * depth * lfos[0];
    }
    pitch = Math.min(Math.max(pitch, 0), MAX_PITCH);
    const frequency = 2 ** (pitch / 4096 - 16) * LA_RATE;
    if (this.pcm || this.silent) {
      this.frequency = frequency;
    } else {
      // The position keeps its phase in a wave of another length.
      if (this.frequency) {
        this.position *= this.frequency / frequency;
      }
      this.frequency = frequency;
      this.tvf.advance(seconds);
      this.cutoff =
        this.baseCutoff +
        this.tvf.ramp.value +
        (selected(lfos, p.tvf_lfo_select) * p.tvf_lfo_depth * CUTOFF_LFO_UNITS) / 100;
      const pw = this.pulseWidth + (selected(lfos, p.pw_lfo_select) * p.pw_lfo_depth * PW_LFO_UNITS) / 100;
      waveShape(
        this.shape,
        rate / frequency,
        this.cutoff,
        Math.min(Math.max(pw, 0), 255),
        this.resonance,
        this.sawtooth,
      );
    }
    let units = this.tva.ramp.value + (selected(lfos, p.tva_lfo_select) * p.tva_lfo_depth * TREMOLO_UNITS) / 100;
    if (this.tva.dead) {
      units = 0;
    }
    const amp = units <= 0 ? 0 : 2 ** ((Math.min(units, 255) - 256) / 16);
    this.step = (amp - this.amp) / frames;
  }

  // Adds the frames of the partial to out, from to end, or sets them for
  // a ring modulator, by its level.
  render(out, from, end, rate, level, add) {
    if (this.dead) {
      if (!add) {
        out.fill(0, from, end);
      }
      return;
    }
    let amp = this.amp;
    const step = this.step;
    if (this.pcm) {
      const { data, length, loop } = this.pcm;
      const delta = ((this.frequency / this.pcm.frequency) * LA_RATE) / rate;
      let position = this.position;
      for (let i = from; i < end; i++) {
        let sample = 0;
        if (position < length || loop) {
          const index = position | 0;
          const a = data[index];
          sample = a + (position - index) * (data[index + 1] - a);
        }
        position += delta;
        if (loop && position >= length) {
          position -= length;
        }
        amp += step;
        out[i] = (add ? out[i] : 0) + level * amp * sample;
      }
      this.position = position;
    } else {
      const shape = this.shape;
      const length = shape.length;
      let position = this.position;
      for (let i = from; i < end; i++) {
        const sample = synthSample(position, shape);
        position += 1;
        if (position > length) {
          position -= length;
        }
        amp += step;
        out[i] = (add ? out[i] : 0) + level * amp * sample;
      }
      this.position = position;
    }
    this.amp = amp;
  }
}

// The structures 1 to 7: whether each partial is PCM, and whether the
// second ring-modulates the first, mixed with the first.
const STRUCTURES = [
  [false, false, false],
  [false, false, true],
  [true, false, false],
  [true, false, true],
  [false, true, true],
  [true, true, false],
  [true, true, true],
];

// The ring modulator's overflow, as Munt's.
function overflowed(sample) {
  return sample < -1 ? sample + 2 : sample > 1 ? sample - 2 : sample;
}

// A tone of a patch: its common parameters, LFOs, partials' balance, and
// equalizer and chorus, through which its keys sound.
class Tone {
  constructor(tone, keyShift, fineTune, level, rate, seed) {
    const common = tone.common;
    this.common = common;
    this.partials = [tone.partial1, tone.partial2];
    this.structure = STRUCTURES[common.structure];
    this.pitchShift = ((keyShift - 24) * 4096) / 12 + ((fineTune - 50) * 4096) / 1200;
    this.level = level;
    const balance = common.partial_balance;
    const mute = common.partial_mute;
    this.partialLevels = [mute & 1 ? Math.min((100 - balance) / 50, 1) : 0, mute & 2 ? Math.min(balance / 50, 1) : 0];
    this.lfos = [1, 2, 3].map((n) => new LFO(common[`lfo${n}_wave`], common[`lfo${n}_rate`], seed + n));
    this.equalizer = new Equalizer(common, rate);
    this.chorus = new Chorus(common, rate);
    this.held = 0;
  }

  pcm(p) {
    return this.structure[this.partials.indexOf(p)];
  }

  // A key starts: LFOs set to sync start over, also when none was held.
  keyOn() {
    this.lfos.forEach((lfo, n) => {
      const sync = this.common[`lfo${n + 1}_sync`];
      if (sync === 2 || (sync === 1 && this.held === 0)) {
        lfo.reset();
      }
    });
    this.held++;
  }

  keyOff() {
    this.held--;
  }

  advance(seconds) {
    for (const lfo of this.lfos) {
      lfo.advance(seconds);
    }
  }
}

// The pitch envelope of a key's tone, in 1/4096 octaves: levels -50 to +50
// are 1, 1.5 or 2 octaves at most, by its velocity range, and its times 0 to
// 50 as an envelope's to 100.
class PitchEnvelope {
  constructor(common, key) {
    const range = [1, 1.5, 2][common.penv_velocity] * 4096;
    const level = (value) => ((value - 50) / 50) * range;
    const subtraction = common.penv_time_kf ? (key - 60) >> (5 - common.penv_time_kf) : 0;
    const seconds = (time) => (time <= 0 ? 0 : 0.004 * 2 ** (Math.max(2 * time - subtraction, 0) / 8));
    this.points = [
      [level(common.penv_l1), seconds(common.penv_t1)],
      [level(common.penv_l2), seconds(common.penv_t2)],
      [level(common.penv_sus), seconds(common.penv_t3)],
    ];
    this.end = [level(common.penv_end), seconds(common.penv_t4)];
    this.value = level(common.penv_l0);
    this.from = this.value;
    this.index = 0;
    this.time = 0;
    this.released = false;
  }

  release() {
    this.released = true;
    this.from = this.value;
    this.points = [this.end];
    this.index = 0;
    this.time = 0;
  }

  advance(seconds) {
    while (this.index < this.points.length) {
      const [target, duration] = this.points[this.index];
      this.time += seconds;
      if (this.time < duration) {
        this.value = this.from + (target - this.from) * (this.time / duration);
        return;
      }
      seconds = this.time - duration;
      this.value = this.from = target;
      this.time = 0;
      this.index++;
    }
  }
}

// A key of a patch: the partials of its tones, until they have all faded.
class Key {
  constructor(patch, tones, frequency, volume, gate, rate) {
    const exact = 69 + 12 * Math.log2(frequency / 440);
    const key = Math.round(exact);
    const cents = 100 * (exact - key);
    this.gate = gate;
    this.at = 0;
    this.tick = 0;
    this.tones = tones.map((tone) => {
      tone.keyOn();
      const partials = tone.partials.map((p, n) =>
        tone.partialLevels[n] > 0 || tone.structure[2]
          ? new Partial(p, tone.common, tone, key, cents, VELOCITY, patch.gain)
          : null,
      );
      return {
        tone,
        partials,
        levels: tone.partialLevels.map((level) => level * volume * tone.level * PAIR_LEVEL),
        pitch: new PitchEnvelope(tone.common, key),
        // The LFO set to start with each key, LFO 1's, the key's own.
        lfos:
          tone.common.lfo1_sync === 2
            ? [new LFO(tone.common.lfo1_wave, tone.common.lfo1_rate, key), ...tone.lfos.slice(1)]
            : tone.lfos,
        delays: [1, 2, 3].map((n) => lfoDelaySeconds(tone.common[`lfo${n}_delay`])),
        values: new Float64Array(3),
      };
    });
    this.released = false;
    this.ring = null;
  }

  get done() {
    return this.tones.every(({ partials }) => partials.every((partial) => !partial || partial.dead));
  }

  release() {
    this.released = true;
    for (const { tone, partials, pitch } of this.tones) {
      tone.keyOff();
      pitch.release();
      for (const partial of partials) {
        partial?.release();
      }
    }
  }

  // Adds the key's frames from to end to its tones' buses.
  render(buses, from, end, rate) {
    let at = from;
    while (at < end) {
      if (this.tick === 0) {
        if (!this.released && this.at >= this.gate) {
          this.release();
        }
        this.control(rate);
        this.tick = CONTROL_FRAMES;
      }
      const stop = Math.min(end, at + this.tick);
      for (let n = 0; n < this.tones.length; n++) {
        this.renderTone(this.tones[n], buses[n], at, stop, rate);
      }
      this.tick -= stop - at;
      this.at += stop - at;
      at = stop;
    }
  }

  control(rate) {
    const seconds = CONTROL_FRAMES / rate;
    const elapsed = this.at / rate;
    for (const { tone, partials, pitch, lfos, delays, values } of this.tones) {
      pitch.advance(seconds);
      if (lfos !== tone.lfos) {
        lfos[0].advance(seconds);
      }
      // Each LFO starts after its delay, and swells for half as long.
      for (let n = 0; n < 3; n++) {
        const delay = delays[n];
        values[n] = lfos[n].value * (delay === 0 ? 1 : Math.min(Math.max((elapsed - delay) / (0.5 * delay), 0), 1));
      }
      for (const partial of partials) {
        if (partial) {
          const mode = partial.p.wg_penv_mode;
          const offset = mode === 1 ? pitch.value : mode === 2 ? -pitch.value : 0;
          partial.control(CONTROL_FRAMES, rate, offset, values);
        }
      }
    }
  }

  renderTone({ tone, partials, levels }, bus, from, end, rate) {
    const [first, second] = partials;
    const ring = tone.structure[2];
    if (!ring) {
      if (first && levels[0] > 0) {
        first.render(bus, from, end, rate, levels[0], true);
      }
      if (second && levels[1] > 0) {
        second.render(bus, from, end, rate, levels[1], true);
      }
      return;
    }
    // Partial 1 mixed with the ring modulation of both.
    if (!this.ring || this.ring[0].length < end) {
      this.ring = [new Float32Array(Math.max(end, 128)), new Float32Array(Math.max(end, 128))];
    }
    const [a, b] = this.ring;
    first.render(a, from, end, rate, 1, false);
    second.render(b, from, end, rate, 1, false);
    for (let i = from; i < end; i++) {
      bus[i] += levels[0] * a[i] + levels[1] * overflowed(a[i]) * overflowed(b[i]);
    }
  }
}

// A patch: its tones, keys sounding, and effects, mixed to stereo.
class Patch {
  constructor(data, rate) {
    const { patch } = data;
    this.rate = rate;
    this.mode = patch.key_mode;
    this.split = 36 + patch.split_point;
    // Tone balance 100 is the upper tone alone, 0 the lower.
    const balance = patch.tone_balance;
    this.tones = [
      new Tone(data.upper, patch.upper_key_shift, patch.upper_fine, Math.min(balance / 50, 1), rate, 1),
      new Tone(data.lower, patch.lower_key_shift, patch.lower_fine, Math.min((100 - balance) / 50, 1), rate, 4),
    ];
    this.gain = levelToAmpSubtraction(patch.total_volume);
    this.reverb = new Reverb(patch.reverb_type, patch.reverb_balance, rate);
    // Which tones the reverb has, by output mode 1 to 4.
    this.reverbed = [
      [true, true],
      [true, true],
      [true, false],
      [false, true],
    ][patch.output_mode];
    this.keys = [];
    this.buses = [new Float32Array(128), new Float32Array(128)];
    // The block's reverb's left and right, and the patch's own.
    this.mixes = [0, 1, 2, 3].map(() => new Float32Array(128));
    this.quiet = true;
  }

  // The tones a key plays: in Whole mode the upper's, in Split mode the
  // upper's from the split point up and the lower's below, and both else.
  play(frequency, volume, gate) {
    const key = 69 + 12 * Math.log2(frequency / 440);
    let tones = this.tones;
    if (this.mode === 0 || this.mode === 4) {
      tones = [this.tones[0]];
    } else if (this.mode === 2 || this.mode === 6 || this.mode === 7) {
      tones = [key >= this.split ? this.tones[0] : this.tones[1]];
    }
    const playing = new Key(this, tones, frequency, volume, gate, this.rate);
    playing.buses = tones.map((tone) => this.tones.indexOf(tone));
    this.keys.push(playing);
    this.quiet = false;
  }

  get sounding() {
    return this.keys.length > 0;
  }

  // Adds the keys' frames from to end to the tones' buses.
  render(from, end) {
    for (const key of this.keys) {
      key.render(
        key.buses.map((n) => this.buses[n]),
        from,
        end,
        this.rate,
      );
    }
    this.keys = this.keys.filter((key) => !key.done);
  }

  // Adds the block's buses through the effects to left and right, and
  // clears them for the next.
  mix(left, right, frames) {
    const seconds = frames / this.rate;
    const [wetLeft, wetRight, dryLeft, dryRight] = this.mixes.map((channel) => channel.subarray(0, frames).fill(0));
    this.tones.forEach((tone, n) => {
      tone.advance(seconds);
      const bus = this.buses[n].subarray(0, frames);
      tone.equalizer.process(bus);
      const [l, r] = tone.chorus.process(bus);
      const reverbed = this.reverbed[n];
      for (let i = 0; i < frames; i++) {
        dryLeft[i] += l[i];
        dryRight[i] += r[i];
        if (reverbed) {
          wetLeft[i] += l[i];
          wetRight[i] += r[i];
        }
      }
      bus.fill(0);
    });
    this.reverb.process(wetLeft, wetRight, dryLeft, dryRight);
    let peak = 0;
    for (let i = 0; i < frames; i++) {
      left[i] += dryLeft[i];
      right[i] += dryRight[i];
      peak = Math.max(peak, Math.abs(dryLeft[i]), Math.abs(dryRight[i]));
    }
    this.quiet = !this.sounding && peak < SILENT;
  }
}

// Biquads of the cookbook's (Robert Bristow-Johnson's), for the equalizer.
class Biquad {
  constructor([b0, b1, b2, a0, a1, a2]) {
    this.b = [b0 / a0, b1 / a0, b2 / a0];
    this.a = [a1 / a0, a2 / a0];
    this.x = [0, 0];
    this.y = [0, 0];
  }

  process(samples) {
    const [b0, b1, b2] = this.b;
    const [a1, a2] = this.a;
    let [x1, x2] = this.x;
    let [y1, y2] = this.y;
    for (let i = 0; i < samples.length; i++) {
      const x = samples[i];
      const y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1;
      x1 = x;
      y2 = y1;
      y1 = y;
      samples[i] = y;
    }
    this.x = [x1, x2];
    this.y = [y1, y2];
  }
}

const EQ_LOW = [63, 75, 88, 105, 125, 150, 175, 210, 250, 300, 350, 420, 500, 600, 700, 840];
const EQ_HIGH = [
  250, 300, 350, 420, 500, 600, 700, 840, 1000, 1200, 1400, 1700, 2000, 2400, 2800, 3400, 4000, 4800, 5700, 6700, 8000,
  9500,
];
const EQ_Q = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.2, 6.0];

// A tone's equalizer: a shelf of the lows and a peak of the highs.
class Equalizer {
  constructor(common, rate) {
    this.filters = [];
    const low = common.eq_lg - 12;
    if (low !== 0) {
      const A = 10 ** (low / 40);
      const w = (2 * Math.PI * EQ_LOW[common.eq_lf]) / rate;
      const alpha = (Math.sin(w) / 2) * Math.SQRT2;
      const cos = Math.cos(w);
      const root = 2 * Math.sqrt(A) * alpha;
      this.filters.push(
        new Biquad([
          A * (A + 1 - (A - 1) * cos + root),
          2 * A * (A - 1 - (A + 1) * cos),
          A * (A + 1 - (A - 1) * cos - root),
          A + 1 + (A - 1) * cos + root,
          -2 * (A - 1 + (A + 1) * cos),
          A + 1 + (A - 1) * cos - root,
        ]),
      );
    }
    const high = common.eq_hg - 12;
    if (high !== 0) {
      const A = 10 ** (high / 40);
      const w = (2 * Math.PI * Math.min(EQ_HIGH[common.eq_hf], 0.45 * rate)) / rate;
      const alpha = Math.sin(w) / (2 * EQ_Q[common.eq_hq]);
      const cos = Math.cos(w);
      this.filters.push(new Biquad([1 + alpha * A, -2 * cos, 1 - alpha * A, 1 + alpha / A, -2 * cos, 1 - alpha / A]));
    }
  }

  process(samples) {
    for (const filter of this.filters) {
      filter.process(samples);
    }
  }
}

// A tone's chorus, mono to stereo: the sound delayed by a few milliseconds,
// swept by a sine, the left's a quarter of a cycle before the right's, so
// their mix, as from a mono speaker, does not pump, mixed with it by the
// balance, 50 as loud as each other. Chorus 2 sweeps a longer delay.
class Chorus {
  constructor(common, rate) {
    this.rate = rate;
    this.delay = ((common.chorus_type === 1 ? 12 : 7) / 1000) * rate;
    this.depth = ((common.chorus_depth / 100) * 2 * rate) / 1000;
    this.hz = 0.1 * 2 ** (common.chorus_rate / 15);
    const balance = common.chorus_balance / 100;
    this.dry = Math.min(2 * (1 - balance), 1);
    this.wet = Math.min(2 * balance, 1);
    this.line = new Float32Array(2 ** Math.ceil(Math.log2(this.delay + this.depth + 4)));
    this.mask = this.line.length - 1;
    this.write = 0;
    this.phase = 0;
    this.out = [new Float32Array(128), new Float32Array(128)];
  }

  process(samples) {
    const frames = samples.length;
    if (this.out[0].length < frames) {
      this.out = [new Float32Array(frames), new Float32Array(frames)];
    }
    const [left, right] = this.out;
    const step = this.hz / this.rate;
    for (let i = 0; i < frames; i++) {
      this.line[this.write] = samples[i];
      const sweep = this.depth * cos(this.phase - 0.25);
      left[i] = this.dry * samples[i] + this.wet * this.tap(this.delay + sweep);
      right[i] = this.dry * samples[i] + this.wet * this.tap(this.delay + this.depth * cos(this.phase));
      this.write = (this.write + 1) & this.mask;
      this.phase += step;
    }
    this.phase -= Math.floor(this.phase);
    return [left.subarray(0, frames), right.subarray(0, frames)];
  }

  tap(delay) {
    const at = this.write - delay + this.line.length;
    const index = Math.floor(at);
    const a = this.line[index & this.mask];
    return a + (at - index) * (this.line[(index + 1) & this.mask] - a);
  }
}

// The reverb types 1 to 10, halls and rooms: their size, against the
// delays below, seconds to fade by 60 dB, and pre-delay in seconds.
const REVERBS = [
  [0.8, 1.6, 0.015], // Small Hall
  [1.0, 2.2, 0.022], // Medium Hall
  [1.3, 3.0, 0.03], // Large Hall
  [1.2, 3.5, 0.025], // Chapel
  [0.3, 0.4, 0.003], // Box
  [0.4, 1.0, 0.005], // Small Metal Room
  [0.4, 0.6, 0.005], // Small Room
  [0.6, 0.9, 0.008], // Medium Room
  [0.8, 1.2, 0.01], // Medium Large Room
  [1.0, 1.6, 0.012], // Large Room
];
// The delays of the reverb's network, in seconds, mutually prime in frames.
const REVERB_DELAYS = [0.0297, 0.0371, 0.0411, 0.0437, 0.0533, 0.0599, 0.0677, 0.0731];

const HADAMARD_SCALE = 1 / Math.sqrt(8);

// A reverb: a feedback delay network of eight delays mixed by a Hadamard
// matrix, each damped above a few kHz, after a pre-delay, added to the
// sound by the balance, 50 as loud as each other.
class Reverb {
  constructor(type, balance, rate) {
    const [size, seconds, predelay] = REVERBS[type] ?? REVERBS[1];
    this.lines = REVERB_DELAYS.map((delay) => new Float32Array(Math.round(delay * size * rate) | 1));
    this.gains = this.lines.map((line) => 10 ** ((-3 * line.length) / (seconds * rate)));
    this.positions = this.lines.map(() => 0);
    this.damped = this.lines.map(() => 0);
    this.damping = Math.exp((-2 * Math.PI * 5000) / rate);
    this.pre = new Float32Array(Math.max(Math.round(predelay * rate), 1));
    this.prePosition = 0;
    this.dry = Math.min(2 * (1 - balance / 100), 1);
    this.wet = Math.min((2 * balance) / 100, 1) * 0.35;
    this.values = new Float32Array(8);
  }

  // Adds the reverb of the wet left and right to out, which has the dry.
  process(wetLeft, wetRight, left, right) {
    const { lines, gains, positions, damped, values, pre } = this;
    for (let i = 0; i < left.length; i++) {
      left[i] *= this.dry;
      right[i] *= this.dry;
      const p = this.prePosition;
      const input = pre[p];
      pre[p] = 0.5 * (wetLeft[i] + wetRight[i]);
      this.prePosition = p + 1 === pre.length ? 0 : p + 1;
      for (let n = 0; n < 8; n++) {
        const line = lines[n];
        damped[n] = line[positions[n]] + this.damping * (damped[n] - line[positions[n]]);
        values[n] = damped[n];
      }
      // A fast Walsh-Hadamard transform, scaled to keep the energy.
      for (let size = 1; size < 8; size *= 2) {
        for (let start = 0; start < 8; start += 2 * size) {
          for (let k = start; k < start + size; k++) {
            const a = values[k];
            const b = values[k + size];
            values[k] = a + b;
            values[k + size] = a - b;
          }
        }
      }
      let l = 0;
      let r = 0;
      for (let n = 0; n < 8; n++) {
        const line = lines[n];
        line[positions[n]] = gains[n] * (values[n] * HADAMARD_SCALE + (n & 1 ? -input : input));
        positions[n] = positions[n] + 1 === line.length ? 0 : positions[n] + 1;
        if (n & 1) {
          r += damped[n];
        } else {
          l += damped[n];
        }
      }
      left[i] += this.wet * l;
      right[i] += this.wet * r;
    }
  }
}

class Song extends AudioWorkletProcessor {
  constructor({ processorOptions: score }) {
    super();
    // A note with a patch plays with the D-50's voice, of its patch's keys;
    // the others with the additive synthesizer's.
    this.patches = new Map(
      Object.entries(score.patches ?? {}).map(([name, data]) => [name, new Patch(data, sampleRate)]),
    );
    this.voices = score.notes
      .map((note) =>
        note.patch
          ? { start: Math.round(note.start * sampleRate), note, patch: this.patches.get(note.patch) }
          : voice(note, sampleRate),
      )
      .sort((a, b) => a.start - b.start);
    this.frames = Math.round(score.seconds * sampleRate);
    this.loop = score.loop;
    this.drive = DRIVE / Math.max(loudest(score.notes.filter((note) => !note.patch)), 1e-9);
    this.gain = PEAK / (PEAK_FRACTION * DRIVE);
    // The song's position, the next voice to start, and those sounding,
    // each with its own frame and phases.
    this.frame = 0;
    this.next = 0;
    this.sounding = [];
    this.left = new Float32Array(128);
    this.right = new Float32Array(128);
    this.stopped = false;
    this.port.onmessage = ({ data }) => {
      if (data === "stop") {
        this.stopped = true;
      }
    };
  }

  // Whether a patch's key, or its effects, still sound, as of the last block.
  get lasting() {
    for (const patch of this.patches.values()) {
      if (patch.sounding || !patch.quiet) {
        return true;
      }
    }
    return false;
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
        } else if (!this.sounding.length && !this.lasting) {
          this.port.postMessage("ended");
          this.finish(out, at, channels);
          return false;
        }
      }
      while (this.next < this.voices.length && this.voices[this.next].start === this.frame) {
        const started = this.voices[this.next++];
        if (started.patch) {
          const { frequency, volume, seconds } = started.note;
          started.patch.play(frequency, volume, Math.round(seconds * sampleRate));
        } else {
          this.sounding.push({ voice: started, at: 0, phases: started.phases.slice() });
        }
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
      for (const patch of this.patches.values()) {
        patch.render(at, end);
      }
      this.frame += end - at;
      at = end;
    }
    this.finish(out, out.length, channels);
    return true;
  }

  // Mixes the block's frames from to end through tanh, as the
  // synthesizer does, to all channels, with the patches' in stereo.
  finish(out, end, channels) {
    for (let i = 0; i < end; i++) {
      out[i] = this.gain * Math.tanh(out[i] * this.drive);
    }
    for (let c = 1; c < channels.length; c++) {
      channels[c].set(out);
    }
    if (!this.patches.size) {
      return;
    }
    const left = this.left.subarray(0, out.length).fill(0);
    const right = this.right.subarray(0, out.length).fill(0);
    for (const patch of this.patches.values()) {
      patch.mix(left, right, out.length);
    }
    const limit = (sample) => LA_LIMIT * Math.tanh((LA_GAIN * sample) / LA_LIMIT);
    if (channels.length === 1) {
      for (let i = 0; i < end; i++) {
        out[i] += limit(0.5 * (left[i] + right[i]));
      }
      return;
    }
    for (let i = 0; i < end; i++) {
      channels[0][i] += limit(left[i]);
      channels[1][i] += limit(right[i]);
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
