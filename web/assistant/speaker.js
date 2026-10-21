// Plays the speech the service streams, see app.js: its 16 bit samples,
// posted as they arrive, at the speech's own sample rate, resampled to the
// context's. It waits while the next sentence is synthesized, and fades out
// when stopped, rather than click.
//
// Messages: an ArrayBuffer of samples, "end" after the last, and "stop". It
// posts "ended" once it has played them all, or faded out.

// How long it fades out when stopped, in seconds.
const FADE_SECONDS = 0.02;

class Speaker extends AudioWorkletProcessor {
  constructor({ processorOptions }) {
    super();
    // Samples of the speech per sample of the context.
    this.step = processorOptions.sampleRate / sampleRate;
    this.volume = processorOptions.volume ?? 1;
    this.samples = new Float32Array(1 << 16);
    this.length = 0;
    // Where it plays, in samples, fractional.
    this.position = 0;
    // Whether all the samples came.
    this.complete = false;
    // The gain left while fading out, once stopped.
    this.fade = null;
    this.port.onmessage = ({ data }) => {
      if (data === "end") {
        this.complete = true;
      } else if (data === "stop") {
        this.fade ??= 1;
      } else {
        this.add(new Int16Array(data));
      }
    };
  }

  add(samples) {
    // Drops what has played, but the sample before the position, which
    // the interpolation needs.
    const played = Math.floor(this.position) - 1;
    if (played > 0) {
      this.samples.copyWithin(0, played, this.length);
      this.length -= played;
      this.position -= played;
    }
    if (this.length + samples.length > this.samples.length) {
      let size = this.samples.length;
      while (size < this.length + samples.length) {
        size *= 2;
      }
      const grown = new Float32Array(size);
      grown.set(this.samples.subarray(0, this.length));
      this.samples = grown;
    }
    for (let i = 0; i < samples.length; i++) {
      this.samples[this.length + i] = samples[i] / 32768;
    }
    this.length += samples.length;
  }

  // The sample at i, silence before the first and after the last.
  at(i) {
    return i >= 0 && i < this.length ? this.samples[i] : 0;
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    const fadeStep = 1 / (FADE_SECONDS * sampleRate);
    for (let n = 0; n < out.length; n++) {
      if (this.fade !== null && this.fade <= 0) {
        this.port.postMessage("ended");
        return false;
      }
      if (this.complete && this.position >= this.length) {
        this.port.postMessage("ended");
        return false;
      }
      const i = Math.floor(this.position);
      // Waits for the samples the interpolation needs, while more come,
      // in silence, so stopped it ends at once.
      if (!this.complete && i + 2 >= this.length) {
        if (this.fade !== null) {
          this.port.postMessage("ended");
          return false;
        }
        return true;
      }
      // Catmull-Rom interpolation of the four samples around the position.
      const t = this.position - i;
      const p0 = this.at(i - 1);
      const p1 = this.at(i);
      const p2 = this.at(i + 1);
      const p3 = this.at(i + 2);
      const sample = p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
      let gain = this.volume;
      if (this.fade !== null) {
        gain *= this.fade;
        this.fade -= fadeStep;
      }
      out[n] = sample * gain;
      this.position += this.step;
    }
    return true;
  }
}

registerProcessor("speaker", Speaker);
