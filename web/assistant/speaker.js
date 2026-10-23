// Plays the audio the service streams, see app.js: the speech of an answer,
// or the radio, its 16 bit samples, posted as they arrive, at the audio's
// own sample rate, resampled to the context's. It waits while the next
// sentence is synthesized, and fades out when stopped, rather than click.
//
// Options: the audio's sampleRate, its channels, interleaved, 1 by default,
// its volume, and prebuffer, for a live stream such as the radio: the
// seconds of audio it holds before it plays, also after it ran dry, so a
// slow network makes it pause rather than stutter.
//
// Messages: an ArrayBuffer of samples, whole frames of the channels, "end"
// after the last, and "stop". It posts "ended" once it has played them all,
// or faded out, and with prebuffer, "playing" when it starts to sound, and
// "waiting" when it runs dry.

// How long it fades out when stopped, in seconds.
const FADE_SECONDS = 0.02;

class Speaker extends AudioWorkletProcessor {
  constructor({ processorOptions }) {
    super();
    // Frames of the audio per frame of the context.
    this.step = processorOptions.sampleRate / sampleRate;
    this.volume = processorOptions.volume ?? 1;
    // The frames of each channel.
    this.samples = Array.from({ length: processorOptions.channels ?? 1 }, () => new Float32Array(1 << 16));
    this.length = 0;
    // Where it plays, in frames, fractional.
    this.position = 0;
    // Whether all the samples came.
    this.complete = false;
    // The gain left while fading out, once stopped.
    this.fade = null;
    // The frames it holds before it plays, and whether it is holding them.
    this.prebuffer = (processorOptions.prebuffer ?? 0) * processorOptions.sampleRate;
    this.buffering = this.prebuffer > 0;
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
    const channels = this.samples.length;
    const frames = samples.length / channels;
    // Drops what has played, but the frame before the position, which the
    // interpolation needs, once that is half the buffer or room is needed.
    // Moving what is left at every add copied the radio's backlog, the 20
    // seconds its host sends at once, 11 times a second.
    const played = Math.floor(this.position) - 1;
    const room = this.samples[0].length;
    if (played > 0 && (played > room / 2 || this.length + frames > room)) {
      for (const channel of this.samples) {
        channel.copyWithin(0, played, this.length);
      }
      this.length -= played;
      this.position -= played;
    }
    if (this.length + frames > this.samples[0].length) {
      let size = this.samples[0].length;
      while (size < this.length + frames) {
        size *= 2;
      }
      this.samples = this.samples.map((channel) => {
        const grown = new Float32Array(size);
        grown.set(channel.subarray(0, this.length));
        return grown;
      });
    }
    for (let c = 0; c < channels; c++) {
      const channel = this.samples[c];
      for (let i = 0; i < frames; i++) {
        channel[this.length + i] = samples[i * channels + c] / 32768;
      }
    }
    this.length += frames;
  }

  // The sample of the channel at i, silence before the first and after the
  // last.
  at(channel, i) {
    return i >= 0 && i < this.length ? channel[i] : 0;
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const fadeStep = 1 / (FADE_SECONDS * sampleRate);
    for (let n = 0; n < out[0].length; n++) {
      if (this.fade !== null && this.fade <= 0) {
        this.port.postMessage("ended");
        return false;
      }
      if (this.complete && this.position >= this.length) {
        this.port.postMessage("ended");
        return false;
      }
      if (this.buffering) {
        if (!this.complete && this.length - this.position < this.prebuffer) {
          if (this.fade !== null) {
            this.port.postMessage("ended");
            return false;
          }
          return true;
        }
        this.buffering = false;
        this.port.postMessage("playing");
      }
      const i = Math.floor(this.position);
      // Waits for the samples the interpolation needs, while more come,
      // in silence, so stopped it ends at once.
      if (!this.complete && i + 2 >= this.length) {
        if (this.fade !== null) {
          this.port.postMessage("ended");
          return false;
        }
        if (this.prebuffer > 0) {
          this.buffering = true;
          this.port.postMessage("waiting");
        }
        return true;
      }
      let gain = this.volume;
      if (this.fade !== null) {
        gain *= this.fade;
        this.fade -= fadeStep;
      }
      // Catmull-Rom interpolation of the four samples around the position.
      const t = this.position - i;
      for (let c = 0; c < out.length; c++) {
        // Mono on every channel of the output.
        const channel = this.samples[Math.min(c, this.samples.length - 1)];
        const p0 = this.at(channel, i - 1);
        const p1 = this.at(channel, i);
        const p2 = this.at(channel, i + 1);
        const p3 = this.at(channel, i + 2);
        const sample = p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
        out[c][n] = sample * gain;
      }
      this.position += this.step;
    }
    return true;
  }
}

registerProcessor("speaker", Speaker);
