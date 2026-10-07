// Turns the microphone's float samples into 16 bit chunks for the server,
// downsampling first if the audio context does not run at the model's rate.

class Recorder extends AudioWorkletProcessor {
  constructor({ processorOptions }) {
    super();
    this.ratio = sampleRate / processorOptions.targetRate;
    this.chunk = processorOptions.chunk;
    this.buffer = new Int16Array(this.chunk);
    this.length = 0;
    this.sum = 0;
    this.count = 0;
    this.position = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) {
      return true;
    }
    if (this.ratio === 1) {
      for (const sample of input) {
        this.write(sample);
      }
      return true;
    }
    // Average the input samples each output sample spans, a simple
    // low-pass filter for the downsampling.
    for (const sample of input) {
      this.sum += sample;
      this.count += 1;
      this.position += 1;
      if (this.position >= this.ratio) {
        this.write(this.sum / this.count);
        this.sum = 0;
        this.count = 0;
        this.position -= this.ratio;
      }
    }
    return true;
  }

  write(sample) {
    this.buffer[this.length++] = Math.max(-32768, Math.min(32767, Math.round(sample * 32768)));
    if (this.length === this.chunk) {
      this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
      this.buffer = new Int16Array(this.chunk);
      this.length = 0;
    }
  }
}

registerProcessor("recorder", Recorder);
