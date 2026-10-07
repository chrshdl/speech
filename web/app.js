// Streams microphone or file audio to the server and shows the transcript
// as it grows.

const $ = (id) => document.getElementById(id);
const ui = {
  model: $("model"),
  lm: $("lm"),
  status: $("status"),
  stepRecord: $("step-record"),
  stepSay: $("step-say"),
  partial: $("partial"),
  wave: $("wave"),
  record: $("record"),
  recordLabel: $("record-label"),
  file: $("file"),
  lines: $("lines"),
  clear: $("clear"),
};

// Audio is sent in chunks of 100 ms.
const CHUNK_SECONDS = 0.1;

let socket = null;
let pending = null; // The start request waiting for the server to be ready.
let audio = null; // The running microphone or file stream.
let state = "idle";
let speed = null; // How many times faster than real time decoding runs.

// Decoded 16 bit audio is exactly x / 32768, so this restores its samples.
function toInt16(sample) {
  return Math.max(-32768, Math.min(32767, Math.round(sample * 32768)));
}

function setStatus(text, error = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle("error", error);
}

function setState(next) {
  state = next;
  document.body.dataset.state = next;
  const busy = next !== "idle";
  ui.recordLabel.textContent = next === "recording" ? "Pause" : "Record";
  ui.record.disabled = next === "starting" || next === "stopping";
  ui.model.disabled = busy;
  ui.lm.disabled = busy;
  ui.file.disabled = busy;
  ui.stepRecord.classList.toggle("active", !busy);
  ui.stepSay.classList.toggle("active", busy);
}

function shortName(path) {
  // "examples/librispeech/models/ctc_streaming_100h" -> "librispeech / ctc_streaming_100h"
  const parts = path.split("/").filter((part) => part !== "models" && part !== "examples");
  return parts.join(" / ");
}

async function loadModels() {
  const { models, lms } = await (await fetch("/models")).json();
  ui.model.replaceChildren(...models.map((path) => new Option(shortName(path), path)));
  ui.lm.replaceChildren(
    new Option("None", ""),
    ...lms.map((path) => new Option(shortName(path).replace(/\.npz$/, ""), path)),
  );
  const model = models.find((path) => path.endsWith("ctc_streaming_100h"));
  if (model) {
    ui.model.value = model;
  }
  const lm = lms.find((path) => path.endsWith("lm-librispeech.npz"));
  if (lm) {
    ui.lm.value = lm;
  }
  if (models.length === 0) {
    setStatus("No trained models found. Train one, or start the server with --models.", true);
    ui.record.disabled = true;
    ui.file.disabled = true;
  }
}

function addLine(text) {
  const item = document.createElement("li");
  const content = document.createElement("span");
  content.textContent = text;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "delete";
  remove.title = "Delete";
  remove.setAttribute("aria-label", `Delete "${text}"`);
  remove.innerHTML =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';
  remove.addEventListener("click", () => deleteLine(item));
  item.append(content, remove);
  ui.lines.prepend(item);
}

function deleteLine(item) {
  // Keep the keyboard focus in the list: on the next line, else the one
  // before.
  const neighbor = item.nextElementSibling ?? item.previousElementSibling;
  const hadFocus = item.contains(document.activeElement);
  item.remove();
  if (hadFocus) {
    neighbor?.querySelector(".delete").focus();
  }
}

function onMessage(event) {
  const message = JSON.parse(event.data);
  if (message.type === "ready") {
    pending?.resolve(message);
  } else if (message.type === "partial") {
    ui.partial.textContent = message.text;
    if (message.rtf) {
      speed = (1 / message.rtf).toFixed(0);
      setStatus(`Decoding ${speed}× faster than real time`);
    }
  } else if (message.type === "line") {
    addLine(message.text);
  } else if (message.type === "done") {
    setState("idle");
    setStatus(speed ? `Done, decoded ${speed}× faster than real time` : "Done");
  } else if (message.type === "error") {
    setStatus(message.message, true);
    if (pending) {
      pending.reject(new Error(message.message));
    } else {
      stopAudio();
      setState("idle");
    }
  }
}

function connect() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${location.host}/stream`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => resolve();
    socket.onerror = () => reject(new Error("Cannot reach the server. Is serve.py running?"));
    socket.onmessage = onMessage;
    socket.onclose = () => {
      if (state !== "idle") {
        stopAudio();
        setState("idle");
        setStatus("The connection to the server closed.", true);
      }
    };
  });
}

// Asks the server to load the model and returns its sample rate.
async function startStream() {
  await connect();
  const ready = new Promise((resolve, reject) => {
    pending = { resolve, reject };
  });
  socket.send(JSON.stringify({ type: "start", model: ui.model.value, lm: ui.lm.value || null }));
  try {
    return (await ready).sample_rate;
  } finally {
    pending = null;
  }
}

function send(samples) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(samples.buffer);
  }
}

async function startMicrophone(rate) {
  const stream = await navigator.mediaDevices.getUserMedia({
    // The models were trained on unprocessed recordings.
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  let context = new AudioContext({ sampleRate: rate });
  let source;
  try {
    source = context.createMediaStreamSource(stream);
  } catch {
    // Firefox only captures at the device's rate, so the worklet resamples.
    await context.close();
    context = new AudioContext();
    source = context.createMediaStreamSource(stream);
  }
  await context.audioWorklet.addModule("/recorder.js");
  const recorder = new AudioWorkletNode(context, "recorder", {
    processorOptions: { targetRate: rate, chunk: Math.round(rate * CHUNK_SECONDS) },
  });
  recorder.port.onmessage = (event) => send(new Int16Array(event.data));
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  source.connect(recorder);
  // The worklet outputs silence, but it only runs while connected.
  recorder.connect(context.destination);
  const samples = new Float32Array(analyser.fftSize);
  audio = {
    waveform() {
      analyser.getFloatTimeDomainData(samples);
      return samples;
    },
    stop() {
      stream.getTracks().forEach((track) => track.stop());
      context.close();
    },
  };
}

async function decodeFile(file, rate) {
  const data = await file.arrayBuffer();
  // Decoding resamples to the context's rate, rendering mixes to mono.
  const decoded = await new OfflineAudioContext(1, 1, rate).decodeAudioData(data);
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * rate), rate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  return (await offline.startRendering()).getChannelData(0);
}

async function startFile(file, rate) {
  const samples = await decodeFile(file, rate);
  const chunk = Math.round(rate * CHUNK_SECONDS);
  let position = 0;
  let current = new Float32Array(chunk);
  // Send the file at real-time pace, as a microphone would.
  const timer = setInterval(() => {
    if (position >= samples.length) {
      stop();
      return;
    }
    current = samples.subarray(position, position + chunk);
    const pcm = new Int16Array(current.length);
    for (let i = 0; i < current.length; i++) {
      pcm[i] = toInt16(current[i]);
    }
    send(pcm);
    position += chunk;
  }, CHUNK_SECONDS * 1000);
  audio = {
    waveform: () => current,
    stop: () => clearInterval(timer),
  };
}

function stopAudio() {
  audio?.stop();
  audio = null;
}

async function start(file = null) {
  setState("starting");
  setStatus(file ? `Loading the model for ${file.name}…` : "Loading the model…");
  ui.partial.textContent = "";
  speed = null;
  try {
    const rate = await startStream();
    if (file) {
      await startFile(file, rate);
    } else {
      await startMicrophone(rate);
    }
    setStatus(file ? `Transcribing ${file.name}` : "Listening");
    setState("recording");
  } catch (error) {
    stopAudio();
    setState("idle");
    const denied = error.name === "NotAllowedError";
    setStatus(denied ? "Microphone access was denied. Allow it in the browser to record." : error.message, true);
  }
}

function stop() {
  if (state !== "recording") {
    return;
  }
  stopAudio();
  setState("stopping");
  socket.send(JSON.stringify({ type: "stop" }));
}

function drawWave() {
  const canvas = ui.wave;
  const scale = window.devicePixelRatio || 1;
  const width = canvas.clientWidth * scale;
  const height = canvas.clientHeight * scale;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.strokeStyle = getComputedStyle(canvas).getPropertyValue("--wave-line");
  context.lineWidth = 2 * scale;
  context.beginPath();
  const samples = audio?.waveform();
  const middle = height / 2;
  if (!samples || samples.length === 0) {
    context.moveTo(0, middle);
    context.lineTo(width, middle);
  } else {
    // Average the samples under each pixel step for a smooth line.
    const step = samples.length / width;
    for (let x = 0; x < width; x++) {
      let sum = 0;
      const begin = Math.floor(x * step);
      const end = Math.max(begin + 1, Math.floor((x + 1) * step));
      for (let i = begin; i < end && i < samples.length; i++) {
        sum += samples[i];
      }
      const y = middle - Math.max(-1, Math.min(1, (sum / (end - begin)) * 4)) * middle * 0.9;
      if (x === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }
  }
  context.stroke();
  requestAnimationFrame(drawWave);
}

ui.clear.addEventListener("click", () => ui.lines.replaceChildren());
ui.record.addEventListener("click", () => (state === "recording" ? stop() : start()));
ui.file.addEventListener("change", () => {
  const [file] = ui.file.files;
  ui.file.value = "";
  if (file) {
    start(file);
  }
});

setState("idle");
requestAnimationFrame(drawWave);
loadModels().catch((error) => setStatus(`Cannot load the models: ${error.message}`, true));
