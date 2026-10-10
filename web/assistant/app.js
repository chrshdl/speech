// A client of the speech recognizer service: shows the cards the service
// sends, and starts a capture when the microphone button is tapped. See
// docs/backend-api.md.

const $ = (id) => document.getElementById(id);
const ui = {
  mic: $("mic"),
  status: $("status"),
  stage: $("stage"),
  greeting: $("greeting"),
  clock: $("clock"),
  date: $("date"),
  cardTemplate: $("card-template"),
  asked: $("asked"),
  askedText: $("asked-text"),
};

// Audio is sent in chunks of 100 ms.
const CHUNK_SECONDS = 0.1;

let info = null; // From /v1/info: the sample rate and where audio comes from.
let dialogState = "IDLE";
let card = null; // The card shown: its data and element.
let audio = null; // The microphone stream of the browser, while capturing.
let starting = false;
let heard = null; // The final transcript of the last capture, by its id.

// The landing page.

function updateClock() {
  const now = new Date();
  const hour = now.getHours();
  ui.greeting.textContent = hour < 5 ? "Good night" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  ui.clock.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  ui.date.textContent = now.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" });
}

function setStatus(text) {
  ui.status.textContent = text;
}

// Requests.

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  return response;
}

// Events.

function connectEvents() {
  const events = new EventSource("/v1/events");
  events.onopen = () => {
    document.body.dataset.connection = "online";
    setStatus("");
  };
  events.onerror = () => {
    // EventSource reconnects by itself.
    document.body.dataset.connection = "offline";
    setStatus("Reconnecting…");
  };
  const on = (name, handle) =>
    events.addEventListener(name, (event) => handle(JSON.parse(event.data).payload));

  on("Recognition.DialogStateChanged", ({ state }) => {
    dialogState = state;
    document.body.dataset.state = state;
    ui.mic.setAttribute("aria-label", state === "LISTENING" ? "Stop" : "Talk");
  });
  on("Recognition.CaptureFinished", ({ dialogRequestId, text }) => {
    heard = { id: dialogRequestId, text };
  });
  on("Display.ShowCard", ({ card: data }) => showCard(data));
  on("Display.ClearCard", ({ cardId }) => {
    if (card && card.data.id === cardId) {
      hideCard();
    }
  });
  on("Audio.StopAudioStream", () => stopAudio());
}

// Cards.

const RENDERERS = {
  TranscriptCard(data, body) {
    const text = document.createElement("p");
    text.className = "transcript" + (data.isFinal ? "" : " partial");
    text.textContent = data.text;
    body.append(text);
  },

  NoticeCard(data, body) {
    const text = document.createElement("p");
    text.className = "notice";
    text.textContent = data.text;
    body.append(text);
  },

  WeatherCard(data, body) {
    const degrees = (value) => `${Math.round(value)}°`;
    body.innerHTML = `
      <div class="weather-now">
        ${weatherIcon(data.condition, data.isDay)}
        <div>
          <p class="weather-temperature">${degrees(data.temperature)}</p>
          <p class="weather-description"></p>
        </div>
      </div>
      <ul class="weather-details">
        <li>Feels like <strong>${degrees(data.feelsLike)}</strong></li>
        <li>Humidity <strong>${data.humidity}%</strong></li>
        <li>Wind <strong>${Math.round(data.windSpeed)} ${data.windSpeedUnit}</strong></li>
      </ul>
      <ol class="weather-days" style="--days: ${data.days.length}">
        ${data.days
          .map(
            (day, i) => `
          <li class="weather-day">
            <div class="weather-day-name">${i === 0 ? "Today" : weekday(day.date)}</div>
            ${weatherIcon(day.condition)}
            <div class="weather-day-range">${degrees(day.high)}<span class="low">${degrees(day.low)}</span></div>
          </li>`,
          )
          .join("")}
      </ol>`;
    body.querySelector(".weather-description").textContent = data.description;
  },
};

const TITLES = {
  WeatherCard: (data) => `Weather in ${data.location}`,
};

function weekday(isoDate) {
  // A date of the forecast's place, so read it as a local date.
  const [year, month, day] = isoDate.split("-").map(Number);
  return new Date(year, month - 1, day).toLocaleDateString([], { weekday: "short" });
}

function showCard(data) {
  const render = RENDERERS[data.type];
  if (!render) {
    return; // A card type this client does not know.
  }
  // A card with the id of the card shown updates it in place.
  if (!card || card.data.id !== data.id || card.data.type !== data.type) {
    hideCard();
    const element = ui.cardTemplate.content.firstElementChild.cloneNode(true);
    element.querySelector(".card-close").addEventListener("click", () => dismissCard());
    ui.stage.append(element);
    card = { data, element };
  }
  card.data = data;
  const title = TITLES[data.type] ? TITLES[data.type](data) : data.title;
  card.element.querySelector(".card-title").textContent = title;
  const body = card.element.querySelector(".card-body");
  body.replaceChildren();
  render(data, body);
  showAsked(data);
  document.body.classList.add("has-card");
}

// Above an answer, such as the weather, what was asked for it. A
// transcript card shows it already.
function showAsked(data) {
  const asked = data.type !== "TranscriptCard" && heard && heard.id === data.id && heard.text;
  ui.askedText.textContent = asked ? `“${heard.text}”` : "";
  ui.asked.hidden = !asked;
}

function hideCard() {
  if (!card) {
    return;
  }
  const { element } = card;
  card = null;
  ui.asked.hidden = true;
  document.body.classList.remove("has-card");
  element.classList.add("leaving");
  element.addEventListener("animationend", () => element.remove(), { once: true });
  // Without animations, as with reduced motion, remove it at once.
  setTimeout(() => element.remove(), 400);
}

function dismissCard() {
  if (!card) {
    return;
  }
  const { id } = card.data;
  hideCard();
  post("/v1/display/card-dismissed", { cardId: id });
}

// Capturing.

async function toggleCapture() {
  if (document.body.dataset.connection !== "online" || starting) {
    return;
  }
  if (dialogState === "LISTENING") {
    stopCapture();
  } else if (dialogState === "IDLE") {
    await startCapture();
  }
}

async function startCapture() {
  starting = true;
  try {
    // Open the microphone before starting, so no speech is lost while the
    // capture starts. Its audio waits until the stream is open.
    if (info.audioInput === "CLIENT") {
      audio = await openMicrophone(info.sampleRate);
    }
    const response = await post("/v1/recognition/start-capture", { initiator: "TAP_TO_TALK" });
    if (!response.ok) {
      stopAudio();
      if (response.status !== 409) {
        setStatus("Could not start listening.");
      }
      return;
    }
    const { audioStream } = await response.json();
    if (audioStream && audio) {
      audio.stream(audioStream.url);
    }
  } catch (error) {
    stopAudio();
    setStatus(error.name === "NotAllowedError" ? "Allow the microphone to talk." : "Could not start listening.");
  } finally {
    starting = false;
  }
}

function stopCapture() {
  if (audio) {
    // Closing the audio stream ends the capture after all the audio sent.
    audio.end();
  } else {
    post("/v1/recognition/stop-capture");
  }
}

function stopAudio() {
  if (audio) {
    audio.close();
    audio = null;
  }
}

async function openMicrophone(rate) {
  const media = await navigator.mediaDevices.getUserMedia({
    // The models were trained on unprocessed recordings.
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  let context = new AudioContext({ sampleRate: rate });
  let source;
  try {
    source = context.createMediaStreamSource(media);
  } catch {
    // Firefox only captures at the device's rate, so the worklet resamples.
    await context.close();
    context = new AudioContext();
    source = context.createMediaStreamSource(media);
  }
  await context.audioWorklet.addModule("/recorder.js");
  const recorder = new AudioWorkletNode(context, "recorder", {
    processorOptions: { targetRate: rate, chunk: Math.round(rate * CHUNK_SECONDS) },
  });
  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  source.connect(recorder);
  // The worklet outputs silence, but it only runs while connected.
  recorder.connect(context.destination);

  let socket = null;
  const waiting = []; // Chunks recorded before the stream opened.
  recorder.port.onmessage = (event) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(event.data);
    } else if (!socket) {
      waiting.push(event.data);
    }
  };

  // Show the loudness on the button.
  const samples = new Float32Array(analyser.fftSize);
  let frame = requestAnimationFrame(function level() {
    analyser.getFloatTimeDomainData(samples);
    const rms = Math.sqrt(samples.reduce((sum, x) => sum + x * x, 0) / samples.length);
    ui.mic.style.setProperty("--level", Math.min(1, rms * 8).toFixed(3));
    frame = requestAnimationFrame(level);
  });

  return {
    stream(url) {
      const scheme = location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${scheme}//${location.host}${url}`);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => waiting.splice(0).forEach((chunk) => socket.send(chunk));
      // The service closes the stream when the capture ends.
      socket.onclose = () => stopAudio();
    },
    end() {
      if (socket) {
        socket.close();
      }
    },
    close() {
      cancelAnimationFrame(frame);
      ui.mic.style.removeProperty("--level");
      media.getTracks().forEach((track) => track.stop());
      context.close();
      if (socket && socket.readyState <= WebSocket.OPEN) {
        socket.close();
      }
    },
  };
}

// Start.

async function start() {
  updateClock();
  setInterval(updateClock, 1000);
  ui.mic.addEventListener("click", toggleCapture);
  document.addEventListener("keydown", (event) => {
    if (event.code === "Space" && !event.repeat && event.target === document.body) {
      event.preventDefault();
      toggleCapture();
    } else if (event.code === "Escape") {
      dismissCard();
    }
  });
  try {
    info = await (await fetch("/v1/info")).json();
  } catch {
    setStatus("The service is not reachable.");
    document.body.dataset.connection = "offline";
    return;
  }
  connectEvents();
}

start();
