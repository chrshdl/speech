// A client of the speech recognizer service: shows the cards the service
// sends, and starts a capture when the microphone button is tapped. See
// docs/backend-api.md.

const $ = (id) => document.getElementById(id);
const ui = {
  mic: $("mic"),
  themeToggle: $("theme-toggle"),
  status: $("status"),
  stage: $("stage"),
  greeting: $("greeting"),
  clock: $("clock"),
  date: $("date"),
  cardTemplate: $("card-template"),
  asked: $("asked"),
  askedText: $("asked-text"),
  meadow: document.querySelector("main"),
  unicorn: document.querySelector(".step-aside"),
  gameLayer: $("game-layer"),
  game: $("game"),
  steerLeft: $("steer-left"),
  steerRight: $("steer-right"),
};

// Audio is sent in chunks of 100 ms.
const CHUNK_SECONDS = 0.1;

let info = null; // From /v1/info: the sample rate and where audio comes from.
let dialogState = "IDLE";
let card = null; // The card shown: its data and element.
let audio = null; // The microphone stream of the browser, while capturing.
let starting = false;
let heard = null; // The final transcript of the last capture, by its id.
let game = null; // The Superball being played, and the id of its card.
let commands = null; // The id of the capture that hears the game's commands.
let lastCommands = null; // The last such id, also once that capture finished.
let startingCommands = false; // Whether such a capture is being started.
let listening = null; // The id of the capture listening, if any.
let restarts = []; // When listening for commands restarted by itself.
let wifi = null; // Joining a Wi-Fi network, where the service can, see wifi.js.

// The landing page.

function updateClock() {
  const now = new Date();
  const hour = now.getHours();
  // ui.greeting.textContent = hour < 5 ? "Good night" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  ui.clock.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  ui.date.textContent = now.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" });
}

function setStatus(text) {
  ui.status.textContent = text;
}

// Day and night.

function setTheme(theme) {
  const night = theme === "night";
  if (night) {
    document.documentElement.dataset.theme = "night";
  } else {
    delete document.documentElement.dataset.theme;
  }
  ui.themeToggle.setAttribute("aria-pressed", String(night));
  const label = night ? "Switch to day" : "Switch to night";
  ui.themeToggle.setAttribute("aria-label", label);
  ui.themeToggle.title = label;
  try {
    localStorage.setItem("theme", theme);
  } catch {
    // Remembering the theme is only a convenience.
  }
}

function toggleTheme() {
  setTheme(document.documentElement.dataset.theme === "night" ? "day" : "night");
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

  on("Recognition.DialogStateChanged", ({ state, dialogRequestId }) => {
    const wasListening = dialogState === "LISTENING";
    dialogState = state;
    document.body.dataset.state = state;
    ui.mic.setAttribute("aria-label", state === "LISTENING" ? "Stop" : "Talk");
    // Chime when listening for a question starts and ends, but not for the
    // game's commands, which it listens for all along.
    if (state === "LISTENING" && !wasListening) {
      listening = dialogRequestId;
      // A capture this client started chimed before it opened the
      // microphone; one started elsewhere chimes now.
      if (!isCommands(listening) && !chimed) {
        playTone(START_TONE);
      }
      chimed = false;
    } else if (wasListening && state !== "LISTENING") {
      if (!isCommands(listening)) {
        playTone(END_TONE);
      }
      listening = null;
    }
    listenForCommands();
  });
  on("Recognition.CaptureFinished", ({ dialogRequestId, text }) => {
    if (dialogRequestId === commands) {
      commands = null;
    } else {
      heard = { id: dialogRequestId, text };
    }
  });
  on("Recognition.CommandRecognized", ({ command }) => {
    if (game) {
      game.superball.command(command);
    }
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

// A note of music, on a 32 by 48 grid, in its own svg so that each can
// move without the card being drawn again.
const SONG_NOTE = `<svg class="song-note" viewBox="0 0 32 48" aria-hidden="true">
  <ellipse cx="11" cy="38" rx="9" ry="7" transform="rotate(-20 11 38)"/>
  <rect x="17" y="6" width="4" height="32" rx="2"/>
  <path d="M19 6 C27 9 31 17 25 26 C26 18 24 15 19 14 Z"/>
</svg>`;

RENDERERS.SongCard = (data, body) => {
  body.innerHTML = `
    <div class="song">
      <div class="song-notes">${SONG_NOTE.repeat(3)}</div>
      <div class="song-progress"><span class="song-left"></span></div>
    </div>`;
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
  if (data.type === "GameCard") {
    startGame(data);
    return;
  }
  const render = RENDERERS[data.type];
  if (!render) {
    return; // A card type this client does not know.
  }
  // Another card, such as the answer to a question, ends the game.
  endGame();
  // A card with the id of the card shown updates it in place.
  const fresh = !card || card.data.id !== data.id || card.data.type !== data.type;
  if (fresh) {
    hideCard();
    const element = ui.cardTemplate.content.firstElementChild.cloneNode(true);
    element.querySelector(".card-close").addEventListener("click", () => dismissCard());
    ui.stage.append(element);
    card = { data, element };
  } else if (data.type === "SongCard") {
    // Shown again, as when the events reconnect, the song plays on.
    return;
  }
  card.data = data;
  const title = TITLES[data.type] ? TITLES[data.type](data) : data.title;
  card.element.querySelector(".card-title").textContent = title;
  const body = card.element.querySelector(".card-body");
  body.replaceChildren();
  render(data, body);
  showAsked(data);
  document.body.classList.add("has-card");
  if (data.type === "SongCard") {
    playSong(card);
  }
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
  const { data, element } = card;
  card = null;
  if (data.type === "GameCard") {
    endGame();
    return;
  }
  if (data.type === "SongCard") {
    stopSong();
  }
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

// Tones: a rising chime when listening starts and a falling one when it
// ends, made with Web Audio, so they need no files. Browsers only play
// sound after the user taps, so the first tap of the button unlocks them.

// The start tone is the cue to speak, so it is the louder one.
const START_TONE = { notes: [659.25, 987.77], volume: 0.32 }; // E5, then B5.
const END_TONE = { notes: [987.77, 659.25], volume: 0.2 };
const NOTE_STEP = 0.09;
const NOTE_LENGTH = 0.25;
let tones = null;
// Whether this client chimed for the capture it starts, before listening.
let chimed = false;

function unlockTones() {
  try {
    tones ??= new AudioContext();
    if (tones.state === "suspended") {
      tones.resume();
    }
  } catch {
    // Without Web Audio there are no tones, which only accompany the page.
  }
}

// Plays the tone, and resolves once it has played, at once without one.
async function playTone({ notes, volume }) {
  if (!tones) {
    return;
  }
  if (tones.state !== "running") {
    // Firefox resumes the context only after the tap that unlocked it.
    try {
      await tones.resume();
    } catch {
      return;
    }
    if (tones.state !== "running") {
      return;
    }
  }
  const start = tones.currentTime + 0.01;
  notes.forEach((frequency, i) => {
    const time = start + i * NOTE_STEP;
    const oscillator = tones.createOscillator();
    oscillator.frequency.value = frequency;
    const gain = tones.createGain();
    gain.gain.setValueAtTime(0, time);
    gain.gain.linearRampToValueAtTime(volume, time + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + NOTE_LENGTH - 0.01);
    oscillator.connect(gain).connect(tones.destination);
    oscillator.start(time);
    oscillator.stop(time + NOTE_LENGTH);
  });
  await new Promise((resolve) => setTimeout(resolve, toneSeconds({ notes }) * 1000));
}

function toneSeconds({ notes }) {
  return (notes.length - 1) * NOTE_STEP + NOTE_LENGTH;
}

// Songs: a SongCard plays the song the service rendered, on the tones'
// context, and is dismissed once it has played. The card going, such as
// when it is closed or replaced, stops the song.

let song = null; // The song card playing, and its audio source once loaded.

// Whether the song card was started before, and remembers it was. The page
// keeps this across a reload, so a card that the service shows again after
// one, such as when the page crashed playing it, is not played again.
function startedBefore(id) {
  try {
    if (sessionStorage.getItem("song") === id) {
      return true;
    }
    sessionStorage.setItem("song", id);
  } catch {
    // Without storage, every card plays.
  }
  return false;
}

async function playSong(songCard) {
  stopSong();
  const { url, id } = songCard.data;
  if (startedBefore(id)) {
    dismissCard();
    return;
  }
  const playing = { card: songCard, source: null };
  song = playing;
  try {
    unlockTones();
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`${url}: ${response.status}`);
    }
    const audio = await tones.decodeAudioData(await response.arrayBuffer());
    if (tones.state !== "running") {
      // Before a tap allows sound, such as right after the page loaded, it
      // does not resume.
      await Promise.race([tones.resume(), new Promise((resolve) => setTimeout(resolve, 1000))]);
      if (tones.state !== "running") {
        throw new Error("Sound is not allowed before a tap");
      }
    }
    if (song !== playing) {
      return; // Stopped while loading.
    }
    const source = tones.createBufferSource();
    source.buffer = audio;
    source.connect(tones.destination);
    source.onended = () => {
      if (song === playing) {
        song = null;
        if (card && card.data.id === id) {
          dismissCard();
        }
      }
    };
    playing.source = source;
    source.start();
    // The rainbow shows as the song plays, from when it starts.
    const left = songCard.element.querySelector(".song-left");
    left.style.animationDuration = `${audio.duration}s`;
    songCard.element.classList.add("playing");
  } catch (error) {
    console.error("Could not play the song:", error);
    if (song === playing) {
      song = null;
      if (card && card.data.id === id) {
        dismissCard();
      }
    }
  }
}

function stopSong() {
  if (!song) {
    return;
  }
  const { source } = song;
  song = null;
  if (source) {
    source.onended = null;
    source.stop();
  }
}

function isCommands(id) {
  return startingCommands || (id !== null && id === lastCommands);
}

// Superball, see game.js. The game's card starts it, and while it is played
// a COMMANDS capture listens for "left" and "right".

function startGame(data) {
  if (game && game.id === data.id) {
    return;
  }
  hideCard();
  card = { data, element: null };
  const superball = new Superball({
    view: ui.meadow,
    unicorn: ui.unicorn,
    layer: ui.gameLayer,
    overlay: ui.game,
    onOver: () => {
      if (commands) {
        stopCapture();
      }
    },
    // Once the score was shown, the game's card goes.
    onEnd: () => dismissCard(),
  });
  game = { id: data.id, superball };
  restarts = [];
  superball.start(data.level);
  listenForCommands();
}

function endGame() {
  if (!game) {
    return;
  }
  game.superball.destroy();
  game = null;
  if (commands) {
    stopCapture();
  }
}

// Listens for the game's commands while it needs them, also again when the
// capture ended by itself, unless that keeps happening.
async function listenForCommands() {
  if (!game || !game.superball.listening || commands || dialogState !== "IDLE" || starting) {
    return;
  }
  const now = Date.now();
  restarts = restarts.filter((time) => now - time < 10000);
  if (restarts.length >= 3) {
    setStatus("I can't hear you. Is the microphone on?");
    game.superball.quit();
    return;
  }
  restarts.push(now);
  await startCapture("COMMANDS");
}

function steer(command) {
  if (game) {
    game.superball.command(command);
  }
}

// Capturing.

async function toggleCapture() {
  unlockTones();
  if (document.body.dataset.connection !== "online" || starting) {
    return;
  }
  if (game) {
    // The button stops the game, which then shows the score.
    const { superball } = game;
    if (superball.listening) {
      superball.quit();
      return;
    }
    dismissCard();
  }
  if (dialogState === "LISTENING") {
    stopCapture();
  } else if (dialogState === "IDLE") {
    await startCapture("TAP_TO_TALK");
  }
}

async function startCapture(initiator) {
  starting = true;
  startingCommands = initiator === "COMMANDS";
  try {
    // Chime before listening, so the microphone opening never dulls the
    // tone, which is the cue to speak, and the microphone misses it.
    if (!startingCommands) {
      // The song stops for the question, so it is neither heard nor talked over.
      stopSong();
      chimed = true;
      await playTone(START_TONE);
    }
    // Open the microphone before starting, so no speech is lost while the
    // capture starts. Its audio waits until the stream is open.
    if (info.audioInput === "CLIENT") {
      audio = await openMicrophone(info.sampleRate);
    }
    const response = await post("/v1/recognition/start-capture", { initiator });
    if (!response.ok) {
      chimed = false;
      stopAudio();
      if (response.status !== 409) {
        setStatus("Could not start listening.");
      }
      return;
    }
    const { dialogRequestId, audioStream } = await response.json();
    if (audioStream && audio) {
      audio.stream(audioStream.url);
    }
    if (initiator === "COMMANDS") {
      commands = dialogRequestId;
      lastCommands = dialogRequestId;
      // The game can end while listening starts.
      if (!game || !game.superball.listening) {
        stopCapture();
      }
    }
  } catch (error) {
    chimed = false;
    stopAudio();
    setStatus(error.name === "NotAllowedError" ? "Allow the microphone to talk." : "Could not start listening.");
  } finally {
    starting = false;
    startingCommands = false;
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
    // Closing the microphone can switch the sound output, such as Bluetooth
    // headphones out of their headset mode, which cuts off what plays, so
    // it closes once the end tone has played.
    audio.close(toneSeconds(END_TONE) + 0.15);
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
    // Stops streaming at once, and closes the microphone after the delay
    // in seconds.
    close(delay = 0) {
      cancelAnimationFrame(frame);
      ui.mic.style.removeProperty("--level");
      recorder.port.onmessage = null;
      if (socket && socket.readyState <= WebSocket.OPEN) {
        socket.close();
      }
      setTimeout(() => {
        media.getTracks().forEach((track) => track.stop());
        context.close();
      }, delay * 1000);
    },
  };
}

// Start.

async function start() {
  updateClock();
  setInterval(updateClock, 1000);
  setTheme(document.documentElement.dataset.theme === "night" ? "night" : "day");
  ui.themeToggle.addEventListener("click", toggleTheme);
  ui.mic.addEventListener("click", toggleCapture);
  ui.steerLeft.addEventListener("click", () => steer("LEFT"));
  ui.steerRight.addEventListener("click", () => steer("RIGHT"));
  document.addEventListener("keydown", (event) => {
    if (wifi && wifi.isOpen) {
      if (event.code === "Escape") {
        wifi.back();
      }
      return;
    }
    if (event.code === "Space" && !event.repeat && event.target === document.body) {
      event.preventDefault();
      toggleCapture();
    } else if (event.code === "Escape") {
      dismissCard();
    } else if (game && (event.code === "ArrowLeft" || event.code === "ArrowRight")) {
      event.preventDefault();
      steer(event.code === "ArrowLeft" ? "LEFT" : "RIGHT");
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
  if (info.wifi) {
    wifi = new WifiSetup({ root: $("wifi"), button: $("wifi-button") });
    wifi.start();
  }
}

start();
