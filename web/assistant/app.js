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
  keys: $("keys"),
  keysTitle: $("keys-title"),
  keysBoard: $("keys-board"),
  keysClose: $("keys-close"),
  keysOpen: $("keys-open"),
  keysDown: $("keys-down"),
  keysUp: $("keys-up"),
  keysOctave: $("keys-octave"),
  drums: $("drums"),
  drumsTitle: $("drums-title"),
  drumsKit: $("drums-kit"),
  drumsClose: $("drums-close"),
  drumsOpen: $("drums-open"),
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
let keyboard = null; // The synthesizer's keyboard shown, and the id of its card.
let drums = null; // The drums shown, and the id of their card.
let commands = null; // The id of the capture that hears the game's commands.
let lastCommands = null; // The last such id, also once that capture finished.
let startingCommands = false; // Whether such a capture is being started.
let listening = null; // The id of the capture listening, if any.
// The id of the last capture of a wake word heard by mistake, which ends
// without the end chime.
let mistaken = null;
let restarts = []; // When listening for commands restarted by itself.
let wifi = null; // Joining a Wi-Fi network, where the service can, see wifi.js.

// The landing page.

function updateClock() {
  const now = new Date();
  const hour = now.getHours();
  // ui.greeting.textContent = hour < 5 ? "Good night" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  // A 24 hour clock, also where the language's default has AM and PM.
  ui.clock.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
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
    checkShown();
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
      if (!isCommands(listening) && listening !== mistaken) {
        playTone(END_TONE);
      }
      listening = null;
    }
    duckRadio();
    listenForCommands();
  });
  on("Recognition.CaptureFinished", ({ dialogRequestId, text, reason }) => {
    if (reason === "NO_WAKE_WORD") {
      mistaken = dialogRequestId;
    }
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
    // Also beneath another card.
    if (radio && radio.id === cardId) {
      stopRadio();
    }
  });
  on("Audio.StopAudioStream", () => stopAudio());
  on("Speech.Speak", (speech) => speak(speech));
  on("Speech.StopSpeaking", ({ speechId }) => stopSpeaking(speechId));
}

// After the events reconnect, such as when the service restarted, the card
// shown before stays only if the service still shows it, and the radio plays
// on only if it still plays it. What came since is the service's own.
async function checkShown() {
  const [shown, playing] = [card, radio];
  if (!shown && !playing) {
    return;
  }
  try {
    const state = await (await fetch("/v1/state")).json();
    if (playing && radio === playing && state.playing?.id !== playing.id) {
      stopRadio();
    }
    if (shown && card === shown && state.card?.id !== shown.data.id) {
      hideCard();
    }
  } catch {
    // Checked again when the events reconnect.
  }
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

// A level just set, such as the brightness: its icon, how full the bar is,
// and the percent.
function renderMeter(body, kind, icon, percent) {
  body.innerHTML = `
    <div class="meter meter-${kind}">
      <svg class="meter-icon" viewBox="0 0 64 64" aria-hidden="true">${icon}</svg>
      <div class="meter-bar"><span class="meter-fill"></span></div>
      <p class="meter-percent">${percent}%</p>
    </div>`;
  body.querySelector(".meter-fill").style.width = `${percent}%`;
}

// The radio: the song's cover, or else the station's logo, beside what
// plays, under bars that bounce while it sounds.
RENDERERS.RadioCard = (data, body) => {
  body.innerHTML = `
    <div class="radio">
      <div class="radio-cover"><img alt="" /></div>
      <div class="radio-info">
        <p class="radio-live">
          <span class="radio-bars" aria-hidden="true"><span></span><span></span><span></span><span></span></span>
          Live
        </p>
        <p class="radio-song"></p>
        <p class="radio-artist"></p>
      </div>
    </div>`;
  const cover = body.querySelector(".radio-cover img");
  // Faded in once loaded, rather than drawn as it arrives.
  cover.addEventListener("load", () => cover.classList.add("loaded"), { once: true });
  cover.src = data.cover || data.logo;
  body.querySelector(".radio-song").textContent = data.song;
  body.querySelector(".radio-artist").textContent = data.artist;
};

RENDERERS.BrightnessCard = (data, body) => renderMeter(body, "brightness", sun(), data.percent);
RENDERERS.VolumeCard = (data, body) => renderMeter(body, "volume", loudspeaker(), data.percent);

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
  if (data.type === "KeyboardCard") {
    startKeyboard(data);
    return;
  }
  if (data.type === "DrumsCard") {
    startDrums(data);
    return;
  }
  const render = RENDERERS[data.type];
  if (!render) {
    return; // A card type this client does not know.
  }
  // Another card, such as the answer to a question, ends the game, or
  // closes the keyboard or the drums.
  endGame();
  endKeyboard();
  endDrums();
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
  if (data.type === "RadioCard") {
    playRadio(data);
    showRadioSounding();
  }
  if (data.type === "VolumeCard" && fresh) {
    // Heard at the new volume.
    playTone(VOLUME_TONE);
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
  if (data.type === "KeyboardCard") {
    endKeyboard();
    return;
  }
  if (data.type === "DrumsCard") {
    endDrums();
    return;
  }
  if (data.type === "SongCard") {
    stopSong();
  }
  if (speaking && speaking.cardId === data.id) {
    stopSpeaking();
  }
  ui.asked.hidden = true;
  document.body.classList.remove("has-card");
  // It fades where it is, but out of the stage's layout, so a card that
  // replaces it, and what was asked above that, are laid out where they
  // stay, rather than below it and then jumping once it is gone.
  const { offsetTop, offsetLeft, offsetWidth } = element;
  Object.assign(element.style, {
    position: "absolute",
    top: `${offsetTop}px`,
    left: `${offsetLeft}px`,
    width: `${offsetWidth}px`,
  });
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
  if (radio && radio.id === id) {
    stopRadio();
  }
  post("/v1/display/card-dismissed", { cardId: id });
}

// Tones: a rising chime when listening starts and a falling one when it
// ends, made with Web Audio, so they need no files. Browsers only play
// sound after the user taps, so the first tap of the button unlocks them.

// The start tone is the cue to speak, so it is the louder one.
const START_TONE = { notes: [659.25, 987.77], volume: 0.32 }; // E5, then B5.
const END_TONE = { notes: [987.77, 659.25], volume: 0.2 };
// The volume just turned, at its new level.
const VOLUME_TONE = { notes: [783.99], volume: 0.32 }; // G5.
const NOTE_STEP = 0.09;
const NOTE_LENGTH = 0.25;
let tones = null;
// Whether this client chimed for the capture it starts, before listening.
let chimed = false;
// While nothing plays, the tones' context is suspended: running, it renders
// silence all the time, which took about a sixth of a core of a Raspberry Pi 4.
// It suspends once nothing played for IDLE_MS.
const IDLE_MS = 3000;
let idleTimer = null;
let suspending = null; // The tones' context suspending.

function unlockTones() {
  try {
    tones ??= new AudioContext();
    if (tones.state === "suspended") {
      tones.resume();
      // Suspended again, unless something plays.
      idleTones();
    }
  } catch {
    // Without Web Audio there are no tones, which only accompany the page.
  }
}

// Readies Web Audio when the page loads: the tones' context, silent until a
// tap allows sound, with the synthesizer of songs added to it. Added only
// when the first song came, to a context already playing, it took WPE
// WebKit on a Raspberry Pi 4 about 5 seconds, which the song waited for,
// and a third of a second before.
function prepareSound() {
  try {
    tones ??= new AudioContext();
    loadSynth().catch(() => {});
    loadSpeaker().catch(() => {});
    loadDrumSounds().catch(() => {});
  } catch {
    // Without Web Audio there are no tones, songs or speech.
  }
}

// Suspends the tones' context once nothing played for IDLE_MS, also not a
// song, and while the keyboard or the drums are shown, so a key or a drum
// sounds at once.
function idleTones() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    if (song || keyboard || drums || speaking || radio || tones.state !== "running") {
      return;
    }
    suspending = tones.suspend().catch(() => {}).finally(() => {
      suspending = null;
    });
  }, IDLE_MS);
}

// Keeps the tones' context from suspending, and waits until it has, if it is
// suspending, so that what plays next resumes it.
async function wakeTones() {
  clearTimeout(idleTimer);
  await suspending;
}

// How much later than it renders the tones' context plays the sound.
function outputLatency() {
  return tones?.outputLatency || 0;
}

// The tones' context adding synth.js, tried again after it failed.
function loadSynth() {
  synth ??= tones.audioWorklet.addModule("synth.js").catch((error) => {
    synth = null;
    throw error;
  });
  return synth;
}

// Plays the tone, and resolves once it has been heard, at once without one.
async function playTone({ notes, volume }) {
  if (!tones) {
    return;
  }
  await wakeTones();
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
  const rendered = start - tones.currentTime + toneSeconds({ notes });
  await new Promise((resolve) => setTimeout(resolve, rendered * 1000));
  // Rendered by now, it is heard as much later as the output's latency, which
  // right after the context resumed can be less than it is once playing.
  await new Promise((resolve) => setTimeout(resolve, outputLatency() * 1000));
  idleTones();
}

function toneSeconds({ notes }) {
  return (notes.length - 1) * NOTE_STEP + NOTE_LENGTH;
}

// Songs: a SongCard plays the song's notes live, with the voice of the
// additive synthesizer in synth.js, or of a Roland D-50 for the notes with a
// patch, on the tones' context, in stereo, and is
// dismissed once the song has played, or plays it over and over until it is
// closed, for a song that loops. The card going, such as when it is closed
// or replaced, stops the song.

let song = null; // The song card playing, and its synthesizer once started.
let synth = null; // The tones' context loading synth.js.

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
  const playing = { card: songCard, node: null };
  song = playing;
  try {
    unlockTones();
    const [response] = await Promise.all([fetch(url), loadSynth()]);
    if (!response.ok) {
      throw new Error(`${url}: ${response.status}`);
    }
    const score = await response.json();
    await wakeTones();
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
    const node = new AudioWorkletNode(tones, "song", {
      numberOfInputs: 0,
      outputChannelCount: [2],
      processorOptions: score,
    });
    node.port.onmessage = ({ data }) => {
      if (data === "ended" && song === playing) {
        stopSong();
        if (card && card.data.id === id) {
          dismissCard();
        }
      }
    };
    node.connect(tones.destination);
    playing.node = node;
    // The rainbow shows as the song plays, from when it starts, and again
    // each time a looping song starts over.
    const left = songCard.element.querySelector(".song-left");
    left.style.animationDuration = `${score.seconds}s`;
    songCard.element.classList.toggle("looping", score.loop);
    songCard.element.classList.add("playing");
  } catch (error) {
    console.error("Could not play the song:", error);
    if (song === playing) {
      song = null;
      idleTones();
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
  const { node } = song;
  song = null;
  idleTones();
  if (node) {
    node.port.onmessage = null;
    node.port.postMessage("stop");
    node.disconnect();
  }
}

// Speech: the service speaks some answers, such as the weather, streaming
// each sentence as soon as it is synthesized. The speaker in speaker.js plays
// it on the tones' context as it arrives. A capture starting, or the card
// going, stops it.

// Below full scale, as the voice is normalized to it, so a chime with it
// does not clip.
const SPEECH_VOLUME = 0.8;
// How long the speaker fades out when stopped, as in speaker.js.
const SPEECH_FADE = 0.02;
let speaking = null; // The speech playing: its ids, request and node.
let speaker = null; // The tones' context loading speaker.js.

function loadSpeaker() {
  speaker ??= tones.audioWorklet.addModule("speaker.js").catch((error) => {
    speaker = null;
    throw error;
  });
  return speaker;
}

async function speak({ speechId, cardId, url, sampleRate }) {
  stopSpeaking();
  if (!tones) {
    return;
  }
  const playing = { id: speechId, cardId, request: new AbortController(), node: null };
  speaking = playing;
  duckRadio();
  try {
    const [response] = await Promise.all([fetch(url, { signal: playing.request.signal }), loadSpeaker()]);
    if (!response.ok) {
      throw new Error(`${url}: ${response.status}`);
    }
    await wakeTones();
    if (tones.state !== "running") {
      // Before a tap allows sound, it does not resume, as for songs.
      await Promise.race([tones.resume(), new Promise((resolve) => setTimeout(resolve, 1000))]);
      if (tones.state !== "running") {
        throw new Error("Sound is not allowed before a tap");
      }
    }
    if (speaking !== playing) {
      return; // Stopped while starting.
    }
    const node = new AudioWorkletNode(tones, "speaker", {
      numberOfInputs: 0,
      outputChannelCount: [1],
      processorOptions: { sampleRate, volume: SPEECH_VOLUME },
    });
    node.port.onmessage = ({ data }) => {
      if (data === "ended" && speaking === playing) {
        speaking = null;
        node.disconnect();
        duckRadio();
        idleTones();
      }
    };
    node.connect(tones.destination);
    playing.node = node;
    await feedSpeaker(response, node, 1, () => speaking === playing);
    node.port.postMessage("end");
  } catch (error) {
    if (error.name !== "AbortError") {
      console.error("Could not speak:", error);
    }
    if (speaking === playing) {
      stopSpeaking();
    }
  }
}

// Posts the 16 bit samples of the response to the speaker node as they
// arrive, in whole frames of so many channels, which a read can split, until
// the response ends or playing() is false.
async function feedSpeaker(response, node, channels, playing) {
  const frame = 2 * channels;
  const reader = response.body.getReader();
  let rest = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (done || !playing()) {
      return;
    }
    const bytes = new Uint8Array((rest ? rest.length : 0) + value.length);
    if (rest) {
      bytes.set(rest);
    }
    bytes.set(value, rest ? rest.length : 0);
    const whole = bytes.length - (bytes.length % frame);
    rest = whole < bytes.length ? bytes.slice(whole) : null;
    const samples = bytes.buffer.slice(0, whole);
    node.port.postMessage(samples, [samples]);
  }
}

// Stops the speaker node, which fades out, and disconnects it once it has.
function silenceSpeaker(node) {
  node.port.onmessage = null;
  node.port.postMessage("stop");
  setTimeout(() => node.disconnect(), (SPEECH_FADE + 0.05) * 1000);
}

// Stops the speech playing, or only the one with the id.
function stopSpeaking(id) {
  if (!speaking || (id && speaking.id !== id)) {
    return;
  }
  const { request, node } = speaking;
  speaking = null;
  request.abort();
  if (node) {
    silenceSpeaker(node);
  }
  duckRadio();
  idleTones();
}

// Web radio: a RadioCard streams the station's audio from the service, which
// decodes it, and a speaker of speaker.js plays it as it arrives, as speech,
// so the page needs neither the internet nor codecs. It is turned down while
// the service listens or speaks, plays on beneath the cards shown over it,
// such as the volume's, and stops when its card is cleared or closed.

// Below full scale, as speech, so a chime over it does not clip.
const RADIO_VOLUME = 0.8;
// The share of its volume it keeps while turned down.
const RADIO_DUCKED = 0.15;
// The seconds of audio it holds before it plays, also after it ran dry, so a
// slow network makes it pause rather than stutter.
const RADIO_PREBUFFER = 1;
// How long it waits to connect again when the stream ended or failed, and
// how often in a row it tries before it gives up.
const RADIO_RETRY_MS = 3000;
const RADIO_RETRIES = 5;
let radio = null; // The station playing: its card's id, its stream and its nodes.

async function playRadio({ id, url, sampleRate, channels }) {
  if (radio && radio.url === url) {
    // Asked for again, it plays on, for its new card.
    radio.id = id;
    return;
  }
  stopRadio();
  if (!tones) {
    return;
  }
  const playing = { id, url, sampleRate, channels, gain: null, request: null, node: null };
  Object.assign(playing, { retry: null, failures: 0, error: null, sounding: false });
  radio = playing;
  try {
    unlockTones();
    await loadSpeaker();
    await wakeTones();
    if (tones.state !== "running") {
      // Before a tap allows sound, it does not resume, as for songs.
      await Promise.race([tones.resume(), new Promise((resolve) => setTimeout(resolve, 1000))]);
      if (tones.state !== "running") {
        throw new Error("Sound is not allowed before a tap");
      }
    }
  } catch (error) {
    if (radio === playing) {
      console.error("Could not play the radio:", error);
      giveUpRadio();
    }
    return;
  }
  if (radio !== playing) {
    return; // Stopped while starting.
  }
  playing.gain = tones.createGain();
  playing.gain.gain.value = radioVolume();
  playing.gain.connect(tones.destination);
  streamRadio(playing);
}

// Plays the radio's stream on a speaker of its own until the stream ends,
// such as when the station's connection dropped, and then connects again.
async function streamRadio(playing) {
  const request = new AbortController();
  const node = new AudioWorkletNode(tones, "speaker", {
    numberOfInputs: 0,
    outputChannelCount: [playing.channels],
    processorOptions: { sampleRate: playing.sampleRate, channels: playing.channels, prebuffer: RADIO_PREBUFFER },
  });
  node.port.onmessage = ({ data }) => {
    if (data === "playing") {
      playing.failures = 0;
    }
    playing.sounding = data === "playing";
    showRadioSounding();
  };
  node.connect(playing.gain);
  Object.assign(playing, { request, node });
  const current = () => radio === playing && playing.node === node;
  try {
    const response = await fetch(playing.url, { signal: request.signal });
    if (!response.ok) {
      throw new Error(`${playing.url}: ${response.status}`);
    }
    await feedSpeaker(response, node, playing.channels, current);
  } catch (error) {
    playing.error = error;
  }
  if (!current()) {
    return; // Stopped.
  }
  silenceSpeaker(node);
  playing.node = null;
  playing.sounding = false;
  showRadioSounding();
  if (++playing.failures > RADIO_RETRIES) {
    console.error("Could not play the radio:", playing.error);
    giveUpRadio();
    return;
  }
  playing.retry = setTimeout(() => {
    playing.retry = null;
    if (radio === playing) {
      streamRadio(playing);
    }
  }, RADIO_RETRY_MS);
}

// Stops the radio that cannot play, and closes its card.
function giveUpRadio() {
  const { id } = radio;
  stopRadio();
  if (card && card.data.id === id) {
    dismissCard();
  } else {
    post("/v1/display/card-dismissed", { cardId: id });
  }
}

function stopRadio() {
  if (!radio) {
    return;
  }
  const { request, node, gain, retry } = radio;
  radio = null;
  clearTimeout(retry);
  request?.abort();
  if (node) {
    silenceSpeaker(node);
  }
  if (gain) {
    setTimeout(() => gain.disconnect(), (SPEECH_FADE + 0.05) * 1000);
  }
  showRadioSounding();
  idleTones();
}

function radioVolume() {
  const ducked = starting || dialogState !== "IDLE" || speaking;
  return RADIO_VOLUME * (ducked ? RADIO_DUCKED : 1);
}

// Turns the radio down while listening or speaking, and up again after.
function duckRadio() {
  if (radio && radio.gain) {
    radio.gain.gain.setTargetAtTime(radioVolume(), tones.currentTime, 0.08);
  }
}

// The bars of the radio's card bounce while it sounds.
function showRadioSounding() {
  if (card && card.element && card.data.type === "RadioCard") {
    card.element.classList.toggle("playing", Boolean(radio && radio.sounding));
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

// The synthesizer's keyboard, see keyboard.js: its card shows it, and it
// plays the card's patch with the D-50's voice, in synth.js, on the tones'
// context, until it is closed.

function startKeyboard(data) {
  if (keyboard && keyboard.id === data.id) {
    return;
  }
  hideCard();
  card = { data, element: null };
  const shown = { id: data.id, node: null, keys: null };
  keyboard = shown;
  ui.keysTitle.textContent = data.title;
  ui.keys.hidden = false;
  document.body.classList.add("keyboard");
  // A tap allows sound, so a key also wakes the context, where nothing did
  // before, such as when the page reloaded with the keyboard shown. WPE
  // WebKit takes the finger lifting as the tap, not its touching.
  const wake = () => {
    if (tones && tones.state !== "running") {
      tones.resume().catch(() => {});
    }
  };
  shown.keys = new Keys({
    element: ui.keysBoard,
    onDown: (id, frequency) => {
      wake();
      shown.node?.port.postMessage({ down: id, frequency });
    },
    onUp: (id) => {
      wake();
      shown.node?.port.postMessage({ up: id });
    },
  });
  showOctave();
  loadKeyboard(shown, data.url);
}

// Readies the keyboard's synthesizer with the patch from url.
async function loadKeyboard(shown, url) {
  try {
    unlockTones();
    const [response] = await Promise.all([fetch(url), loadSynth()]);
    if (!response.ok) {
      throw new Error(`${url}: ${response.status}`);
    }
    const patch = await response.json();
    await wakeTones();
    if (keyboard !== shown) {
      return; // Closed while loading.
    }
    // Running already when the first key is pressed, where a tap before
    // allowed sound, so that key need not wait for it.
    if (tones.state !== "running") {
      tones.resume().catch(() => {});
    }
    const node = new AudioWorkletNode(tones, "keyboard", {
      numberOfInputs: 0,
      outputChannelCount: [2],
      processorOptions: { patch },
    });
    node.connect(tones.destination);
    shown.node = node;
  } catch (error) {
    console.error("Could not ready the keyboard:", error);
  }
}

// Shifts the keyboard's keys an octave down, by -1, or up, by 1.
function shiftKeyboard(octaves) {
  if (!keyboard) {
    return;
  }
  keyboard.keys.shift(octaves);
  showOctave();
}

// Shows the lowest key of the keyboard, and which way it can shift.
function showOctave() {
  const { keys } = keyboard;
  ui.keysOctave.textContent = keys.name;
  ui.keysDown.disabled = !keys.canShiftDown;
  ui.keysUp.disabled = !keys.canShiftUp;
}

// The keys button asks the service for the keyboard, as asking for it by
// voice does, and its tap allows sound, so the first key sounds at once.
function openKeyboard() {
  unlockTones();
  if (document.body.dataset.connection !== "online") {
    return;
  }
  post("/v1/intents/PlayKeyboard").catch(() => {});
}

function endKeyboard() {
  if (!keyboard) {
    return;
  }
  const { keys, node } = keyboard;
  keyboard = null;
  keys.destroy();
  if (node) {
    node.port.postMessage("stop");
    node.disconnect();
  }
  ui.keys.hidden = true;
  document.body.classList.remove("keyboard");
  idleTones();
}

// The drums, see drums.js: their card shows them, and each drum hit plays
// its recording, decoded once, on the tones' context, until they are closed.

// Below full scale, as speech, so a chime over a drum does not clip. The
// recordings peak at full scale.
const DRUMS_VOLUME = 0.8;
// The drums' sounds are mixed through a limiter, as several drums hit at once
// add up: up to DRUMS_LIMIT, as loud as one drum, they pass as they are, and
// above it they are bent by tanh toward full scale. A WaveShaperNode does
// that without delay, unlike a DynamicsCompressorNode, which looks ahead.
const DRUMS_LIMIT = DRUMS_VOLUME;
let drumSounds = null; // Decoding the drums' sounds, see decodeDrums().

function loadDrumSounds() {
  drumSounds ??= decodeDrums(tones).catch((error) => {
    drumSounds = null;
    throw error;
  });
  return drumSounds;
}

function startDrums(data) {
  if (drums && drums.id === data.id) {
    return;
  }
  hideCard();
  card = { data, element: null };
  const shown = { id: data.id, kit: null, sounds: null, mix: null };
  drums = shown;
  ui.drumsTitle.textContent = data.title;
  ui.drums.hidden = false;
  document.body.classList.add("drums");
  // A tap allows sound, so a drum also wakes the context, as a key does,
  // also on lifting, which WPE WebKit takes as the tap.
  const wake = () => {
    if (tones && tones.state !== "running") {
      tones.resume().catch(() => {});
    }
  };
  shown.kit = new DrumKit({
    element: ui.drumsKit,
    onHit: (name) => {
      wake();
      playDrum(shown, name);
    },
    onLift: wake,
  });
  loadDrums(shown);
}

// Readies the drums' sounds and their mix.
async function loadDrums(shown) {
  try {
    unlockTones();
    const sounds = await loadDrumSounds();
    await wakeTones();
    if (drums !== shown) {
      return; // Closed while loading.
    }
    if (tones.state !== "running") {
      tones.resume().catch(() => {});
    }
    // The limiter's curve spans twice full scale, which the gain before it
    // halves into the curve's -1 to 1.
    const gain = tones.createGain();
    gain.gain.value = DRUMS_VOLUME / 2;
    const limiter = tones.createWaveShaper();
    const curve = new Float32Array(1025);
    for (let i = 0; i < curve.length; i++) {
      const x = 2 * ((2 * i) / (curve.length - 1) - 1);
      const over = Math.max(Math.abs(x) - DRUMS_LIMIT, 0);
      const bent = (1 - DRUMS_LIMIT) * Math.tanh(over / (1 - DRUMS_LIMIT));
      curve[i] = Math.sign(x) * (Math.min(Math.abs(x), DRUMS_LIMIT) + bent);
    }
    limiter.curve = curve;
    gain.connect(limiter).connect(tones.destination);
    shown.mix = { gain, limiter };
    shown.sounds = sounds;
  } catch (error) {
    console.error("Could not ready the drums:", error);
  }
}

function playDrum(shown, name) {
  if (!shown.sounds) {
    return;
  }
  const source = tones.createBufferSource();
  source.buffer = shown.sounds.get(name);
  source.connect(shown.mix.gain);
  source.start();
}

// The drums button asks the service for the drums, as asking for them by
// voice does, and its tap allows sound, so the first drum sounds at once.
function openDrums() {
  unlockTones();
  if (document.body.dataset.connection !== "online") {
    return;
  }
  post("/v1/intents/PlayDrums").catch(() => {});
}

function endDrums() {
  if (!drums) {
    return;
  }
  const { kit, mix } = drums;
  drums = null;
  kit.destroy();
  if (mix) {
    // What still rings, such as a cymbal, fades rather than clicks.
    mix.gain.gain.setTargetAtTime(0, tones.currentTime, 0.02);
    setTimeout(() => {
      mix.gain.disconnect();
      mix.limiter.disconnect();
    }, 150);
  }
  ui.drums.hidden = true;
  document.body.classList.remove("drums");
  idleTones();
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
  if (keyboard || drums) {
    // The button closes the keyboard or the drums, to ask something else.
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
  duckRadio();
  try {
    // Chime before listening, so the microphone opening never dulls the
    // tone, which is the cue to speak, and the microphone misses it.
    // Speech stops for listening, so the microphone does not hear it.
    stopSpeaking();
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
    duckRadio();
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
    audio.close(toneSeconds(END_TONE) + outputLatency() + 0.15);
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
  prepareSound();
  updateClock();
  setInterval(updateClock, 1000);
  setTheme(document.documentElement.dataset.theme === "night" ? "night" : "day");
  ui.themeToggle.addEventListener("click", toggleTheme);
  ui.mic.addEventListener("click", toggleCapture);
  ui.keysOpen.addEventListener("click", openKeyboard);
  ui.drumsOpen.addEventListener("click", openDrums);
  ui.steerLeft.addEventListener("click", () => steer("LEFT"));
  ui.steerRight.addEventListener("click", () => steer("RIGHT"));
  // Also on lifting, as a click may not come: WPE WebKit makes none while
  // it takes another finger as down, such as of a chord on the keys.
  ui.keysClose.addEventListener("pointerup", () => dismissCard());
  ui.keysClose.addEventListener("click", () => dismissCard());
  ui.drumsClose.addEventListener("pointerup", () => dismissCard());
  ui.drumsClose.addEventListener("click", () => dismissCard());
  // The octave buttons shift on touching, at once, which is also why no
  // click is needed, and on a click from the keyboard, without a pointer.
  for (const [button, octaves] of [
    [ui.keysDown, -1],
    [ui.keysUp, 1],
  ]) {
    button.addEventListener("pointerdown", () => shiftKeyboard(octaves));
    button.addEventListener("click", (event) => {
      if (event.detail === 0) {
        shiftKeyboard(octaves);
      }
    });
  }
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
