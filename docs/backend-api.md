# Speech recognizer service API

`backend.py` runs streaming speech recognition as a service, such as on a
Raspberry Pi that a head unit or another client talks to. The client
starts and stops captures, and the service sends events with the
transcript and the display cards to show. When an utterance asks for
something the service can answer, such as "what's the weather?", it
recognizes the intent and shows the answer, see [Intents](#intents). The
client renders what the service tells it, and the service decides what to
show and for how long. Asked to "play superball", it starts a game, which
the user steers by saying "left" and "right", see [Commands](#commands).

The service also serves a client at `/`: a landing page with a
microphone button, see [The browser client](#the-browser-client).

The interactive reference, with the schemas of every request, response
and event, is served at `/docs`, and the OpenAPI document at
`/openapi.json`.

## Concepts

The API has three topics:

| Topic | Purpose |
|---|---|
| `Recognition` | Starting and stopping captures, the dialog state and the transcript |
| `Display` | The display cards to show and clear |
| `Audio` | Streaming a capture's audio from the client |

Requests go in over REST, and events come out as server-sent events.

**Capture.** One utterance, from start-capture to the final transcript,
or a stream of short commands. Its `dialogRequestId` is in each of its
events, and is the id of its display cards. There is one capture at a
time.

**Initiator.** How the capture was started, which decides how it ends:

| Initiator | Ends |
|---|---|
| `HOLD_TO_TALK` | When the client calls stop-capture, or closes its audio stream, as the button is released |
| `TAP_TO_TALK` | At the end of speech: a pause of 1 second after the transcript starts. Also with stop-capture. |
| `COMMANDS` | Only with stop-capture, or when the client closes its audio stream. It listens for short commands, such as "left" and "right" to steer a game, and reports each in a `CommandRecognized` event, see [Commands](#commands). |

A `HOLD_TO_TALK` or `TAP_TO_TALK` capture ends after 60 seconds, a
`COMMANDS` capture after 30 minutes, and any when no audio arrives for 5
seconds. A `TAP_TO_TALK` capture that recognizes no speech in 8 seconds
ends too.
`backend.py -h` lists the options that change these limits.

**Dialog state.** `IDLE`, ready for a capture; `LISTENING` to a capture;
`THINKING` while decoding the final transcript, which takes a moment
with a language model. Show a listening indicator from the state.

**Audio input.** Where the audio comes from is set when the service
starts, see `/v1/info`:

- `MICROPHONE`: the microphone of the device the service runs on. The
  client only starts and stops captures.
- `CLIENT`: the client streams each capture's audio over a WebSocket, as
  16 bit little-endian mono samples at the model's sample rate, see
  [Audio streaming](#audio-streaming).

## Endpoints

All bodies are JSON with camelCase names.

| Request | Response |
|---|---|
| `GET /v1/info` | The model, its sample rate, the audio input and the language model |
| `GET /v1/state` | The dialog state, the capture, and the card shown |
| `GET /v1/events` | The event stream, see [Events](#events) |
| `POST /v1/recognition/start-capture` `{"initiator": "TAP_TO_TALK"}` | `202` `{"dialogRequestId": "…", "audioStream": {…} or null}`, `409` unless `IDLE`, `422` for `COMMANDS` if the service recognizes none |
| `POST /v1/recognition/stop-capture` `{"dialogRequestId": "…"}`, the body optional | `202` `{"dialogRequestId": "…"}`, `409` unless `LISTENING` to that capture |
| `POST /v1/display/card-dismissed` `{"cardId": "…"}` | `204`, `409` if that card is not shown |
| `POST /v1/intents/{intent}` | The card that answers the intent, such as `PlayKeyboard`, as if it was heard, without slots, also shown in a `ShowCard` event; `404` for an intent the service does not answer, `409` unless `IDLE`. For a button, such as the client's that opens the keyboard |
| `WS /v1/audio/{streamId}` | Takes a capture's audio, see [Audio streaming](#audio-streaming) |
| `GET /v1/wifi` | The Wi-Fi status, see [Wi-Fi](#wi-fi) |
| `POST /v1/wifi/scan` | `{"networks": […]}`, the networks in reach, see [Wi-Fi](#wi-fi) |
| `POST /v1/wifi/connect` `{"ssid": "…", "password": "…"}` | `{"status": {…}, "error": null}` once joined, see [Wi-Fi](#wi-fi) |
| `GET /v1/songs/{songId}` | A song's notes, `{"seconds": …, "loop": …, "notes": [{"start", "seconds", "frequency", "partials", "volume", "attack", "release", "patch"}, …], "patches": {…}}` in seconds and Hz, which a `SongCard` plays, `404` for no such song. A note's `patch`, if not null, names one of the `patches`, the Roland D-50 patches that play those notes, see below |
| `GET /v1/patches/{patch}` | A Roland D-50 patch's parameters, which a `KeyboardCard` plays, `404` for no such patch, see below |

Errors have a `detail` with the reason, such as
`{"detail": "Cannot start a capture while LISTENING."}`.

## Events

`GET /v1/events` is a stream of
[server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html),
such as from a browser's `EventSource`. Each is named `Topic.Action` and
has a message as data:

```
event: Recognition.CaptureFinished
data: {"id": 42, "topic": "Recognition", "action": "CaptureFinished", "timestamp": "2026-10-10T10:41:48.290263Z", "payload": {"dialogRequestId": "9cf0…", "text": "turn on the radio", "reason": "END_OF_SPEECH"}}
```

The `id` grows by one with each message. The stream starts with the
dialog state and the card shown, if any, so a client that connects or
reconnects can draw the screen from them. It sends a keep-alive comment
when there are no events for a while.

| Event | Payload | When |
|---|---|---|
| `Recognition.DialogStateChanged` | `state`, `dialogRequestId` (null when `IDLE`) | The dialog state changes |
| `Recognition.TranscriptUpdated` | `dialogRequestId`, `text` | The transcript so far changes. It can still change, also earlier words with a language model. |
| `Recognition.EndOfSpeechDetected` | `dialogRequestId` | A `TAP_TO_TALK` capture heard the end of speech |
| `Recognition.CaptureFinished` | `dialogRequestId`, `text`, `reason` | The final transcript. `text` is empty if no speech was recognized, and for a `COMMANDS` capture. |
| `Recognition.IntentRecognized` | `dialogRequestId`, `intent`, `score`, `slots` | The capture asked for an intent, such as `GetWeather` with `{"city": "hamburg"}`. The card with the answer follows. |
| `Recognition.CommandRecognized` | `dialogRequestId`, `command`, `score`, `text` | A `COMMANDS` capture heard a command, such as `LEFT`, as soon as it was said. `text` is its transcript so far. |
| `Display.ShowCard` | `card` | Show the card, see [Display cards](#display-cards) |
| `Display.ClearCard` | `cardId` | Dismiss the card with the id |
| `Audio.StartAudioStream` | `streamId`, `url`, `sampleRate`, `encoding` (`LINEAR16`), `channels` (1) | Stream the capture's audio, with audio input from the client |
| `Audio.StopAudioStream` | `streamId` | Stop streaming |

`reason` is one of:

| Reason | The capture ended because |
|---|---|
| `END_OF_SPEECH` | A `TAP_TO_TALK` capture heard speech followed by a pause |
| `STOP_CAPTURE` | The client called stop-capture or closed its audio stream |
| `NO_SPEECH` | A `TAP_TO_TALK` capture recognized no speech in time |
| `MAX_DURATION` | It reached the longest duration |
| `AUDIO_TIMEOUT` | No audio arrived in time |
| `ERROR` | The service failed, see its log |

## Display cards

The service decides what the screen shows. `ShowCard` replaces the card
shown, and a card with the id of the card shown updates it in place. Each
card has a `type` and an `id`, the capture's `dialogRequestId`:

```json
{
  "type": "TranscriptCard",
  "id": "9cf0…",
  "title": "Listening…",
  "text": "turn on the",
  "isFinal": false
}
```

| Card | Fields | Shows |
|---|---|---|
| `TranscriptCard` | `title`, `text`, `isFinal` | The transcript, growing while `isFinal` is false. Show words that can still change as such. |
| `NoticeCard` | `title`, `text` | A notice, such as "Sorry, I didn't catch that" when no speech was recognized |
| `GameCard` | `title`, `level` | Play Superball at the `level`, from 1, the easiest, to 5, the fastest, see [Commands](#commands). It stays until the client dismisses it when the game ends, or another card replaces it. |
| `BrightnessCard` | `title`, `percent` | The brightness of the screen, just set, in percent |
| `KeyboardCard` | `title`, `patch`, `url` | Show a keyboard that plays the Roland D-50 patch at `url`, titled with its name. It stays until the user closes it, or another card replaces it. |
| `SongCard` | `title`, `url`, `seconds`, `loop` | Play the song whose notes are at `url`, which lasts `seconds`, over and over with `loop`. It stays until the client dismisses it when the song has played, or for a loop when the user closes it, or another card replaces it. |
| `WeatherCard` | `location`, `condition`, `description`, `isDay`, `temperature`, `feelsLike`, `humidity`, `windSpeed`, `days`, `temperatureUnit`, `windSpeedUnit` | The current weather and the forecast. `condition` is one of `CLEAR`, `PARTLY_CLOUDY`, `CLOUDY`, `FOG`, `DRIZZLE`, `RAIN`, `SNOW` and `THUNDERSTORM`, and `days` lists the `date`, `condition`, `high` and `low` of today and the next three days. A day's condition is how its daylight hours look: rainy or snowy if 0.5 mm or more falls in two or more of them, light rain below 1 mm, and otherwise clear, partly cloudy or cloudy by its sunshine. |

The lifecycle of a capture's card:

1. start-capture shows a `TranscriptCard` titled "Listening…", which
   replaces any card shown.
2. It is updated as the transcript changes.
3. When the capture finishes, it becomes the answer to the intent the
   capture asked for, such as a `WeatherCard`. Otherwise it becomes the
   final transcript, titled "You said", or a `NoticeCard` if no speech
   was recognized or no audio arrived.
4. 8 seconds later the service sends `ClearCard`, or 30 seconds later
   for an answer, which takes longer to read, but never for a
   `GameCard`, a `SongCard` or a `KeyboardCard`. A new capture replaces the
   card sooner, except a `COMMANDS` capture, which shows no cards.

When the user dismisses the card, such as by tapping its X, tell the
service with card-dismissed. It then neither updates the card nor sends
`ClearCard` for it, even while the capture goes on.

Clients should render the card types they know and ignore the others, so
new types can be added.

## Intents

An intent is something an utterance asks for, with a name and sample
utterances, such as `GetWeather` with "what's the weather", "how's the
weather today" and "what's the weather in {city}". A sample can have a
slot, such as `{city}`, which stands for any of its values, and the
value said is passed to the intent's handler.

When a capture finishes, the service aligns the model's output for
the whole utterance to each sample, character by character, and scores
how close the best alignment comes to the model's own best guess. If the
closest sample scores at least -1.5, as a mean log probability per
character, the capture asked for its intent. For a slot, the sample is
first aligned with a wildcard in its place, which finds where the slot
was said, and each value is aligned to that part of the utterance. The
best values then compete with the other samples as whole sentences.

Matching the model's output rather than the transcript finds the intent
also when the transcript is misspelled, as it often is for short commands
with the models trained on audiobooks. With the 3,750 hour model after
its first epoch, on synthesized questions for the weather in 16 voices,
-1.5 matched 99% of them, 4% of other short commands such as "what's the
news?", and none of 300 LibriSpeech sentences. Asked for the weather in
one of 16 cities, it found the intent in 382 of 384 questions and the
city in 86%, but missed names the model hears badly, such as Leipzig.
The 100 hour model is less confident, so scores lower: at -2 it matched
as many questions and found the city in 82%. `--intent-threshold` sets
the score. The log names the closest intent and its score also when it
does not match.

`IntentRecognized` has the intent's `slots`, such as
`{"city": "hamburg"}`.

| Intent | Answer |
|---|---|
| `PlayGame` | A `GameCard` at the level asked for, such as "play superball level three", or else 1 |
| `SetBrightness` | A `BrightnessCard`, having set the screen's brightness to the percent asked for in tens, such as "set the brightness to fifty percent", or to "full" or "minimum" brightness, never below 10%, at which the screen still shows. With `--backlight`, see below |
| `PlayKeyboard` | A `KeyboardCard` of the synthesizer's keyboard, playing Fantasia, such as "play the synthesizer" or "keyboard" |
| `PlaySong` | A `SongCard` of the song asked for by one of its names, such as "play twinkle twinkle", of the songs in `--songs`, see below |
| `GetWeather` | A `WeatherCard` from [Open-Meteo](https://open-meteo.com) for the city asked for, or else for `--weather-location`, by default Berlin. A city is looked up in `speech/backend/places.tsv`, also when misspelled, such as "breman" for Bremen, and else with Open-Meteo, which `--weather-location` can name any place for. A place Open-Meteo does not know gives a `NoticeCard`, and so does no internet access. |

`places.tsv` lists the towns of Germany, Austria and Switzerland with
50,000 people or more, the world's capitals and its cities of 3 million
or more, with the names people say for them, from
[GeoNames](https://www.geonames.org) under CC BY 4.0. `speech/backend/places.py`
regenerates it.

`--songs` names a directory of songs, JSON files of their notes, see
`speech/backend/songs.py` and `examples/songs`, where
`examples/songs/from_midi.py` makes one from a MIDI file. The client
plays them live with the voice of an additive synthesizer, each harmonic
a sine with a detuned pair, which thickens it like a chorus, see
`web/assistant/synth.js`. Without songs, `PlaySong` is not recognized.

A song, or a voice of it, can instead name a Roland D-50 patch, such as
`"patch": "fantasia"`, the D-50's first factory patch. The client then
plays its notes as the D-50 does, with linear arithmetic synthesis: each
note's two tones of two partials, synthesized as Roland's LA chip makes
them, after [Munt](https://github.com/munt/munt)'s emulation of it, with
the D-50's own envelope times, cutoff and LFO rates measured on recordings
of it, and the tones' equalizer and chorus, and a reverb, in stereo. A patch is the
D-50's own parameters, in `speech/backend/patches`, which
`examples/songs/d50_patch.py` makes from a D-50 bank dump. The D-50's PCM
samples are Roland's, so the client has stand-ins for those the patches
use, see `speech/backend/d50.py`, which refuses a patch it cannot play.

`--backlight` names the screen's backlight in `/sys/class/backlight`, or
`auto` for the only one, which the service then needs to be allowed to
write. With `--state-dir`, it keeps the brightness there and sets it again
when it starts, also once a backlight that appears later, as its driver
loads, can be written, see `speech/backend/brightness.py`.

To add an intent, give `speech.intents.Intent` its samples, in the
model's characters, and the values of its slots, and pass a handler whose
`card(card_id, slots)` returns the card to show, see
`speech/backend/weather.py` and `backend.py`.

## Commands

A `COMMANDS` capture recognizes short commands as they are said, to steer
a game, without a tap for each. Superball's are `LEFT`, from "left", "go
left", "move left" or "turn left", and `RIGHT` likewise, also as "wight",
since young children often say an r as a w.

A pause of 0.3 seconds after speech ends each command, and the next
starts a new transcript, while the model keeps streaming. Pauses within a
command, such as between "go" and "left", were at most 0.44 seconds and
mostly about 0.1. As a command is said, its transcript so far is
matched to the commands as intents are, see [Intents](#intents), also
with a space before or after, which a transcript that starts within a
stream has. So a command is reported as soon as it matches, often before
the pause, and once.

With the 3,750 hour model after its first epoch and the training
transcripts' LM, on an M1 MacBook Air, 16 synthesized voices each said 7
commands and 3 other words, a second apart, streamed in real time in
100 ms chunks. It recognized 92 of the 112 commands, never the wrong
direction, and none of the other words, "yay", "oh no" and "wow". It
reported them 0.21 seconds after the end of the word at the median, and
0.43 at most. Most of those it missed were said by the few unusual
voices.

With only two commands, almost any short utterance is close to one of
them. The kid's "yes" sounded to the model nearly like "lest", and
moved the unicorn left. So everyday words, such as "yes", "no", "yay" and
"hello", compete with the commands, and an utterance closer to one of
them moves nothing, see `speech/backend/game.py`. Words that sound like a
command as a child says it, such as "what" for "wight", are left out.
`--command-threshold` sets the lowest score of a command, -1.25 by
default, apart from the intents'.

With the 3,750 hour model after two epochs, the competing words halved the
other words taken for a command among Speech Commands' 10,197 test clips
of other words, from 31 to 14, also among those not in the list, and
recognized 86% rather than 88% of its 808 clips of "left" and "right".
Recorded by `examples/speech_commands/record.py`, the kid said each
command 12 times and 12 other words: streamed as the game does, it
recognized 23 of the 24 commands, none in the wrong direction, and moved on
none of the other words, where it had taken "yes" for "left" before.

After a third epoch, which also trained on Speech Commands' training
clips, the model took more other words for a command at the earlier
threshold of -1.5, 34 of the test clips rather than 14. -1.25 takes 21, and
recognizes 678 rather than 701 of the 808 commands. Of the kid's
clips, decoded whole by `examples/speech_commands/evaluate.py`, both
recognized 20 of the 24 commands, as the model after two epochs recognized
21 at -1.5, and took none of the other words for one.

## Audio streaming

With audio input from the client, start-capture's response and the
`StartAudioStream` event both say where to stream the audio:

```json
{"streamId": "235c…", "url": "/v1/audio/235c…", "sampleRate": 16000, "encoding": "LINEAR16", "channels": 1}
```

Open a WebSocket to the `url` and send binary messages of 16 bit
little-endian mono samples at `sampleRate`, such as 100 ms each, as they
are recorded. The service closes the WebSocket when the capture ends.

To end a `HOLD_TO_TALK` capture, close the WebSocket after the last audio
rather than call stop-capture. stop-capture travels on another
connection, so it can arrive before audio that is still on its way, which
is then lost. Closing the WebSocket ends the capture after all the audio
sent.

A stream that no capture takes is refused with HTTP 403, and text
messages close the WebSocket with code 1003.

## Wi-Fi

With `--wifi-interface wlan0`, the client can join a Wi-Fi network, so a
device without a keyboard, such as a Pi with a touchscreen, gets online
for the weather. `/v1/info` then has `"wifi": true`, and without the
option the endpoints answer `404`. The service talks to wpa_supplicant
through its control socket in `/run/wpa_supplicant`, or the directory
`--wifi-control-dir` names, so its user must be in the group of
wpa_supplicant's `ctrl_interface`, such as
`ctrl_interface=DIR=/run/wpa_supplicant GROUP=netdev`. wpa_supplicant saves
the network in its own config file, which needs `update_config=1`, and a
DHCP client, such as systemd-networkd's, gets the address.

`GET /v1/wifi` returns the status:

```json
{"state": "CONNECTED", "ssid": "Home", "ipAddress": "192.168.1.20", "configured": true}
```

| `state` | |
|---|---|
| `CONNECTED` | Joined a network, with an IPv4 address |
| `CONNECTING` | Joining a network, or joined and waiting for an address |
| `DISCONNECTED` | No network joined |
| `UNAVAILABLE` | wpa_supplicant does not answer |

`configured` says whether a network is saved, which the device joins when
in reach.

`POST /v1/wifi/scan` scans for the networks in reach, which takes up to
some 10 seconds, the first time after wpa_supplicant started the
longest. They come one per name, with the strongest signal of its access
points, the strongest first, and hidden networks are left out:

```json
{"networks": [{"ssid": "Home", "secured": true, "signal": 4}]}
```

`signal` is in bars from 0 to 4. It answers `409` while a network is
joined, and `503` when wpa_supplicant does not answer.

`POST /v1/wifi/connect` joins a network, with an empty or missing
`password` for an open one. A password has 8 to 63 printable ASCII
characters, as WPA requires, and a name at most 32 bytes, or it answers
`422`. It returns once the network is joined and has an address, or
failed, which takes up to a minute, with the status and the error:

| `error` | |
|---|---|
| `null` | Joined, with an address |
| `WRONG_PASSWORD` | The network refused the password |
| `NOT_JOINED` | The network was not joined in time, such as when it is out of reach |
| `NO_ADDRESS` | Joined, but no address came in time |

A network joined replaces the networks saved before, with its key, and for
WPA3 its passphrase, which WPA3 needs. A network not joined is forgotten,
and the networks saved before stay, so a mistyped password never loses a
network that works. One network is joined at a time, and a second
request answers `409`.

Anyone who reaches the service can change its Wi-Fi, so only enable this
on a service that only the device itself, or trusted networks, reach.

## Flows

Tap to talk, with the microphone of the service's device:

```
Client                                     Service
  | POST start-capture TAP_TO_TALK  ------>  |
  |  <------ 202 {dialogRequestId}           |
  |  <------ DialogStateChanged LISTENING    |
  |  <------ ShowCard "Listening…"           |
  |                                          |  the user speaks
  |  <------ TranscriptUpdated               |
  |  <------ ShowCard (updated)              |  repeated as the text grows
  |  <------ EndOfSpeechDetected             |  a pause
  |  <------ DialogStateChanged THINKING     |
  |  <------ CaptureFinished {text, reason}  |
  |  <------ ShowCard "You said"             |
  |  <------ DialogStateChanged IDLE         |
  |  <------ ClearCard                       |  8 seconds later
```

A game, with audio from the client:

```
Client                                     Service
  | POST start-capture TAP_TO_TALK  ------>  |  "play superball"
  |  <------ …                               |
  |  <------ CaptureFinished {text, reason}  |
  |  <------ IntentRecognized PlayGame       |
  |  <------ ShowCard GameCard {level}       |  the game starts
  |  <------ DialogStateChanged IDLE         |
  | POST start-capture COMMANDS  --------->  |
  |  <------ 202 {dialogRequestId, audioStream}
  |  <------ DialogStateChanged LISTENING    |
  | WS audioStream.url, audio  ----------->  |
  |  <------ CommandRecognized LEFT          |  "left"
  |  <------ CommandRecognized RIGHT         |  "right", and so on
  | close the WebSocket  ----------------->  |  the game is over
  |  <------ StopAudioStream                 |
  |  <------ CaptureFinished STOP_CAPTURE    |
  |  <------ DialogStateChanged IDLE         |
  | POST card-dismissed {cardId}  -------->  |  after showing the score
```

Hold to talk, with audio from the client:

```
Client                                     Service
  | POST start-capture HOLD_TO_TALK ------>  |  the button is pressed
  |  <------ 202 {dialogRequestId, audioStream}
  |  <------ DialogStateChanged LISTENING    |
  |  <------ ShowCard "Listening…"           |
  |  <------ StartAudioStream                |
  | WS audioStream.url, audio  ----------->  |
  |  <------ TranscriptUpdated, ShowCard, …
  | close the WebSocket  ----------------->  |  the button is released
  |  <------ DialogStateChanged THINKING     |
  |  <------ StopAudioStream                 |
  |  <------ CaptureFinished {text, reason}  |
  |  <------ ShowCard "You said"             |
  |  <------ DialogStateChanged IDLE         |
```

## The browser client

`web/assistant`, served at `/`, is a client for a screen such as the
Pi's, made for a young child: a sunny meadow with a rainbow, drifting
clouds, a clock, and a unicorn whose speech bubble suggests what to ask,
with a round rainbow microphone button in the upper right corner.
Tapping it, or pressing Space, starts a `TAP_TO_TALK` capture, and
tapping it while listening stops it. A rising chime sounds when listening
starts and a falling one when it ends, also for captures with the
service's microphone, but not while a game listens for its commands. The
model ignores the chime that the microphone picks up. The button shows the dialog state,
glowing with the loudness while listening and its rainbow ring spinning
while listening and thinking, and the page shows the cards the service
sends, for which the unicorn steps aside. A round button with keys in the
lower right corner, as big as the microphone button, opens the
synthesizer's keyboard, with `POST /v1/intents/PlayKeyboard`, for a child
who would rather press than ask; it shows only on the landing page, with
no card, game or capture. The button in the upper left
corner switches to a night sky, with a sleepy moon, stars and fireflies,
and the page remembers the choice. Above an answer, such as the
weather, it shows what was asked, from `CaptureFinished`. The rounded Fredoka font is in
`web/assistant/fonts` under the SIL Open Font License, so the page needs
no internet.

Asked to "play superball", it plays a game on the meadow: the unicorn
trots to the bottom, a countdown starts, and flowers fall for it to
catch, faster from level to level. Saying "left" or "right" moves the
unicorn one of five lanes, as do the arrow keys and the big buttons at the
sides, which light up when a command is heard. Each flower caught is a
point, every 10 points the next level begins, up to 5, and three flowers
missed end the game, which then shows the score. A flower falls no
farther from the one before than the unicorn can move in time, so each
can be caught. Tapping the microphone button
ends it sooner. While playing, the page listens with a `COMMANDS`
capture, see `web/assistant/game.js`.

Where the service can join a Wi-Fi network, a Wi-Fi button next to the
day and night button shows whether one is joined, with a badge when not,
and opens a sheet to join one: pick a network from the list, or name one
that is not listed, type its password on the keyboard below it, as a
touchscreen without a keyboard needs, and wait while the service joins
it. The sheet opens by itself when no network is saved. See
`web/assistant/wifi.js`.

Asked to play a song, it shows the song's title with notes that dance
while it plays and a rainbow that fills as it goes, plays it live with a
synthesizer on Web Audio's audio thread, an AudioWorklet, and dismisses
the card once it has played, or for a song that
loops, plays it until the card is closed. Closing the card, or tapping
the microphone button, stops it.

With audio input from the client the page records from the browser's
microphone, which browsers only allow on `localhost` or over HTTPS. With
the service's microphone it only starts and stops captures.

## Running the service

```
uv run --no-default-groups backend.py best_model-int8.onnx --lm lm.npz --host 0.0.0.0
```

The model is a `.onnx` file from `export.py`, see the README's Raspberry
Pi section. `--lm-weight`, `--word-bonus` and `--unk-penalty` set how the
LM counts in the beam search, by default as in `stream.py` and `eval.py`;
`tune_lm.py` finds the best for a model and an LM, see the README's
Language model section. `--audio-input client` takes the audio from the client
instead of the microphone, and `--mic-device` picks the microphone. The
service listens on port 8080 of `127.0.0.1`, so only the device itself
can reach it, unless `--host` says otherwise. The API has no
authentication, so only open it to networks you trust.

To run it as a service that starts with the Pi, adapt
`deploy/speech-backend.service`, see the comments in it.
