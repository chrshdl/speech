# Speech recognizer service API

`backend.py` runs streaming speech recognition as a service, such as on a
Raspberry Pi that a head unit or another client talks to. The client
starts and stops captures, and the service sends events with the
transcript and the display cards to show. The client renders what the
service tells it, and the service decides what to show and for how long.

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

**Capture.** One utterance, from start-capture to the final transcript.
Its `dialogRequestId` is in each of its events, and is the id of its
display cards. There is one capture at a time.

**Initiator.** How the capture was started, which decides how it ends:

| Initiator | Ends |
|---|---|
| `HOLD_TO_TALK` | When the client calls stop-capture, or closes its audio stream, as the button is released |
| `TAP_TO_TALK` | At the end of speech: a pause of 1 second after the transcript starts. Also with stop-capture. |

Either ends after 60 seconds, and when no audio arrives for 5 seconds.
A `TAP_TO_TALK` capture that recognizes no speech in 8 seconds ends too.
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
| `POST /v1/recognition/start-capture` `{"initiator": "TAP_TO_TALK"}` | `202` `{"dialogRequestId": "…", "audioStream": {…} or null}`, `409` unless `IDLE` |
| `POST /v1/recognition/stop-capture` `{"dialogRequestId": "…"}`, the body optional | `202` `{"dialogRequestId": "…"}`, `409` unless `LISTENING` to that capture |
| `POST /v1/display/card-dismissed` `{"cardId": "…"}` | `204`, `409` if that card is not shown |
| `WS /v1/audio/{streamId}` | Takes a capture's audio, see [Audio streaming](#audio-streaming) |

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
| `Recognition.CaptureFinished` | `dialogRequestId`, `text`, `reason` | The final transcript, the last event of every capture. `text` is empty if no speech was recognized. |
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

The lifecycle of a capture's card:

1. start-capture shows a `TranscriptCard` titled "Listening…", which
   replaces any card shown.
2. It is updated as the transcript changes.
3. When the capture finishes, it becomes the final transcript, titled
   "You said", or a `NoticeCard` if no speech was recognized or no audio
   arrived.
4. 8 seconds later the service sends `ClearCard`. A new capture replaces
   the card sooner.

When the user dismisses the card, such as by tapping its X, tell the
service with card-dismissed. It then neither updates the card nor sends
`ClearCard` for it, even while the capture goes on.

Clients should render the card types they know and ignore the others, so
new types can be added.

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

## Running the service

```
uv run --no-default-groups backend.py best_model-int8.onnx --lm lm.npz --host 0.0.0.0
```

The model is a `.onnx` file from `export.py`, see the README's Raspberry
Pi section. `--audio-input client` takes the audio from the client
instead of the microphone, and `--mic-device` picks the microphone. The
service listens on port 8080 of `127.0.0.1`, so only the device itself
can reach it, unless `--host` says otherwise. The API has no
authentication, so only open it to networks you trust.

To run it as a service that starts with the Pi, adapt
`deploy/speech-backend.service`, see the comments in it.
