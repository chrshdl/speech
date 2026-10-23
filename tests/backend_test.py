import asyncio
import contextlib
import dataclasses
import datetime
import json
import threading

import httpx
import numpy as np
import pytest
import shared
import torch
import uvicorn
import websockets

from speech import export, loader, streaming
from speech.backend import drums, engine, game, keyboard, radio, songs
from speech.backend.app import create_app
from speech.backend.engine import Microphone, Recognizer, Settings
from speech.backend.messages import AudioInputKind, NoticeCard, RadioCard, VolumeCard
from speech.intents import Intent, Match
from speech.models import CTC, word_lm
from speech.utils import wave

CHUNK = 1600


def export_model(path, blank_only=False):
    """Exports a tiny model. A blank_only model never recognizes speech."""
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    model = CTC(preproc.input_dim, preproc.vocab_size, shared.model_config)
    if blank_only:
        with torch.no_grad():
            model.fc.bias[model.blank] = 100
    export.export(model, preproc, str(path), int8=False)
    return str(path)


@pytest.fixture
def model_path(tmp_path):
    return export_model(tmp_path / "model.onnx")


def make_recognizer(
    path,
    audio_input=AudioInputKind.CLIENT,
    intents=None,
    handlers=None,
    voice=None,
    **settings,
):
    model = streaming.load(path)
    new_search = streaming.search_factory(model.blank, model.char_to_int)
    settings = Settings(
        **(
            {"card_timeout": 0.2, "brief_card_timeout": 0.2, "audio_timeout": 1.0}
            | settings
        )
    )
    return Recognizer(
        lambda: streaming.Transcriber(model, new_search()),
        model.sample_rate,
        audio_input,
        settings,
        model_name="tiny",
        intents=intents,
        handlers=handlers,
        voice=voice,
    )


def transcribe(path, audio):
    """The final transcript of a Transcriber fed the audio in chunks."""
    transcriber = streaming.Transcriber(streaming.load(path))
    for start in range(0, len(audio), CHUNK):
        transcriber.push(audio[start : start + CHUNK])
    transcriber.push(audio[:0], final=True)
    return transcriber.text(final=True).strip()


@contextlib.asynccontextmanager
async def running(recognizer, microphone=None, radio=None):
    """Serves the recognizer on a free port and yields its address."""
    config = uvicorn.Config(
        create_app(recognizer, microphone, radio=radio), port=0, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"127.0.0.1:{port}"
    finally:
        recognizer.shutdown()
        server.should_exit = True
        await task


class Events:
    def __init__(self, address):
        """Collects the service's events, after its first, the state."""
        self.url = f"http://{address}/v1/events"
        self.queue = asyncio.Queue()

    async def __aenter__(self):
        self.task = asyncio.create_task(self._read())
        self.first = await self.next()
        return self

    async def __aexit__(self, *args):
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task

    async def _read(self):
        async with (
            httpx.AsyncClient(timeout=None) as client,
            client.stream("GET", self.url) as response,
        ):
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    await self.queue.put(json.loads(line[6:]))
        await self.queue.put(None)

    async def next(self, timeout=5):
        return await asyncio.wait_for(self.queue.get(), timeout)

    async def until(self, action, timeout=5):
        """The events up to and including the next with the action."""
        events = []
        while True:
            event = await self.next(timeout)
            assert event is not None, f"The stream ended before {action}"
            events.append(event)
            if event["action"] == action:
                return events


def actions(events):
    """The actions of the events, without repeats of updates in a row."""
    names = [e["action"] for e in events]
    return [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]


async def stream_audio(address, start, audio):
    """Streams the audio as start-capture's audioStream says."""
    audio_stream = start["audioStream"]
    assert audio_stream["sampleRate"] == 16000
    url = f"ws://{address}{audio_stream['url']}"
    socket = await websockets.connect(url)
    for i in range(0, len(audio), CHUNK):
        await socket.send(audio[i : i + CHUNK].astype("<i2").tobytes())
    return socket


@dataclasses.dataclass
class Service:
    address: str
    events: Events
    client: httpx.AsyncClient

    async def post(self, path, body=None):
        return await self.client.post(path, json=body)

    async def start(self, initiator):
        """Starts a capture and returns start-capture's response."""
        response = await self.post(
            "/v1/recognition/start-capture", {"initiator": initiator}
        )
        assert response.status_code == 202
        return response.json()

    async def get(self, path):
        return (await self.client.get(path)).json()


@contextlib.asynccontextmanager
async def service(recognizer, microphone=None, radio=None):
    """Serves the recognizer, subscribed to its events."""
    async with (
        running(recognizer, microphone, radio) as address,
        Events(address) as events,
        httpx.AsyncClient(base_url=f"http://{address}") as client,
    ):
        yield Service(address, events, client)


def test_info_and_state(model_path):
    async def run():
        async with service(make_recognizer(model_path)) as svc:
            assert await svc.get("/v1/info") == {
                "model": "tiny",
                "sampleRate": 16000,
                "audioInput": "CLIENT",
                "languageModel": None,
                "wifi": False,
            }
            assert await svc.get("/v1/state") == {
                "dialogState": "IDLE",
                "dialogRequestId": None,
                "card": None,
                "playing": None,
                "popup": None,
            }
            assert svc.events.first["payload"] == {
                "state": "IDLE",
                "dialogRequestId": None,
            }

    asyncio.run(run())


def test_hold_to_talk(model_path):
    audio, _ = wave.array_from_wave("test0.wav")
    expected = transcribe(model_path, audio)
    assert expected

    async def run():
        async with service(make_recognizer(model_path)) as svc:
            start = await svc.start("HOLD_TO_TALK")
            socket = await stream_audio(svc.address, start, audio)
            # Releasing the button ends the audio stream, and with it the
            # capture after all the audio.
            await socket.close()
            events = await svc.events.until("ClearCard")
            state = await svc.get("/v1/state")
        return start, events, state

    start, events, state = asyncio.run(run())
    dialog = start["dialogRequestId"]
    assert actions(events)[:3] == [
        "DialogStateChanged",
        "ShowCard",
        "StartAudioStream",
    ]
    assert actions(events)[-7:] == [
        "ShowCard",
        "DialogStateChanged",
        "StopAudioStream",
        "CaptureFinished",
        "ShowCard",
        "DialogStateChanged",
        "ClearCard",
    ]
    updates = [e["payload"] for e in events if e["action"] == "TranscriptUpdated"]
    assert updates
    assert all(u["dialogRequestId"] == dialog for u in updates)
    by_action = {e["action"]: e["payload"] for e in events}
    assert by_action["CaptureFinished"] == {
        "dialogRequestId": dialog,
        "text": expected,
        "reason": "STOP_CAPTURE",
    }
    assert by_action["StopAudioStream"] == {
        "streamId": start["audioStream"]["streamId"]
    }
    assert events[-3]["payload"]["card"] == {
        "type": "TranscriptCard",
        "id": dialog,
        "title": "You said",
        "text": expected,
        "isFinal": True,
    }
    assert by_action["ClearCard"] == {"cardId": dialog}
    assert state == {
        "dialogState": "IDLE",
        "dialogRequestId": None,
        "card": None,
        "playing": None,
        "popup": None,
    }
    ids = [e["id"] for e in events]
    assert ids == sorted(ids)


def test_events_from_start(model_path):
    """The first events of a capture, and a late subscriber's first ones."""

    async def run():
        async with service(make_recognizer(model_path)) as svc:
            start = await svc.start("TAP_TO_TALK")
            first = await svc.events.until("StartAudioStream")
            async with Events(svc.address) as late:
                snapshot = [late.first, await late.next()]
            await svc.post("/v1/recognition/stop-capture")
            await svc.events.until("CaptureFinished")
        return start, first, snapshot

    start, first, snapshot = asyncio.run(run())
    dialog = start["dialogRequestId"]
    assert [e["action"] for e in first] == [
        "DialogStateChanged",
        "ShowCard",
        "StartAudioStream",
    ]
    assert first[0]["payload"] == {"state": "LISTENING", "dialogRequestId": dialog}
    assert first[1]["payload"]["card"] == {
        "type": "TranscriptCard",
        "id": dialog,
        "title": "Listening…",
        "text": "",
        "isFinal": False,
    }
    assert first[2]["payload"] == start["audioStream"]
    assert [e["payload"] for e in snapshot] == [e["payload"] for e in first[:2]]


def test_tap_to_talk_end_of_speech(model_path):
    audio, _ = wave.array_from_wave("test0.wav")

    async def run():
        # The random model rarely pauses, so end at the first text.
        async with service(make_recognizer(model_path, end_of_speech=0)) as svc:
            start = await svc.start("TAP_TO_TALK")
            await stream_audio(svc.address, start, audio)
            return await svc.events.until("CaptureFinished")

    events = asyncio.run(run())
    assert actions(events)[-4:] == [
        "EndOfSpeechDetected",
        "DialogStateChanged",
        "StopAudioStream",
        "CaptureFinished",
    ]
    assert events[-1]["payload"]["reason"] == "END_OF_SPEECH"
    assert events[-1]["payload"]["text"]


@pytest.mark.parametrize(
    ("initiator", "settings", "reason"),
    [
        ("TAP_TO_TALK", {"no_speech_timeout": 0.5}, "NO_SPEECH"),
        ("HOLD_TO_TALK", {"max_capture": 0.5}, "MAX_DURATION"),
    ],
)
def test_capture_limits(tmp_path, initiator, settings, reason):
    path = export_model(tmp_path / "blank.onnx", blank_only=True)
    audio, _ = wave.array_from_wave("test0.wav")

    async def run():
        async with service(make_recognizer(path, **settings)) as svc:
            start = await svc.start(initiator)
            await stream_audio(svc.address, start, audio)
            events = await svc.events.until("CaptureFinished")
            return start, events[-1], await svc.events.next()

    start, finished, card = asyncio.run(run())
    assert finished["payload"]["reason"] == reason
    assert finished["payload"]["text"] == ""
    assert card["payload"]["card"] == {
        "type": "NoticeCard",
        "id": start["dialogRequestId"],
        "title": "Sorry, I didn't catch that",
        "text": "Please try again.",
    }


def test_audio_timeout(model_path):
    async def run():
        async with service(make_recognizer(model_path, audio_timeout=0.3)) as svc:
            # The client never streams the audio.
            await svc.start("TAP_TO_TALK")
            events = await svc.events.until("CaptureFinished")
            return events[-1], await svc.events.next()

    finished, card = asyncio.run(run())
    assert finished["payload"]["reason"] == "AUDIO_TIMEOUT"
    assert card["payload"]["card"]["title"] == "No audio"


def test_conflicts(model_path):
    async def run():
        async with service(make_recognizer(model_path)) as svc:
            stop = "/v1/recognition/stop-capture"
            assert (await svc.post(stop)).status_code == 409

            # There is no wake word engine.
            wakeword = await svc.post(
                "/v1/recognition/start-capture", {"initiator": "WAKEWORD"}
            )
            assert wakeword.status_code == 422

            start = await svc.start("HOLD_TO_TALK")
            again = await svc.post(
                "/v1/recognition/start-capture", {"initiator": "TAP_TO_TALK"}
            )
            assert again.status_code == 409
            assert again.json() == {"detail": "Cannot start a capture while LISTENING."}
            other = await svc.post(stop, {"dialogRequestId": "other"})
            assert other.status_code == 409

            # Audio streams that no capture takes are refused.
            with pytest.raises(websockets.InvalidStatus):
                await websockets.connect(f"ws://{svc.address}/v1/audio/other")
            url = f"ws://{svc.address}{start['audioStream']['url']}"
            socket = await websockets.connect(url)
            with pytest.raises(websockets.InvalidStatus):
                await websockets.connect(url)
            # Audio must be binary.
            await socket.send("hello")
            with pytest.raises(websockets.ConnectionClosedError) as closed:
                await socket.recv()
            assert closed.value.rcvd.code == 1003

            response = await svc.post(
                stop, {"dialogRequestId": start["dialogRequestId"]}
            )
            assert response.status_code == 202
            assert response.json() == {"dialogRequestId": start["dialogRequestId"]}
            await svc.events.until("CaptureFinished")

    asyncio.run(run())


class FakeInputStream:
    """Plays test0.wav to the callback in 100 ms chunks, as a microphone."""

    def __init__(self, samplerate, channels, dtype, blocksize, device, callback):
        assert (samplerate, channels, dtype) == (16000, 1, "int16")
        self.device = 0
        self.blocksize = blocksize
        self.callback = callback
        self.stopped = threading.Event()

    def start(self):
        audio, _ = wave.array_from_wave("test0.wav")

        def play():
            start = 0
            while not self.stopped.wait(0.01):
                chunk = audio[start : start + self.blocksize]
                if len(chunk) < self.blocksize:
                    chunk = np.zeros(self.blocksize, np.int16)
                self.callback(chunk[:, None], len(chunk), None, None)
                start += self.blocksize

        threading.Thread(target=play, daemon=True).start()

    def close(self):
        self.stopped.set()


def test_microphone(model_path, monkeypatch):
    monkeypatch.setattr(engine.sounddevice, "InputStream", FakeInputStream)
    monkeypatch.setattr(engine.sounddevice, "query_devices", lambda d: {"name": "fake"})
    recognizer = make_recognizer(model_path, AudioInputKind.MICROPHONE)
    microphone = Microphone(recognizer)

    async def run():
        async with service(recognizer, microphone) as svc:
            start = await svc.start("HOLD_TO_TALK")
            assert start["audioStream"] is None
            await svc.events.until("TranscriptUpdated")
            await svc.post("/v1/recognition/stop-capture")
            return await svc.events.until("CaptureFinished")

    events = asyncio.run(run())
    assert "StartAudioStream" not in actions(events)
    assert events[-1]["payload"]["reason"] == "STOP_CAPTURE"
    assert events[-1]["payload"]["text"]
    assert microphone.stream is None


def test_shutdown_ends_streams(model_path):
    recognizer = make_recognizer(model_path)

    async def run():
        async with service(recognizer) as svc:
            start = await svc.start("HOLD_TO_TALK")
            socket = await stream_audio(svc.address, start, np.zeros(CHUNK, np.int16))
            await svc.events.until("StartAudioStream")
            recognizer.shutdown()
            # The event stream and the audio stream end.
            assert await svc.events.next() is None
            with pytest.raises(websockets.ConnectionClosedOK):
                await socket.recv()

    asyncio.run(run())
    assert recognizer.state == "IDLE"


class FixedIntents:
    """Finds every utterance as close to an intent as the score says."""

    threshold = -2.0

    def __init__(self, score):
        self.score = score

    def closest_heard(self, transcriber):
        assert len(transcriber.probs()) > 0
        return Match(Intent("GetTime", ("what time is it",)), self.score, "")


class Clock:
    """Answers with a card, or fails."""

    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False

    async def card(self, card_id, slots):
        assert slots == {}
        if self.fail:
            raise OSError("offline")
        return NoticeCard(id=card_id, title="Time", text="It is noon.")

    async def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("score", "fail", "title"),
    [
        (-0.5, False, "Time"),
        (-0.5, True, "Sorry, I can't answer that right now"),
        (-3.0, False, "You said"),
    ],
)
def test_intents(model_path, score, fail, title):
    audio, _ = wave.array_from_wave("test0.wav")
    clock = Clock(fail)
    recognizer = make_recognizer(
        model_path,
        intents=FixedIntents(score),
        # An intent it cannot answer, such as the brightness without a
        # backlight, does not keep the others from closing.
        handlers={"SetBrightness": None, "GetTime": clock},
        answer_card_timeout=0.4,
    )

    async def run():
        async with service(recognizer) as svc:
            start = await svc.start("HOLD_TO_TALK")
            socket = await stream_audio(svc.address, start, audio)
            await socket.close()
            events = await svc.events.until("CaptureFinished")
            events += await svc.events.until("ClearCard")
            return start, events

    start, events = asyncio.run(run())
    dialog = start["dialogRequestId"]
    after = actions(events)[actions(events).index("CaptureFinished") :]
    shown = [e for e in events if e["action"] == "ShowCard"][-1]["payload"]["card"]
    assert shown["title"] == title
    if score >= FixedIntents.threshold:
        assert after == [
            "CaptureFinished",
            "IntentRecognized",
            "ShowCard",
            "DialogStateChanged",
            "ClearCard",
        ]
        intent = next(e for e in events if e["action"] == "IntentRecognized")
        assert intent["payload"] == {
            "dialogRequestId": dialog,
            "intent": "GetTime",
            "score": score,
            "slots": {},
        }
        # Answers stay longer than transcripts.
        times = {e["action"]: e["timestamp"] for e in events}
        shown_at = datetime.datetime.fromisoformat(times["ShowCard"])
        cleared_at = datetime.datetime.fromisoformat(times["ClearCard"])
        assert (cleared_at - shown_at).total_seconds() >= 0.4
    else:
        assert "IntentRecognized" not in after
    assert clock.closed


class TalkingClock(Clock):
    """Answers with a card, and says it in two sentences."""

    def speech(self, card):
        return "It is noon. Time for lunch."


class SlowClock(TalkingClock):
    """Answers once the test lets it."""

    def __init__(self):
        super().__init__()
        self.answer = asyncio.Event()

    async def card(self, card_id, slots):
        await self.answer.wait()
        return await super().card(card_id, slots)


class RecordingVoice:
    """Says each text in a sentence of silence, and records it."""

    sample_rate = 22050

    def __init__(self):
        self.said = []

    def sentences(self, text):
        self.said.append(text)
        yield np.zeros(100, "<i2").tobytes()


def test_display_card_cleared(model_path):
    """
    Dismissing the card of the capture listening ends it at once, without
    an answer, and dismissing it while the answer is found shows and says
    none.
    """
    audio, _ = wave.array_from_wave("test0.wav")
    voice = RecordingVoice()
    clock = SlowClock()
    recognizer = make_recognizer(
        model_path,
        intents=FixedIntents(-0.5),
        handlers={"GetTime": clock},
        voice=voice,
    )

    async def run():
        async with service(recognizer) as svc:
            cleared = "/v1/display/card-dismissed"
            assert (await svc.post(cleared, {"cardId": "x"})).status_code == 409
            start = await svc.start("HOLD_TO_TALK")
            await svc.events.until("StartAudioStream")
            # The user dismisses the card while speaking.
            response = await svc.post(cleared, {"cardId": start["dialogRequestId"]})
            assert response.status_code == 204
            cancelled = await svc.events.until("CaptureFinished")
            cancelled += await svc.events.until("DialogStateChanged")
            # And while the answer is found.
            start = await svc.start("HOLD_TO_TALK")
            socket = await stream_audio(svc.address, start, audio)
            await socket.close()
            answered = []
            while not answered or answered[-1]["payload"].get("state") != "THINKING":
                answered += await svc.events.until("DialogStateChanged")
            await svc.post(cleared, {"cardId": start["dialogRequestId"]})
            clock.answer.set()
            answered += await svc.events.until("DialogStateChanged")
            await asyncio.sleep(recognizer.settings.card_timeout * 2)
            return (
                cancelled,
                answered,
                svc.events.queue.empty(),
                await svc.get("/v1/state"),
            )

    cancelled, answered, no_more, state = asyncio.run(run())
    assert actions(cancelled) == [
        "StopAudioStream",
        "CaptureFinished",
        "DialogStateChanged",
    ]
    assert cancelled[1]["payload"]["reason"] == "CANCELLED"
    assert cancelled[-1]["payload"]["state"] == "IDLE"
    # Answered, but neither shown, nor spoken, nor cleared.
    after = actions(answered)[actions(answered).index("CaptureFinished") :]
    assert after == ["CaptureFinished", "IntentRecognized", "DialogStateChanged"]
    assert voice.said == []
    assert no_more
    assert state["card"] is None


class GatedVoice:
    """
    Synthesizes a sentence each time the test lets it, as samples of the
    sentence's number.
    """

    sample_rate = 22050

    def __init__(self):
        self.next = threading.Semaphore(0)
        self.said = []

    def sentences(self, text):
        self.said.append(text)
        for i, _ in enumerate(text.split(". "), 1):
            self.next.acquire()
            yield np.full(1000, i, "<i2").tobytes()


async def ask(svc, audio):
    """Asks for an intent, and returns the events up to its Speak."""
    start = await svc.start("HOLD_TO_TALK")
    socket = await stream_audio(svc.address, start, audio)
    await socket.close()
    return start, await svc.events.until("Speak")


def test_speaks_the_answer(model_path):
    """
    The answer is spoken, its audio streamed as each sentence is
    synthesized.
    """
    audio, _ = wave.array_from_wave("test0.wav")
    voice = GatedVoice()
    recognizer = make_recognizer(
        model_path,
        intents=FixedIntents(-0.5),
        handlers={"GetTime": TalkingClock()},
        voice=voice,
    )

    async def run():
        async with service(recognizer) as svc:
            start, events = await ask(svc, audio)
            events += await svc.events.until("DialogStateChanged")
            speak = next(e for e in events if e["action"] == "Speak")["payload"]
            chunks = []
            async with svc.client.stream("GET", speak["url"]) as response:
                headers = response.headers
                stream = response.aiter_raw()
                # Each sentence arrives once synthesized, not at the end.
                for _ in range(2):
                    voice.next.release()
                    chunks.append(await asyncio.wait_for(anext(stream), 5))
                chunks.extend([chunk async for chunk in stream])
            again = await svc.client.get(speak["url"])
            return start, events, speak, headers, chunks, again

    start, events, speak, headers, chunks, again = asyncio.run(run())
    after = actions(events)[actions(events).index("CaptureFinished") :]
    assert after == [
        "CaptureFinished",
        "IntentRecognized",
        "ShowCard",
        "Speak",
        "DialogStateChanged",
    ]
    dialog = start["dialogRequestId"]
    assert speak == {
        "speechId": speak["speechId"],
        "cardId": dialog,
        "text": "It is noon. Time for lunch.",
        "url": f"/v1/speech/{speak['speechId']}",
        "sampleRate": 22050,
        "encoding": "LINEAR16",
        "channels": 1,
    }
    assert headers["content-type"] == "audio/L16;rate=22050;channels=1"
    samples = np.frombuffer(b"".join(chunks), "<i2")
    assert np.array_equal(samples, np.repeat([1, 2], 1000))
    # Synthesized, it can be streamed again, as by another client.
    assert np.array_equal(np.frombuffer(again.content, "<i2"), samples)


@pytest.mark.parametrize("stop", ["capture", "dismissed"])
def test_speaking_stops(model_path, stop):
    """
    A capture starting, or the card dismissed, stops the speech, also its
    stream, before the rest is synthesized.
    """
    audio, _ = wave.array_from_wave("test0.wav")
    voice = GatedVoice()
    recognizer = make_recognizer(
        model_path,
        intents=FixedIntents(-0.5),
        handlers={"GetTime": TalkingClock()},
        voice=voice,
    )

    async def run():
        async with service(recognizer) as svc:
            start, events = await ask(svc, audio)
            await svc.events.until("DialogStateChanged")
            speak = events[-1]["payload"]
            async with svc.client.stream("GET", speak["url"]) as response:
                stream = response.aiter_raw()
                voice.next.release()
                first = await asyncio.wait_for(anext(stream), 5)
                if stop == "capture":
                    await svc.start("TAP_TO_TALK")
                else:
                    dismissed = await svc.post(
                        "/v1/display/card-dismissed",
                        {"cardId": start["dialogRequestId"]},
                    )
                    assert dismissed.status_code == 204
                stopped = await svc.events.until("StopSpeaking")
                rest = [chunk async for chunk in stream]
            voice.next.release()
            later = await svc.client.get(speak["url"])
            return speak, first, stopped, rest, later

    speak, first, stopped, rest, later = asyncio.run(run())
    assert stopped[-1]["payload"] == {"speechId": speak["speechId"]}
    if stop == "capture":
        # Stopped before listening starts.
        assert actions(stopped) == ["StopSpeaking"]
    assert np.array_equal(np.frombuffer(first, "<i2"), np.full(1000, 1))
    assert rest == []
    assert later.status_code == 404


def test_answers_unspoken_without_a_voice(model_path):
    """Without a voice, or for a card its handler says nothing for."""
    audio, _ = wave.array_from_wave("test0.wav")
    recognizer = make_recognizer(
        model_path,
        intents=FixedIntents(-0.5),
        handlers={"GetTime": TalkingClock()},
    )

    async def run():
        async with service(recognizer) as svc:
            start = await svc.start("HOLD_TO_TALK")
            socket = await stream_audio(svc.address, start, audio)
            await socket.close()
            events = await svc.events.until("CaptureFinished")
            events += await svc.events.until("DialogStateChanged")
            return events, await svc.client.get("/v1/speech/none")

    events, missing = asyncio.run(run())
    assert "Speak" not in actions(events)
    assert missing.status_code == 404


class LoudTranscriber:
    """
    A transcriber that hears each loud stretch of the audio as the next of
    the words, and the quiet since as the pause.
    """

    def __init__(self, words):
        self.words = list(words)
        self.word = ""
        self.samples = 0
        self.quiet = 0
        # The audio of each transcript, when it was cleared.
        self.heard = []

    def push(self, audio, final=False):
        self.samples += len(audio)
        if len(audio) and np.abs(audio).max() > 1000:
            self.word = self.word or self.words.pop(0)
            self.quiet = 0
        else:
            self.quiet += len(audio)

    @property
    def heard_speech(self):
        return bool(self.word)

    @property
    def pause_seconds(self):
        return self.quiet / 16000

    def text(self, final=False):
        return self.word

    def clear(self):
        self.heard.append(self.samples)
        self.word, self.samples, self.quiet = "", 0, 0


class WordCommands:
    """Recognizes the commands whose first sample a transcript has."""

    threshold = -1.0

    def closest_heard(self, transcriber):
        for intent in game.COMMANDS:
            if intent.samples[0] in transcriber.text():
                return Match(intent, 0.0, intent.samples[0])
        return Match(game.COMMANDS[0], -5.0, "left")


def test_commands():
    """
    A COMMANDS capture reports each command as soon as it is recognized,
    starts a new transcript after the pause that ends it, and shows no
    cards.
    """
    rng = np.random.default_rng(0)

    def said(seconds):
        return rng.normal(0, 3000, int(seconds * 16000)).astype(np.int16)

    def pause(seconds):
        return np.zeros(int(seconds * 16000), np.int16)

    audio = np.concatenate(
        [pause(0.5), said(0.4), pause(0.6), said(0.4), pause(0.6), said(0.4)]
        + [pause(0.6)]
    )
    transcriber = LoudTranscriber(["left", "hello", "go right"])
    recognizer = Recognizer(
        lambda: transcriber,
        16000,
        AudioInputKind.CLIENT,
        Settings(audio_timeout=1.0),
        commands=WordCommands(),
    )

    async def run():
        async with service(recognizer) as svc:
            start = await svc.start("COMMANDS")
            await stream_audio(svc.address, start, audio)
            events = await svc.events.until("CommandRecognized")
            events += await svc.events.until("CommandRecognized")
            await svc.post(
                "/v1/recognition/stop-capture",
                {"dialogRequestId": start["dialogRequestId"]},
            )
            events += await svc.events.until("CaptureFinished")
            return start, events, await svc.get("/v1/state")

    start, events, state = asyncio.run(run())
    dialog = start["dialogRequestId"]
    commands = [e["payload"] for e in events if e["action"] == "CommandRecognized"]
    assert commands == [
        {"dialogRequestId": dialog, "command": "LEFT", "score": 0.0, "text": "left"},
        {
            "dialogRequestId": dialog,
            "command": "RIGHT",
            "score": 0.0,
            "text": "go right",
        },
    ]
    assert "ShowCard" not in actions(events)
    assert events[-1]["payload"] == {
        "dialogRequestId": dialog,
        "text": "",
        "reason": "STOP_CAPTURE",
    }
    assert state == {
        "dialogState": "IDLE",
        "dialogRequestId": None,
        "card": None,
        "playing": None,
        "popup": None,
    }
    # Each command had a transcript of its own: 0.4 seconds of it, the
    # pause after it that ended it, and before it, what was left of the
    # pause after the one before.
    assert len(transcriber.heard) == 3
    assert all(n <= 1.3 * 16000 for n in transcriber.heard)


class PlayIntents(FixedIntents):
    def closest_heard(self, transcriber):
        return Match(
            game.INTENT, self.score, "play superball level {level}", {"level": "three"}
        )


def test_game_card_stays(model_path):
    """A game's card stays until the client dismisses it."""
    audio, _ = wave.array_from_wave("test0.wav")
    recognizer = make_recognizer(
        model_path,
        intents=PlayIntents(-0.5),
        handlers={game.INTENT.name: game.Game()},
        answer_card_timeout=0.1,
    )

    async def run():
        async with service(recognizer) as svc:
            start = await svc.start("HOLD_TO_TALK")
            socket = await stream_audio(svc.address, start, audio)
            await socket.close()
            events = await svc.events.until("CaptureFinished")
            events += await svc.events.until("DialogStateChanged")
            await asyncio.sleep(0.4)
            shown = await svc.get("/v1/state")
            commands = await svc.post(
                "/v1/recognition/start-capture", {"initiator": "COMMANDS"}
            )
            card_id = shown["card"]["id"]
            dismissed = await svc.post(
                "/v1/display/card-dismissed", {"cardId": card_id}
            )
            return start, events, shown, commands, dismissed, await svc.get("/v1/state")

    start, events, shown, commands, dismissed, state = asyncio.run(run())
    after = actions(events)[actions(events).index("CaptureFinished") :]
    assert after == [
        "CaptureFinished",
        "IntentRecognized",
        "ShowCard",
        "DialogStateChanged",
    ]
    assert shown["card"] == {
        "type": "GameCard",
        "id": start["dialogRequestId"],
        "title": "Superball",
        "level": 3,
    }
    # Without commands to recognize, there is no COMMANDS capture.
    assert commands.status_code == 422
    assert dismissed.status_code == 204
    assert state["card"] is None


def test_answers_an_intent_for_a_button(model_path):
    """
    An intent asked for without speech, such as by the client's button
    for the keyboard, shows its card as when heard.
    """
    recognizer = make_recognizer(
        model_path,
        handlers={keyboard.INTENT.name: keyboard.Keyboard()},
        answer_card_timeout=0.1,
    )

    async def run():
        async with service(recognizer) as svc:
            answered = await svc.post("/v1/intents/PlayKeyboard")
            events = await svc.events.until("ShowCard")
            await asyncio.sleep(0.3)
            shown = await svc.get("/v1/state")
            unknown = await svc.post("/v1/intents/GetTime")
            await svc.start("HOLD_TO_TALK")
            busy = await svc.post("/v1/intents/PlayKeyboard")
            return answered, events, shown, unknown, busy

    answered, events, shown, unknown, busy = asyncio.run(run())
    assert answered.status_code == 200
    card = answered.json()
    assert card == {
        "type": "KeyboardCard",
        "id": card["id"],
        "title": "Fantasia",
        "patch": "fantasia",
        "url": "/v1/patches/fantasia",
    }
    assert events[-1]["payload"]["card"] == card
    # It stays until dismissed, past the timeout of an answer.
    assert shown["dialogState"] == "IDLE" and shown["card"] == card
    assert unknown.status_code == 404
    assert busy.status_code == 409
    assert busy.json()["detail"] == "Cannot answer PlayKeyboard while LISTENING."


class Station:
    """Plays a radio card, which it updates with each song the test puts."""

    def __init__(self):
        self.songs = asyncio.Queue()
        self.handlers = {radio.PLAY.name: self, radio.STOP.name: radio.Stop()}
        self.streaming = False

    async def card(self, card_id, slots):
        return RadioCard(
            id=card_id,
            title="Radio",
            url="/v1/radio/stream",
            sample_rate=44100,
            channels=2,
            logo="logo",
            song="One",
        )

    async def updates(self, card):
        while True:
            yield card.model_copy(update={"song": await self.songs.get()})

    async def stream(self):
        """Two samples at a time, until closed."""
        self.streaming = True
        try:
            while True:
                yield np.array([1, 2], "<i2").tobytes()
                await asyncio.sleep(0.01)
        finally:
            self.streaming = False

    async def image(self, key):
        return (b"<svg/>", "image/svg+xml") if key == "logo" else None

    async def close(self):
        pass


class Meter:
    """Turns the volume."""

    async def card(self, card_id, slots):
        return VolumeCard(id=card_id, title="Volume", percent=50)

    async def close(self):
        pass


def shown(events):
    """The cards shown by the events, as (type, id, song)."""
    return [
        (card["type"], card["id"], card.get("song"))
        for card in (e["payload"]["card"] for e in events if e["action"] == "ShowCard")
    ]


def test_radio_plays_on_beneath_cards(model_path):
    """
    The radio's card shows again once a card over it goes, with the song
    that played on meanwhile, and goes when the radio is stopped.
    """
    station = Station()
    recognizer = make_recognizer(
        model_path,
        handlers={**station.handlers, "GetTime": Clock()},
        answer_card_timeout=0.2,
    )

    async def run():
        async with service(recognizer) as svc:
            played = (await svc.post("/v1/intents/PlayRadio")).json()
            events = await svc.events.until("ShowCard")
            turned = (await svc.post("/v1/intents/GetTime")).json()
            events += await svc.events.until("ShowCard")
            station.songs.put_nowait("Two")
            await asyncio.sleep(0.05)
            covered = await svc.get("/v1/state")
            # A client that connects now plays the radio beneath the card.
            async with Events(svc.address) as late:
                snapshot = [await late.next(), await late.next()]
            # The clock's card goes.
            events += await svc.events.until("ShowCard")
            station.songs.put_nowait("Three")
            events += await svc.events.until("ShowCard")
            stopped = await svc.post("/v1/intents/StopRadio")
            events += await svc.events.until("ClearCard")
            station.songs.put_nowait("Four")
            await asyncio.sleep(0.05)
            more = not svc.events.queue.empty()
            state = await svc.get("/v1/state")
            return played, turned, events, covered, snapshot, stopped, more, state

    played, turned, events, covered, snapshot, stopped, more, state = asyncio.run(run())
    radio_id, clock_id = played["id"], turned["id"]
    assert shown(events) == [
        ("RadioCard", radio_id, "One"),
        ("NoticeCard", clock_id, None),
        ("RadioCard", radio_id, "Two"),
        ("RadioCard", radio_id, "Three"),
    ]
    # Replaced rather than cleared.
    assert actions(events).count("ClearCard") == 1
    assert covered["card"]["id"] == clock_id
    assert covered["playing"]["song"] == "Two"
    assert shown(snapshot) == [
        ("RadioCard", radio_id, "Two"),
        ("NoticeCard", clock_id, None),
    ]
    assert stopped.status_code == 200 and stopped.json() is None
    assert events[-1]["payload"] == {"cardId": radio_id}
    assert not more
    assert (state["card"], state["playing"]) == (None, None)


def test_card_that_stays_ends_the_radio(model_path):
    station = Station()
    recognizer = make_recognizer(
        model_path,
        handlers={**station.handlers, keyboard.INTENT.name: keyboard.Keyboard()},
    )

    async def run():
        async with service(recognizer) as svc:
            played = (await svc.post("/v1/intents/PlayRadio")).json()
            await svc.post("/v1/intents/PlayKeyboard")
            events = await svc.events.until("ClearCard")
            return played, events, await svc.get("/v1/state")

    played, events, state = asyncio.run(run())
    # Cleared after the keyboard shows.
    assert actions(events)[-2:] == ["ShowCard", "ClearCard"]
    assert events[-2]["payload"]["card"]["type"] == "KeyboardCard"
    assert events[-1]["payload"] == {"cardId": played["id"]}
    assert state["playing"] is None


def test_radio_dismissed(model_path):
    """
    Dismissing a card over the radio shows the radio again, and dismissing
    the radio beneath a card stops it.
    """
    station = Station()
    recognizer = make_recognizer(
        model_path,
        handlers={**station.handlers, "GetTime": Clock()},
        answer_card_timeout=0.2,
    )

    async def run():
        async with service(recognizer) as svc:
            dismissed = "/v1/display/card-dismissed"
            played = (await svc.post("/v1/intents/PlayRadio")).json()
            turned = (await svc.post("/v1/intents/GetTime")).json()
            events = await svc.events.until("ShowCard")
            events += await svc.events.until("ShowCard")
            first = await svc.post(dismissed, {"cardId": turned["id"]})
            events += await svc.events.until("ShowCard")
            turned = (await svc.post("/v1/intents/GetTime")).json()
            events += await svc.events.until("ShowCard")
            beneath = await svc.post(dismissed, {"cardId": played["id"]})
            again = await svc.post(dismissed, {"cardId": played["id"]})
            events += await svc.events.until("ClearCard")
            return played, events, first, beneath, again, await svc.get("/v1/state")

    played, events, first, beneath, again, state = asyncio.run(run())
    assert [card[0] for card in shown(events)] == [
        "RadioCard",
        "NoticeCard",
        "RadioCard",
        "NoticeCard",
    ]
    assert (first.status_code, beneath.status_code) == (204, 204)
    assert again.status_code == 409
    # The clock's card goes, with nothing beneath it any more.
    assert events[-1]["payload"]["cardId"] != played["id"]
    assert (state["card"], state["playing"]) == (None, None)


def test_radio_streams_while_it_plays(model_path):
    """
    The radio's audio streams while it plays, and ends when it stops, which
    ends the station's stream. Its images come through the service.
    """
    station = Station()
    recognizer = make_recognizer(model_path, handlers=station.handlers)

    async def run():
        async with service(recognizer, radio=station) as svc:
            before = await svc.client.get("/v1/radio/stream")
            await svc.post("/v1/intents/PlayRadio")
            async with svc.client.stream("GET", "/v1/radio/stream") as response:
                headers = response.headers
                chunks = response.aiter_raw()
                first = await asyncio.wait_for(anext(chunks), 5)
                await svc.post("/v1/intents/StopRadio")
                rest = [chunk async for chunk in chunks]
            await asyncio.sleep(0.05)
            logo = await svc.client.get("/v1/radio/images/logo")
            missing = await svc.client.get("/v1/radio/images/none")
            return before, headers, first, rest, logo, missing

    before, headers, first, rest, logo, missing = asyncio.run(run())
    assert before.status_code == 404
    assert headers["content-type"] == "audio/L16;rate=44100;channels=2"
    assert np.frombuffer(first, "<i2")[:2].tolist() == [1, 2]
    # Ended within a chunk or so of stopping.
    assert len(b"".join(rest)) <= 8
    assert not station.streaming
    assert (logo.status_code, logo.content) == (200, b"<svg/>")
    assert logo.headers["content-type"] == "image/svg+xml"
    assert missing.status_code == 404


def popups(events):
    """The popups shown by the events, as (type, id)."""
    return [
        (e["payload"]["card"]["type"], e["payload"]["card"]["id"])
        for e in events
        if e["action"] == "ShowPopup"
    ]


def test_popup_goes_sooner(model_path):
    """A popup read at a glance, such as the volume's, goes after its timeout."""
    recognizer = make_recognizer(
        model_path,
        handlers={"VolumeUp": Meter()},
        card_timeout=5,
        brief_card_timeout=0.3,
    )

    async def run():
        async with service(recognizer) as svc:
            await svc.post("/v1/intents/VolumeUp")
            events = await svc.events.until("ShowPopup")
            events += await svc.events.until("ClearCard", timeout=2)
            return events

    events = asyncio.run(run())
    times = [datetime.datetime.fromisoformat(e["timestamp"]) for e in events[-2:]]
    assert 0.3 <= (times[1] - times[0]).total_seconds() < 1


def test_popup_leaves_the_card_shown(model_path):
    """
    The volume's popup shows over a card that stays, such as the drums',
    which neither goes nor shows again, also for a client that connects
    meanwhile, and goes by itself or when dismissed.
    """
    recognizer = make_recognizer(
        model_path,
        handlers={"VolumeUp": Meter(), drums.INTENT.name: drums.Drums()},
        brief_card_timeout=0.3,
    )

    async def run():
        async with service(recognizer) as svc:
            played = (await svc.post("/v1/intents/PlayDrums")).json()
            events = await svc.events.until("ShowCard")
            turned = (await svc.post("/v1/intents/VolumeUp")).json()
            events += await svc.events.until("ShowPopup")
            covered = await svc.get("/v1/state")
            async with Events(svc.address) as late:
                snapshot = [await late.next(), await late.next()]
            events += await svc.events.until("ClearCard", timeout=2)
            gone = await svc.get("/v1/state")
            again = (await svc.post("/v1/intents/VolumeUp")).json()
            events += await svc.events.until("ShowPopup")
            dismissed = await svc.post(
                "/v1/display/card-dismissed", {"cardId": again["id"]}
            )
            await asyncio.sleep(0.5)
            more = not svc.events.queue.empty()
            return played, turned, events, covered, snapshot, gone, dismissed, more

    played, turned, events, covered, snapshot, gone, dismissed, more = asyncio.run(
        run()
    )
    assert actions(events) == ["ShowCard", "ShowPopup", "ClearCard", "ShowPopup"]
    assert shown(events) == [("DrumsCard", played["id"], None)]
    assert popups(events)[0] == ("VolumeCard", turned["id"])
    assert events[2]["payload"] == {"cardId": turned["id"]}
    assert covered["card"]["id"] == played["id"]
    assert covered["popup"]["id"] == turned["id"]
    assert actions(snapshot) == ["ShowCard", "ShowPopup"]
    assert (gone["card"]["id"], gone["popup"]) == (played["id"], None)
    # Dismissed by the client, it is not cleared again.
    assert dismissed.status_code == 204
    assert not more


def test_popup_over_the_radio(model_path):
    """The volume's popup leaves the radio's card as it is."""
    station = Station()
    recognizer = make_recognizer(
        model_path,
        handlers={**station.handlers, "VolumeUp": Meter()},
        brief_card_timeout=0.2,
    )

    async def run():
        async with service(recognizer) as svc:
            played = (await svc.post("/v1/intents/PlayRadio")).json()
            events = await svc.events.until("ShowCard")
            await svc.post("/v1/intents/VolumeUp")
            events += await svc.events.until("ClearCard", timeout=2)
            return played, events, await svc.get("/v1/state")

    played, events, state = asyncio.run(run())
    assert actions(events) == ["ShowCard", "ShowPopup", "ClearCard"]
    assert events[-1]["payload"]["cardId"] != played["id"]
    assert state["card"]["id"] == state["playing"]["id"] == played["id"]


class VolumeIntents(FixedIntents):
    """Hears each utterance as turning the volume up."""

    def closest_heard(self, transcriber):
        return Match(Intent("VolumeUp", ("volume up",)), self.score, "volume up")


def test_popup_answers_a_capture(model_path):
    """
    Asked for by voice over the radio, the volume's popup shows once the
    capture's transcript goes, over the radio's card shown again.
    """
    audio, _ = wave.array_from_wave("test0.wav")
    station = Station()
    recognizer = make_recognizer(
        model_path,
        intents=VolumeIntents(-0.5),
        handlers={**station.handlers, "VolumeUp": Meter()},
    )

    async def run():
        async with service(recognizer) as svc:
            played = (await svc.post("/v1/intents/PlayRadio")).json()
            await svc.events.until("ShowCard")
            start = await svc.start("HOLD_TO_TALK")
            socket = await stream_audio(svc.address, start, audio)
            await socket.close()
            events = await svc.events.until("CaptureFinished")
            events += await svc.events.until("ShowPopup")
            return played, start, events, await svc.get("/v1/state")

    played, start, events, state = asyncio.run(run())
    after = events[[e["action"] for e in events].index("CaptureFinished") :]
    assert actions(after) == [
        "CaptureFinished",
        "IntentRecognized",
        "ShowCard",
        "ShowPopup",
    ]
    assert shown(after) == [("RadioCard", played["id"], "One")]
    assert popups(after) == [("VolumeCard", start["dialogRequestId"])]
    assert state["card"]["id"] == played["id"]
    assert state["popup"]["id"] == start["dialogRequestId"]


class RadioIntents(FixedIntents):
    """Hears each utterance as stopping the radio, as close as the score says."""

    def closest_heard(self, transcriber):
        return Match(radio.STOP, self.score, "stop the radio")


def test_radio_stopped_by_voice(model_path):
    """
    A question over the radio shows its transcript and then the radio
    again, and asking to stop the radio clears both, for the landing page.
    """
    audio, _ = wave.array_from_wave("test0.wav")
    station = Station()
    intents = RadioIntents(-3.0)
    recognizer = make_recognizer(model_path, intents=intents, handlers=station.handlers)

    async def ask(svc):
        start = await svc.start("HOLD_TO_TALK")
        socket = await stream_audio(svc.address, start, audio)
        await socket.close()
        return start

    async def run():
        async with service(recognizer) as svc:
            played = (await svc.post("/v1/intents/PlayRadio")).json()
            await svc.events.until("ShowCard")
            asked = await ask(svc)
            events = await svc.events.until("CaptureFinished")
            # The transcript goes for the radio.
            events += await svc.events.until("ShowCard")
            events += await svc.events.until("ShowCard")
            intents.score = -0.5
            stop = await ask(svc)
            stopped = await svc.events.until("CaptureFinished")
            stopped += await svc.events.until("DialogStateChanged")
            return played, asked, events, stop, stopped, await svc.get("/v1/state")

    played, asked, events, stop, stopped, state = asyncio.run(run())
    assert shown(events)[-2:] == [
        ("TranscriptCard", asked["dialogRequestId"], None),
        ("RadioCard", played["id"], "One"),
    ]
    after = actions(stopped)[actions(stopped).index("CaptureFinished") :]
    assert after == [
        "CaptureFinished",
        "IntentRecognized",
        "ClearCard",
        "DialogStateChanged",
    ]
    cleared = [e["payload"]["cardId"] for e in stopped if e["action"] == "ClearCard"]
    assert cleared == [played["id"], stop["dialogRequestId"]]
    assert (state["card"], state["playing"]) == (None, None)


def test_client_is_revalidated(model_path):
    async def run():
        async with service(make_recognizer(model_path)) as svc:
            page = await svc.client.get("/assistant/")
            script = await svc.client.get("/assistant/app.js")
            again = await svc.client.get(
                "/assistant/app.js", headers={"If-None-Match": script.headers["etag"]}
            )
            return page, script, again

    page, script, again = asyncio.run(run())
    assert page.status_code == script.status_code == 200
    assert (
        page.headers["cache-control"] == script.headers["cache-control"] == "no-cache"
    )
    # An unchanged file is not sent again.
    assert again.status_code == 304


def test_lm_settings_reach_the_decoder(model_path, monkeypatch):
    """--lm-weight, --word-bonus and --unk-penalty set the beam search's."""
    import backend

    parser = backend.make_parser()
    defaults = parser.parse_args([str(model_path)])
    assert (defaults.lm_weight, defaults.word_bonus, defaults.unk_penalty) == (
        word_lm.LM_WEIGHT,
        word_lm.WORD_BONUS,
        word_lm.UNK_PENALTY,
    )
    args = parser.parse_args(
        [str(model_path), "--lm", "lm.npz", "--lm-weight", "0.5", "--word-bonus", "4"]
        + ["--unk-penalty=0"]
    )
    seen = {}

    def search_factory(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return lambda: None

    monkeypatch.setattr(backend.streaming, "search_factory", search_factory)
    # The test model lacks the characters of the intents' samples, which are
    # not what is tested here.
    monkeypatch.setattr(backend, "IntentRecognizer", lambda *args: None)
    monkeypatch.setattr(backend.game, "commands", lambda *args: None)
    backend.load(args, songs.Songs([]), None)
    assert seen["args"][2] == "lm.npz"
    assert seen["kwargs"] == {"lm_weight": 0.5, "word_bonus": 4.0, "unk_penalty": 0.0}


# The wake word.

LOUD = 5000


class FakeDetector:
    """Hears the wake word in each chunk that starts loud."""

    score = 1.0

    def __init__(self):
        self.resets = 0

    def push(self, audio):
        return bool(len(audio)) and audio[0] == LOUD

    def reset(self):
        self.resets += 1


class WakeTranscriber:
    """
    Hears each loud stretch of the audio as the next of the words, the
    first of them the wake word, whose labels, 0 and 1, the output
    probabilities have after a frame of another label.
    """

    frame_seconds = 0.03

    def __init__(self, words, says_wake_word=True):
        self.words = ["olivia", *words]
        self.said = []
        self.quiet = 0
        self.pushed = []
        self.dropped = None
        wake = [2, 3, 0, 3, 1] if says_wake_word else [2, 2, 2, 2, 2]
        self.path = [*wake, 3, 3, 3]

    def push(self, audio, final=False):
        self.pushed.append(audio.copy())
        if len(audio) and np.abs(audio).max() > 1000:
            if self.quiet or not self.said:
                self.said.append(self.words.pop(0))
            self.quiet = 0
        else:
            self.quiet += len(audio)

    @property
    def heard_speech(self):
        return bool(self.said)

    @property
    def pause_seconds(self):
        return self.quiet / 16000

    def text(self, final=False):
        return " ".join(self.said)

    def probs(self):
        probs = np.full((len(self.path), 4), 0.01, np.float32)
        probs[np.arange(len(self.path)), self.path] = 0.97
        return probs

    def drop(self, frames):
        self.dropped = frames
        self.said.pop(0)


def wake_recognizer(transcriber, **settings):
    settings = Settings(
        **({"end_of_speech": 0.2, "wake_word_lookback": 0.5} | settings)
    )
    return Recognizer(
        lambda: transcriber,
        16000,
        AudioInputKind.MICROPHONE,
        settings,
        wake_word=engine.WakeWord(FakeDetector(), np.array([0, 1]), 3),
    )


def chunk(level):
    return np.full(CHUNK, level, np.int16)


async def say(recognizer, *levels):
    for level in levels:
        recognizer.deliver(chunk(level))
        await asyncio.sleep(0.01)


def test_wake_word_starts_a_capture():
    transcriber = WakeTranscriber(["lights"])
    recognizer = wake_recognizer(transcriber)

    async def run():
        async with service(recognizer) as svc:
            # Quiet, the wake word, then the request and a pause.
            await say(recognizer, 1, 2, 3, 4, LOUD, 0, 6000, 6000, *[0] * 10)
            events = await svc.events.until("CaptureFinished")
            await svc.events.until("DialogStateChanged")
            # Heard again in new audio only.
            await say(recognizer, 0)
            return events

    events = asyncio.run(run())
    # Announced once the transcript confirms the wake word.
    assert actions(events)[:2] == ["DialogStateChanged", "ShowCard"]
    assert events[0]["payload"]["state"] == "LISTENING"
    assert events[1]["payload"]["card"]["title"] == "Listening…"
    # The capture starts with the audio before the wake word was heard,
    # as long as the lookback.
    lookback = transcriber.pushed[0]
    assert len(lookback) == 8000
    np.testing.assert_array_equal(
        lookback, np.concatenate([chunk(level) for level in (1, 2, 3, 4, LOUD)])
    )
    # The transcript starts after the wake word.
    assert transcriber.dropped == 5
    finished = events[-1]["payload"]
    assert finished["reason"] == "END_OF_SPEECH"
    assert finished["text"] == "lights"
    assert recognizer.wake_word.detector.resets == 1


class PlayingDetector(FakeDetector):
    """Hears the wake word in each chunk that starts a little louder."""

    def push(self, audio):
        return bool(len(audio)) and audio[0] == LOUD + 1


def test_wake_word_while_playing():
    """
    While the radio plays, the playing detector hears the wake word instead
    of the detector, each reset as it takes over, as it missed the audio.
    """
    transcriber = WakeTranscriber(["lights"])
    station = Station()
    recognizer = Recognizer(
        lambda: transcriber,
        16000,
        AudioInputKind.MICROPHONE,
        Settings(end_of_speech=0.2, wake_word_lookback=0.5),
        handlers=station.handlers,
        wake_word=engine.WakeWord(
            FakeDetector(), np.array([0, 1]), 3, playing_detector=PlayingDetector()
        ),
    )

    async def run():
        async with service(recognizer, radio=station) as svc:
            await say(recognizer, 0)
            await svc.post("/v1/intents/PlayRadio")
            await svc.events.until("ShowCard")
            # What the detector takes for the wake word is not heard.
            await say(recognizer, 0, LOUD, 0)
            await asyncio.sleep(0.1)
            ignored = recognizer.state
            await say(recognizer, LOUD + 1, 0, 6000, 6000, *[0] * 10)
            events = await svc.events.until("CaptureFinished")
            await svc.events.until("DialogStateChanged")
            await say(recognizer, 0)
            return ignored, events

    ignored, events = asyncio.run(run())
    assert ignored == "IDLE"
    assert events[-1]["payload"]["text"] == "lights"
    # Reset taking over, and after the capture.
    wake = recognizer.wake_word
    assert (wake.detector.resets, wake.playing_detector.resets) == (0, 2)


def test_wake_word_without_a_request_ends_quietly():
    transcriber = WakeTranscriber([])
    recognizer = wake_recognizer(transcriber, no_speech_timeout=0.3)

    async def run():
        async with service(recognizer) as svc:
            await say(recognizer, LOUD, *[0] * 16)
            events = await svc.events.until("CaptureFinished")
            events += await svc.events.until("DialogStateChanged")
            return events

    events = asyncio.run(run())
    finished = next(e for e in events if e["action"] == "CaptureFinished")
    assert finished["payload"]["reason"] == "NO_SPEECH"
    # The listening card is cleared, without a notice.
    assert actions(events)[-3:] == [
        "CaptureFinished",
        "ClearCard",
        "DialogStateChanged",
    ]
    assert "NoticeCard" not in json.dumps(events)


def test_wake_word_heard_by_mistake_ends_quietly():
    """A capture whose transcript does not say the wake word ends at once."""
    transcriber = WakeTranscriber(["lights"], says_wake_word=False)
    recognizer = wake_recognizer(transcriber)

    async def run():
        async with service(recognizer) as svc:
            await say(recognizer, LOUD)
            response = await svc.post(
                "/v1/recognition/start-capture", {"initiator": "TAP_TO_TALK"}
            )
            assert response.status_code == 409
            await say(recognizer, *[0] * 12)
            events = await svc.events.until("CaptureFinished")
            return events + await svc.events.until("DialogStateChanged")

    events = asyncio.run(run())
    assert transcriber.dropped is None
    assert actions(events) == [
        "DialogStateChanged",
        "ShowCard",
        "CaptureFinished",
        "ClearCard",
        "DialogStateChanged",
    ]
    finished = events[2]["payload"]
    assert finished["reason"] == "NO_WAKE_WORD"
    assert finished["text"] == ""
    assert events[-1]["payload"]["state"] == "IDLE"


def test_wake_word_only_while_idle(model_path):
    transcriber = WakeTranscriber(["lights"])
    recognizer = wake_recognizer(transcriber)

    async def run():
        async with service(recognizer) as svc:
            response = await svc.post(
                "/v1/recognition/start-capture", {"initiator": "WAKE_WORD"}
            )
            assert response.status_code == 422
            start = await svc.start("HOLD_TO_TALK")
            await say(recognizer, LOUD, 6000)
            await svc.post("/v1/recognition/stop-capture")
            events = await svc.events.until("CaptureFinished")
            return start, events

    start, events = asyncio.run(run())
    # The wake word said during a capture is part of it.
    assert events[-1]["payload"]["dialogRequestId"] == start["dialogRequestId"]
    assert events[-1]["payload"]["text"] == "olivia"
    with pytest.raises(ValueError, match="microphone"):
        Recognizer(
            lambda: transcriber,
            16000,
            AudioInputKind.CLIENT,
            wake_word=engine.WakeWord(FakeDetector(), np.array([0]), 3),
        )


class QuietVad:
    """Hears speech in loud chunks, and silence in quiet ones."""

    class Stream:
        def __init__(self):
            self.silence_seconds = 0.0
            self.seconds = 0.0
            # When each loud chunk started.
            self.loud = []

        def push(self, audio):
            if len(audio) and np.abs(audio).max() > 1000:
                self.silence_seconds = 0.0
                self.loud.append(self.seconds)
            else:
                self.silence_seconds += len(audio) / 16000
            self.seconds += len(audio) / 16000

        def speech_seconds(self, after=0.0):
            return 0.1 * sum(start >= after for start in self.loud)

    def stream(self, history=False):
        return self.Stream()


class EndlessTranscriber(LoudTranscriber):
    """Hears no pause ever, as when a model transcribes noise."""

    @property
    def pause_seconds(self):
        return 0.0


def test_vad_ends_speech():
    transcriber = EndlessTranscriber(["lights"])
    recognizer = Recognizer(
        lambda: transcriber,
        16000,
        AudioInputKind.MICROPHONE,
        Settings(end_of_speech=0.3),
        vad=QuietVad(),
    )

    async def run():
        async with service(recognizer) as svc:
            await svc.start("TAP_TO_TALK")
            await say(recognizer, 6000, 6000, *[0] * 5)
            return await svc.events.until("CaptureFinished")

    events = asyncio.run(run())
    assert "EndOfSpeechDetected" in actions(events)
    assert events[-1]["payload"]["reason"] == "END_OF_SPEECH"
    assert events[-1]["payload"]["text"] == "lights"


class LeftoverTranscriber(WakeTranscriber):
    """Keeps a letter of the wake word in the transcript, as "olivia e"."""

    def drop(self, frames):
        self.dropped = frames
        self.said[0] = "e"


def test_letter_of_the_wake_word_is_not_a_request():
    """
    With a voice activity detector, a pause after the wake word does not end
    the capture, also when the transcript kept a letter of it.
    """
    transcriber = LeftoverTranscriber(["lights"])
    recognizer = Recognizer(
        lambda: transcriber,
        16000,
        AudioInputKind.MICROPHONE,
        Settings(end_of_speech=0.3, wake_word_lookback=0.5, no_speech_timeout=5),
        wake_word=engine.WakeWord(FakeDetector(), np.array([0, 1]), 3),
        vad=QuietVad(),
    )

    async def run():
        async with service(recognizer) as svc:
            # The wake word, a pause of 2 seconds, the request and a pause.
            await say(recognizer, LOUD, *[0] * 20, 6000, 6000, *[0] * 5)
            return await svc.events.until("CaptureFinished")

    events = asyncio.run(run())
    assert events[-1]["payload"]["reason"] == "END_OF_SPEECH"
    assert events[-1]["payload"]["text"] == "e lights"
    # The letter alone was never shown.
    shown = [e["payload"]["card"]["text"] for e in events if e["action"] == "ShowCard"]
    assert shown == ["", "e lights"]
