import asyncio
import contextlib
import dataclasses
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
from speech.backend import engine
from speech.backend.app import create_app
from speech.backend.engine import Microphone, Recognizer, Settings
from speech.backend.messages import AudioInputKind
from speech.models import CTC
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


def make_recognizer(path, audio_input=AudioInputKind.CLIENT, **settings):
    model = streaming.load(path)
    new_search = streaming.search_factory(model.blank, model.char_to_int)
    settings = Settings(**({"card_timeout": 0.2, "audio_timeout": 1.0} | settings))
    return Recognizer(model, new_search, audio_input, settings, model_name="tiny")


def transcribe(path, audio):
    """The final transcript of a Transcriber fed the audio in chunks."""
    transcriber = streaming.Transcriber(streaming.load(path))
    for start in range(0, len(audio), CHUNK):
        transcriber.push(audio[start : start + CHUNK])
    transcriber.push(audio[:0], final=True)
    return transcriber.text(final=True).strip()


@contextlib.asynccontextmanager
async def running(recognizer, microphone=None):
    """Serves the recognizer on a free port and yields its address."""
    config = uvicorn.Config(
        create_app(recognizer, microphone), port=0, log_level="warning"
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
async def service(recognizer, microphone=None):
    """Serves the recognizer, subscribed to its events."""
    async with (
        running(recognizer, microphone) as address,
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
            }
            assert await svc.get("/v1/state") == {
                "dialogState": "IDLE",
                "dialogRequestId": None,
                "card": None,
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
    assert state == {"dialogState": "IDLE", "dialogRequestId": None, "card": None}
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


def test_display_card_cleared(model_path):
    audio, _ = wave.array_from_wave("test0.wav")

    async def run():
        recognizer = make_recognizer(model_path)
        async with service(recognizer) as svc:
            cleared = "/v1/display/card-dismissed"
            assert (await svc.post(cleared, {"cardId": "x"})).status_code == 409
            start = await svc.start("HOLD_TO_TALK")
            await svc.events.until("StartAudioStream")
            # The user dismisses the card while speaking.
            response = await svc.post(cleared, {"cardId": start["dialogRequestId"]})
            assert response.status_code == 204
            socket = await stream_audio(svc.address, start, audio)
            await socket.close()
            events = await svc.events.until("CaptureFinished")
            events += await svc.events.until("DialogStateChanged")
            await asyncio.sleep(recognizer.settings.card_timeout * 2)
            return events, svc.events.queue.empty(), await svc.get("/v1/state")

    events, no_more, state = asyncio.run(run())
    # The card is neither shown again nor cleared, but the events go on.
    assert "TranscriptUpdated" in actions(events)
    assert "ShowCard" not in actions(events)
    assert no_more
    assert state["card"] is None


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
