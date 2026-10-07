import asyncio
import json
import urllib.error
import urllib.request

import pytest
import shared
import torch
from websockets.asyncio.client import connect

import serve
import speech
from speech import loader, streaming
from speech.models import CTC
from speech.models.word_lm import WordLM
from speech.utils import wave

CHUNK = 1600


@pytest.fixture
def root(tmp_path):
    """A directory with a tiny saved CTC model and a tiny LM."""
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    model = CTC(preproc.input_dim, preproc.vocab_size, shared.model_config)
    model_dir = tmp_path / "models" / "tiny"
    model_dir.mkdir(parents=True)
    speech.save(model, preproc, str(model_dir), tag="best")
    WordLM(["hello world", "hello hi", "world"]).save(
        str(tmp_path / "models" / "lm.npz")
    )
    return tmp_path


def expected_lines(model_dir, lm_path, audio, sample_rate):
    """The lines a Transcriber gives on the audio in the server's chunks."""
    model, preproc = speech.load(model_dir, tag="best")
    model.set_eval()
    search = streaming.search_factory(model.blank, preproc.char_to_int, lm_path)()
    transcriber = streaming.Transcriber(model, preproc, sample_rate, search)
    lines = []
    for start in range(0, len(audio), CHUNK):
        transcriber.push(audio[start : start + CHUNK])
        line = transcriber.take_line(serve.PAUSE)
        if line is not None:
            lines.append(line)
    transcriber.push(audio[:0], final=True)
    line = transcriber.text(final=True).strip()
    if line:
        lines.append(line)
    return lines


async def receive(socket):
    return json.loads(await socket.recv())


async def transcribe(url, model_dir, lm_path, audio):
    """Streams the audio like the page does and returns the lines."""
    async with connect(url) as socket:
        await socket.send(
            json.dumps({"type": "start", "model": model_dir, "lm": lm_path})
        )
        ready = await receive(socket)
        assert ready == {"type": "ready", "sample_rate": 16000}
        lines = []
        for start in range(0, len(audio), CHUNK):
            await socket.send(audio[start : start + CHUNK].astype("<i2").tobytes())
            while (message := await receive(socket))["type"] != "partial":
                assert message["type"] == "line"
                lines.append(message["text"])
            assert message["rtf"] > 0
        await socket.send(json.dumps({"type": "stop"}))
        while (message := await receive(socket))["type"] != "done":
            if message["type"] == "line":
                lines.append(message["text"])
            else:
                assert message == {"type": "partial", "text": ""}
        return lines


def get(url):
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.headers["Content-Type"], response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers["Content-Type"], error.read()


def test_serve(root, monkeypatch):
    model_dir = str(root / "models" / "tiny")
    lm_path = str(root / "models" / "lm.npz")
    audio, sample_rate = wave.array_from_wave("test0.wav")
    # The random model rarely pauses, so end a line after every chunk with
    # text to cover lines before the stop.
    monkeypatch.setattr(serve, "PAUSE", 0.0)

    async def run():
        async with serve.start_server(str(root), "127.0.0.1", 0, "cpu") as server:
            port = server.sockets[0].getsockname()[1]
            http = f"http://127.0.0.1:{port}"
            ws = f"ws://127.0.0.1:{port}/stream"

            status, content_type, body = await asyncio.to_thread(get, f"{http}/")
            assert (status, content_type) == (200, "text/html; charset=utf-8")
            assert b"Speech Recording" in body
            status, content_type, _ = await asyncio.to_thread(get, f"{http}/app.js")
            assert (status, content_type) == (200, "text/javascript")
            status, _, _ = await asyncio.to_thread(get, f"{http}/missing")
            assert status == 404
            _, _, body = await asyncio.to_thread(get, f"{http}/models")
            assert json.loads(body) == {"models": [model_dir], "lms": [lm_path]}

            for lm in [None, lm_path]:
                lines = await transcribe(ws, model_dir, lm, audio)
                assert len(lines) > 1
                assert lines == expected_lines(model_dir, lm, audio, sample_rate)

            async with connect(ws) as socket:
                # Audio before start.
                await socket.send(audio[:CHUNK].astype("<i2").tobytes())
                assert (await receive(socket))["type"] == "error"
                # Only the models found can be loaded.
                await socket.send(
                    json.dumps({"type": "start", "model": "/etc", "lm": None})
                )
                assert await receive(socket) == {
                    "type": "error",
                    "message": "Unknown model /etc",
                }

    asyncio.run(run())
