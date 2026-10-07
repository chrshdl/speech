"""
Serves a page for trying the streaming CTC models in the browser, live
from the microphone or with an audio file. The page sends 16 bit audio
over a WebSocket and gets the transcript back as it grows.
"""

import argparse
import asyncio
import glob
import json
import os
import time

import numpy as np
import torch
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

import speech
from speech import models, streaming

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript"),
    "/recorder.js": ("recorder.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
}

# Seconds of silence that end a line.
PAUSE = 1.0


class Models:
    def __init__(self, root, device):
        """
        The models and LMs found under root, loaded when first used: model
        directories hold a best_model, and LMs are .npz files.
        """
        self.device = device
        self.models = sorted(
            os.path.dirname(p)
            for p in glob.glob(os.path.join(root, "**", "best_model"), recursive=True)
        )
        self.lms = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        self.loaded = {}
        self.searches = {}

    def load(self, model_path, lm_path=None):
        """
        Returns the model, its preprocessor and a function that creates a
        decoder for one stream. Only the models and LMs found can be
        loaded.
        """
        if model_path not in self.models:
            raise ValueError(f"Unknown model {model_path}")
        if lm_path is not None and lm_path not in self.lms:
            raise ValueError(f"Unknown language model {lm_path}")
        if model_path not in self.loaded:
            model, preproc = speech.load(model_path, tag="best")
            if not isinstance(model, models.CTC):
                raise ValueError(f"{model_path} is not a CTC model, which can stream")
            if model.rnn.bidirectional:
                raise ValueError(f"{model_path} is bidirectional, so it cannot stream")
            model.to(self.device)
            model.set_eval()
            self.loaded[model_path] = (model, preproc)
        model, preproc = self.loaded[model_path]
        key = (model_path, lm_path)
        if key not in self.searches:
            self.searches[key] = streaming.search_factory(
                model.blank, preproc.char_to_int, lm_path
            )
        return model, preproc, self.searches[key]


def respond(content_type, body, status=200, reason="OK"):
    headers = Headers(
        [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
        ]
    )
    return Response(status, reason, headers, body)


def process_request(available, connection, request):
    """
    Answers plain HTTP requests with the page and the list of models, and
    lets the WebSocket requests for /stream through.
    """
    path = request.path.split("?")[0]
    if path == "/stream":
        return None
    if path == "/models":
        body = json.dumps({"models": available.models, "lms": available.lms})
        return respond("application/json", body.encode())
    if path in STATIC:
        name, content_type = STATIC[path]
        with open(os.path.join(WEB_DIR, name), "rb") as fid:
            return respond(content_type, fid.read())
    return respond("text/plain", b"Not found", 404, "Not Found")


async def stream(available, connection):
    """
    Transcribes one stream. The page sends a start message with the model
    and LM, then 16 bit little-endian audio at the model's sample rate,
    then a stop message. It gets back the partial text after every
    chunk, and a line whenever a pause ends one.
    """

    async def send(**message):
        await connection.send(json.dumps(message))

    transcriber = None
    sample_rate = None
    # Seconds spent decoding and seconds of audio decoded, for the speed.
    compute = heard = 0.0
    async for message in connection:
        if isinstance(message, bytes):
            if transcriber is None:
                await send(type="error", message="Send start before audio.")
                continue
            audio = np.frombuffer(message, dtype="<i2")
            begin = time.time()
            transcriber.push(audio)
            compute += time.time() - begin
            heard += len(audio) / sample_rate
            line = transcriber.take_line(PAUSE)
            if line is not None:
                await send(type="line", text=line)
            rtf = compute / heard if heard else None
            await send(type="partial", text=transcriber.text().strip(), rtf=rtf)
            continue

        request = json.loads(message)
        if request["type"] == "start":
            try:
                model, preproc, new_search = await asyncio.to_thread(
                    available.load, request["model"], request.get("lm")
                )
            except (ValueError, OSError) as error:
                await send(type="error", message=str(error))
                continue
            sample_rate = streaming.model_sample_rate(preproc)
            transcriber = streaming.Transcriber(
                model, preproc, sample_rate, new_search()
            )
            compute = heard = 0.0
            await send(type="ready", sample_rate=sample_rate)
        elif request["type"] == "stop" and transcriber is not None:
            # Flush the frames held back for the lookahead and finish the
            # line.
            transcriber.push(np.zeros(0, dtype=np.int16), final=True)
            line = transcriber.text(final=True).strip()
            if line:
                await send(type="line", text=line)
            await send(type="partial", text="")
            await send(type="done")
            transcriber = None


def start_server(root, host, port, device):
    """Returns the server, to use as an async context manager."""
    available = Models(root, device)
    return serve(
        lambda connection: stream(available, connection),
        host,
        port,
        process_request=lambda connection, request: process_request(
            available, connection, request
        ),
    )


async def main(args):
    async with start_server(args.models, args.host, args.port, args.device) as server:
        print(f"Open http://{args.host}:{args.port} in a browser. Ctrl+C stops.")
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve a page to try the streaming CTC models in the browser."
    )
    parser.add_argument(
        "--models",
        default="examples",
        help="Where to look for model directories and LM .npz files.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        type=torch.device,
        default="cpu",
        help="The device to run the models on. Streaming does little work per "
        "chunk, so the CPU is usually fastest.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
