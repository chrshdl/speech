"""
Runs the speech recognition service, see docs/backend-api.md.
"""

import argparse
import asyncio
import logging
import os

import uvicorn

from speech import streaming
from speech.backend import weather
from speech.backend.app import create_app
from speech.backend.engine import Microphone, Recognizer, Settings
from speech.backend.messages import AudioInputKind
from speech.intents import IntentRecognizer


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model = streaming.load(args.model, "best", "cpu", args.threads)
    new_search = streaming.search_factory(model.blank, model.char_to_int, args.lm)
    settings = Settings(
        end_of_speech=args.end_of_speech,
        no_speech_timeout=args.no_speech_timeout,
        max_capture=args.max_capture,
        audio_timeout=args.audio_timeout,
        card_timeout=args.card_timeout,
        answer_card_timeout=args.answer_card_timeout,
    )
    intents = IntentRecognizer(
        [weather.INTENT], model.char_to_int, model.blank, model.frame_seconds
    )
    handlers = {weather.INTENT.name: weather.Weather(args.weather_location)}
    recognizer = Recognizer(
        model,
        new_search,
        AudioInputKind[args.audio_input.upper()],
        settings,
        model_name=os.path.basename(os.path.normpath(args.model)),
        language_model=os.path.basename(args.lm) if args.lm else None,
        intents=intents,
        handlers=handlers,
    )
    microphone = None
    if recognizer.audio_input == AudioInputKind.MICROPHONE:
        microphone = Microphone(recognizer, args.mic_device)
    config = uvicorn.Config(
        create_app(recognizer, microphone),
        host=args.host,
        port=args.port,
        # A last resort, should a connection still not close.
        timeout_graceful_shutdown=5,
    )
    Server(config, recognizer).run()


class Server(uvicorn.Server):
    def __init__(self, config, recognizer):
        """
        Serves the app, and on Ctrl+C or SIGTERM first ends the event
        streams and the capture, which would otherwise keep their
        connections open and the server from stopping.
        """
        super().__init__(config)
        self.recognizer = recognizer

    def handle_exit(self, sig, frame):
        # Called from a signal handler, so run on the event loop instead.
        self.loop.call_soon_threadsafe(self.recognizer.shutdown)
        super().handle_exit(sig, frame)

    async def serve(self, sockets=None):
        self.loop = asyncio.get_running_loop()
        await super().serve(sockets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve streaming speech recognition with an HTTP API. The API "
        "is documented at /docs and in docs/backend-api.md."
    )
    parser.add_argument(
        "model",
        help="A model exported to .onnx by export.py, or a PyTorch model directory.",
    )
    parser.add_argument("--lm", help="A word LM .npz file from speech.models.word_lm.")
    parser.add_argument(
        "--audio-input",
        choices=["microphone", "client"],
        default="microphone",
        help="Capture from this device's microphone, or from audio the client "
        "streams for each capture.",
    )
    parser.add_argument(
        "--mic-device",
        help="The microphone's name or index, by default the system default. "
        "See `python -m sounddevice` for the list.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="The address to listen on, 0.0.0.0 for every network interface.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--threads", type=int, help="The CPU threads to run the model with."
    )
    parser.add_argument(
        "--weather-location",
        default="Berlin",
        help="The place to answer questions for the weather for.",
    )
    limits = parser.add_argument_group("time limits in seconds")
    defaults = Settings()
    limits.add_argument(
        "--end-of-speech",
        type=float,
        default=defaults.end_of_speech,
        help="A pause after speech this long ends a TAP_TO_TALK capture.",
    )
    limits.add_argument(
        "--no-speech-timeout",
        type=float,
        default=defaults.no_speech_timeout,
        help="A TAP_TO_TALK capture that hears no speech this long ends.",
    )
    limits.add_argument(
        "--max-capture",
        type=float,
        default=defaults.max_capture,
        help="The longest capture.",
    )
    limits.add_argument(
        "--audio-timeout",
        type=float,
        default=defaults.audio_timeout,
        help="A capture ends when no audio arrives this long.",
    )
    limits.add_argument(
        "--card-timeout",
        type=float,
        default=defaults.card_timeout,
        help="A card with the final transcript is cleared after this long.",
    )
    limits.add_argument(
        "--answer-card-timeout",
        type=float,
        default=defaults.answer_card_timeout,
        help="A card that answers a question, such as for the weather, is cleared "
        "after this long.",
    )
    args = parser.parse_args()
    if args.mic_device is not None and args.mic_device.isdigit():
        args.mic_device = int(args.mic_device)
    main(args)
