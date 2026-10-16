"""
Runs the speech recognition service, see docs/backend-api.md.
"""

import argparse
import asyncio
import logging
import os

import uvicorn

from speech import streaming
from speech.backend import brightness, game, songs, weather, wifi
from speech.backend.app import create_app
from speech.backend.engine import Microphone, Recognizer, Settings
from speech.backend.messages import AudioInputKind
from speech.intents import THRESHOLD, IntentRecognizer

# What the service answers.
INTENTS = [weather.INTENT, game.INTENT]


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings(
        end_of_speech=args.end_of_speech,
        no_speech_timeout=args.no_speech_timeout,
        max_capture=args.max_capture,
        audio_timeout=args.audio_timeout,
        card_timeout=args.card_timeout,
        answer_card_timeout=args.answer_card_timeout,
    )
    # Read before the model loads, so a bad song file fails at once.
    jukebox = songs.Songs(songs.load(args.songs) if args.songs else [])
    screen = None
    if args.backlight:
        state = os.path.join(args.state_dir, "brightness") if args.state_dir else None
        screen = brightness.Brightness(brightness.find(args.backlight), state)
    new_transcriber, sample_rate, intents, commands = load(args, jukebox, screen)
    recognizer = Recognizer(
        new_transcriber,
        sample_rate,
        AudioInputKind[args.audio_input.upper()],
        settings,
        model_name=os.path.basename(os.path.normpath(args.model)),
        language_model=os.path.basename(args.lm) if args.lm else None,
        intents=intents,
        handlers={
            weather.INTENT.name: weather.Weather(args.weather_location),
            game.INTENT.name: game.Game(),
            songs.NAME: jukebox,
            brightness.INTENT.name: screen,
        },
        commands=commands,
    )
    microphone = None
    if recognizer.audio_input == AudioInputKind.MICROPHONE:
        microphone = Microphone(recognizer, args.mic_device)
    wifi_control = None
    if args.wifi_interface:
        wifi_control = wifi.Wifi(args.wifi_interface, args.wifi_control_dir)
    config = uvicorn.Config(
        create_app(recognizer, microphone, wifi_control, jukebox),
        host=args.host,
        port=args.port,
        # A last resort, should a connection still not close.
        timeout_graceful_shutdown=5,
    )
    Server(config, recognizer).run()


def load(args, jukebox, screen):
    """
    Loads the streaming CTC model and returns a function that creates the
    transcriber of a capture, the sample rate, and the recognizers of the
    intents, with asking for one of the songs, a songs.Songs, and with a
    screen, a brightness.Brightness, for its brightness, and of the game's
    commands, which match the model's output.
    """
    model = streaming.load(args.model, "best", "cpu", args.threads)
    new_search = streaming.search_factory(model.blank, model.char_to_int, args.lm)

    intents = [*INTENTS]
    if jukebox.intent is not None:
        intents.append(jukebox.intent)
    if screen is not None:
        intents.append(brightness.INTENT)
    intents = IntentRecognizer(
        intents,
        model.char_to_int,
        model.blank,
        model.frame_seconds,
        args.intent_threshold,
    )
    return (
        lambda: streaming.Transcriber(model, new_search()),
        model.sample_rate,
        intents,
        game.commands(model, args.command_threshold),
    )


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
        help="A streaming CTC model exported to .onnx by export.py, or its PyTorch "
        "directory.",
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
        "--intent-threshold",
        type=float,
        default=THRESHOLD,
        help="The lowest score of a matching intent, see speech/intents.py. "
        "Higher matches fewer other utterances and fewer of the intent's. The "
        "default suits the 3,750 hour model, -2 the 100 hour one, whose output "
        "is less confident.",
    )
    parser.add_argument(
        "--command-threshold",
        type=float,
        default=game.THRESHOLD,
        help='The lowest score of a game command, such as "left", see '
        "speech/backend/game.py. Higher moves less on other words, and on fewer "
        "commands.",
    )
    parser.add_argument(
        "--weather-location",
        default="Berlin",
        help="The place to answer questions for the weather for.",
    )
    parser.add_argument(
        "--songs",
        help='A directory of songs to play when asked, such as "play twinkle '
        'twinkle", .json files, see speech/backend/songs.py.',
    )
    parser.add_argument(
        "--backlight",
        help="Let the user set the screen's brightness, as in \"set the brightness "
        'to fifty percent", with this backlight in /sys/class/backlight, or "auto" '
        "for the only one.",
    )
    parser.add_argument(
        "--state-dir",
        help="A directory the service keeps settings in, such as the brightness, "
        "to set them again when it starts.",
    )
    parser.add_argument(
        "--wifi-interface",
        help="Let the client join a Wi-Fi network on this network interface, such "
        "as wlan0, through wpa_supplicant, see docs/backend-api.md. Anyone who "
        "reaches the service can then change its Wi-Fi.",
    )
    parser.add_argument(
        "--wifi-control-dir",
        default=wifi.CONTROL_DIR,
        help="The directory of wpa_supplicant's control sockets, its ctrl_interface.",
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
