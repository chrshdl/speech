"""
Runs the speech recognition service, see docs/backend-api.md.
"""

import argparse
import asyncio
import logging
import os
import threading

import numpy as np
import uvicorn

from speech import streaming, tokens, vad, wake_word
from speech.backend import (
    brightness,
    game,
    keyboard,
    songs,
    voice,
    volume,
    weather,
    wifi,
)
from speech.backend.app import create_app
from speech.backend.engine import Microphone, Recognizer, Settings, WakeWord
from speech.backend.messages import AudioInputKind
from speech.intents import THRESHOLD, IntentRecognizer
from speech.models.word_lm import LM_WEIGHT, UNK_PENALTY, WORD_BONUS

# What the service answers.
INTENTS = [weather.INTENT, game.INTENT, keyboard.INTENT]


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings(
        end_of_speech=args.end_of_speech,
        no_speech_timeout=args.no_speech_timeout,
        max_capture=args.max_capture,
        audio_timeout=args.audio_timeout,
        card_timeout=args.card_timeout,
        answer_card_timeout=args.answer_card_timeout,
        wake_word_lookback=args.wake_word_lookback,
    )
    # Read before the model loads, so a bad song file fails at once.
    jukebox = songs.Songs(songs.load(args.songs) if args.songs else [])
    screen = None
    if args.backlight:
        state = os.path.join(args.state_dir, "brightness") if args.state_dir else None
        screen = brightness.Brightness(args.backlight, state)
    speaker_volume = None
    if args.volume_control:
        state = os.path.join(args.state_dir, "volume") if args.state_dir else None
        speaker_volume = volume.Volume(volume.Mixer(args.volume_control), state)
    speaker = None
    if args.voice:
        speaker = voice.Voice(args.voice)
        # Alongside the model, rather than after it, as it takes as long. A
        # question asked before it is loaded waits for it to be spoken.
        threading.Thread(target=speaker.load, daemon=True).start()
    new_transcriber, sample_rate, intents, commands, wake, voice_activity = load(
        args, jukebox, screen, speaker_volume
    )
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
            keyboard.INTENT.name: keyboard.Keyboard(),
            songs.NAME: jukebox,
            brightness.INTENT.name: screen,
            **(speaker_volume.handlers if speaker_volume is not None else {}),
        },
        commands=commands,
        voice=speaker,
        wake_word=wake,
        vad=voice_activity,
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


def load(args, jukebox, screen, speaker_volume=None):
    """
    Loads the streaming CTC model and returns a function that creates the
    transcriber of a capture, the sample rate, and the recognizers of the
    intents, with asking for one of the songs, a songs.Songs, and with a
    screen, a brightness.Brightness, for its brightness, and with a
    speaker_volume, a volume.Volume, for turning it up and down, and of the game's
    commands, which match the model's output, and the engine.WakeWord, if
    there is a wake word model, and the voice activity detector, if there
    is one. The models are all loaded now, so a capture starts at once when
    the wake word is heard.
    """
    model = streaming.load(args.model, "best", "cpu", args.threads)
    new_search = streaming.search_factory(
        model.blank,
        model.char_to_int,
        args.lm,
        lm_weight=args.lm_weight,
        word_bonus=args.word_bonus,
        unk_penalty=args.unk_penalty,
    )

    intents = [*INTENTS]
    if jukebox.intent is not None:
        intents.append(jukebox.intent)
    if screen is not None:
        intents.append(brightness.INTENT)
    if speaker_volume is not None:
        intents.extend(volume.INTENTS)
    intents = IntentRecognizer(
        intents,
        model.char_to_int,
        model.blank,
        model.frame_seconds,
        args.intent_threshold,
    )
    voice_activity = vad.VadModel(args.vad) if args.vad else None
    if voice_activity is not None and voice_activity.sample_rate != model.sample_rate:
        raise ValueError("The voice activity detector needs audio at another rate.")
    wake = None
    if args.wake_word:
        spotter = wake_word.WakeWordModel(args.wake_word)
        if spotter.sample_rate != model.sample_rate:
            raise ValueError("The wake word model needs audio at another sample rate.")
        labels = tokens.tokenize(spotter.keyword, model.char_to_int)
        detector = wake_word.Detector(spotter, args.wake_word_threshold)
        if voice_activity is not None:
            detector = wake_word.GatedDetector(
                spotter, voice_activity, args.wake_word_threshold
            )
        wake = WakeWord(
            detector,
            np.array([model.char_to_int[t] for t in labels]),
            model.blank,
            model.char_to_int.get(" "),
            args.wake_word_asr_threshold,
        )
    return (
        lambda: streaming.Transcriber(model, new_search()),
        model.sample_rate,
        intents,
        game.commands(model, args.command_threshold),
        wake,
        voice_activity,
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


def make_parser():
    """The service's command line."""
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
        "--lm-weight",
        type=float,
        default=LM_WEIGHT,
        help="Scales the LM scores. tune_lm.py finds the best weight, bonus and "
        "penalty for a model and an LM.",
    )
    parser.add_argument(
        "--word-bonus",
        type=float,
        default=WORD_BONUS,
        help="Score added per word, which offsets the LM's preference for fewer words.",
    )
    parser.add_argument(
        "--unk-penalty",
        type=float,
        default=UNK_PENALTY,
        help="Score added per word outside the LM's vocabulary.",
    )
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
        "--wake-word",
        help="Start a capture when the microphone hears the wake word, with this "
        "model of examples/olivia/train.py, a .onnx file. Needs the microphone's "
        "audio.",
    )
    parser.add_argument(
        "--wake-word-threshold",
        type=float,
        default=wake_word.THRESHOLD,
        help="The lowest probability of the wake word, averaged over the last "
        "decisions, that hears it, see examples/olivia/README.md. Higher hears it "
        "less often in other speech, and misses it more often.",
    )
    parser.add_argument(
        "--vad",
        help="A voice activity detector, Silero VAD's silero_vad.onnx, version 5. "
        "The wake word model then runs only while it hears speech, and the "
        "captures that end by themselves end when it hears speech end, see "
        "examples/olivia/README.md.",
    )
    parser.add_argument(
        "--wake-word-asr-threshold",
        type=float,
        default=wake_word.ASR_THRESHOLD,
        help="The lowest score of the wake word in the speech model's transcript "
        "that confirms it was heard, see examples/olivia/README.md. Higher drops "
        "more of what other speech, music or a TV set it off by, and more wake "
        "words.",
    )
    parser.add_argument(
        "--weather-location",
        default="Berlin",
        help="The place to answer questions for the weather for.",
    )
    parser.add_argument(
        "--voice",
        help="Speak answers, such as the weather, with this Piper voice, an .onnx "
        "file with its .onnx.json beside it, see `python -m piper.download_voices`. "
        "The client plays the speech.",
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
        "--volume-control",
        help='Let the user turn the speaker up and down, as in "volume up" and '
        '"turn it down", with this ALSA mixer control, as "card:control", such '
        'as "usbaudio:PCM", see `amixer -c usbaudio scontrols`.',
    )
    parser.add_argument(
        "--state-dir",
        help="A directory the service keeps settings in, such as the brightness "
        "and the volume, to set them again when it starts.",
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
        "--wake-word-lookback",
        type=float,
        default=defaults.wake_word_lookback,
        help="The audio before the wake word was heard that its capture starts "
        "with, which holds the wake word and what was said right after it.",
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
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    if args.mic_device is not None and args.mic_device.isdigit():
        args.mic_device = int(args.mic_device)
    main(args)
