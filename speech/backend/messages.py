"""
The messages of the speech recognition service, see docs/backend-api.md.

All of them are JSON with camelCase field names.
"""

import datetime
import enum
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, validate_by_name=True)


class Initiator(enum.StrEnum):
    """How a capture was started, which decides how it ends."""

    # The user holds a button while speaking. The capture ends with
    # stop-capture, when the button is released.
    HOLD_TO_TALK = "HOLD_TO_TALK"
    # The user tapped a button. The capture ends at the end of speech.
    TAP_TO_TALK = "TAP_TO_TALK"
    # Listens for short commands, such as "left" and "right" to steer a
    # game, each reported by a CommandRecognized event as soon as it is
    # said, until stop-capture.
    COMMANDS = "COMMANDS"


class DialogState(enum.StrEnum):
    # Ready to start a capture.
    IDLE = "IDLE"
    # Capturing and recognizing speech.
    LISTENING = "LISTENING"
    # The capture has ended and the final transcript is being decoded.
    THINKING = "THINKING"


class AudioInputKind(enum.StrEnum):
    # The microphone of the device the service runs on.
    MICROPHONE = "MICROPHONE"
    # The client streams audio for each capture, see StartAudioStream.
    CLIENT = "CLIENT"


class CaptureEndReason(enum.StrEnum):
    # A TAP_TO_TALK capture heard speech followed by a pause.
    END_OF_SPEECH = "END_OF_SPEECH"
    # The client called stop-capture, or closed its audio stream.
    STOP_CAPTURE = "STOP_CAPTURE"
    # A TAP_TO_TALK capture heard no speech in the time allowed.
    NO_SPEECH = "NO_SPEECH"
    # The capture reached the longest duration allowed.
    MAX_DURATION = "MAX_DURATION"
    # No audio arrived in the time allowed, such as when the client did
    # not stream any.
    AUDIO_TIMEOUT = "AUDIO_TIMEOUT"
    # The service failed, see its log.
    ERROR = "ERROR"


# Display cards.


class TranscriptCard(CamelModel):
    """
    The transcript of a capture, which grows as the user speaks. It keeps
    its id, the capture's dialogRequestId, as it is updated.
    """

    type: Literal["TranscriptCard"] = "TranscriptCard"
    id: str
    title: str
    text: str
    # False while the transcript can still change.
    is_final: bool


class NoticeCard(CamelModel):
    """A titled text, such as a notice that no speech was heard."""

    type: Literal["NoticeCard"] = "NoticeCard"
    id: str
    title: str
    text: str


class WeatherCondition(enum.StrEnum):
    CLEAR = "CLEAR"
    PARTLY_CLOUDY = "PARTLY_CLOUDY"
    CLOUDY = "CLOUDY"
    FOG = "FOG"
    DRIZZLE = "DRIZZLE"
    RAIN = "RAIN"
    SNOW = "SNOW"
    THUNDERSTORM = "THUNDERSTORM"


class WeatherDay(CamelModel):
    date: datetime.date
    condition: WeatherCondition
    high: float
    low: float


class WeatherCard(CamelModel):
    """The current weather and the forecast for the next days."""

    type: Literal["WeatherCard"] = "WeatherCard"
    id: str
    location: str
    condition: WeatherCondition
    # The condition in words, such as "Partly cloudy".
    description: str
    # False at night, for a moon rather than a sun.
    is_day: bool
    temperature: float
    feels_like: float
    # Relative humidity in percent.
    humidity: int
    wind_speed: float
    # Today first.
    days: list[WeatherDay]
    temperature_unit: str = "°C"
    wind_speed_unit: str = "km/h"


class GameCard(CamelModel):
    """
    Play a game, Superball, at the level: the client lets flowers fall for
    the unicorn to catch, which the user steers with the commands of a
    COMMANDS capture. The card stays until the client dismisses it when
    the game ends, or another card replaces it.
    """

    UNTIL_DISMISSED: ClassVar[bool] = True
    type: Literal["GameCard"] = "GameCard"
    id: str
    title: str
    # From 1, the easiest, to 5, where the flowers fall fastest.
    level: int


class SongCard(CamelModel):
    """
    Play a song: the client plays the notes at the url, and dismisses the
    card when it has played, or with loop, plays it over and over until the
    user closes the card. The card stays until then, or until another card
    replaces it.
    """

    UNTIL_DISMISSED: ClassVar[bool] = True
    type: Literal["SongCard"] = "SongCard"
    id: str
    title: str
    # The song's notes, a SongScore, see GET /v1/songs/{song_id}.
    url: str
    # How long the song plays, once round for a loop.
    seconds: float
    loop: bool = False


class BrightnessCard(CamelModel):
    """The brightness of the screen, just set, in percent."""

    type: Literal["BrightnessCard"] = "BrightnessCard"
    id: str
    title: str
    percent: int


class SongNote(CamelModel):
    """
    A note of a song, played with the voice of an additive synthesizer, or
    of a Roland D-50 with a patch.
    """

    # In seconds from the song's start.
    start: float
    seconds: float
    # In Hz.
    frequency: float
    # The amplitudes of its harmonics, the fundamental first.
    partials: list[float]
    volume: float
    # How many seconds it takes to swell and to fade, the fading also on
    # past its end.
    attack: float
    release: float
    # The name of the D-50 patch that plays it instead, of the score's
    # patches, with its own envelopes rather than partials, attack and release.
    patch: str | None = None


class SongScore(CamelModel):
    """The notes of a song, which a SongCard plays."""

    # How long it plays, once round for a loop.
    seconds: float
    # Whether it plays over and over until the card is closed.
    loop: bool
    notes: list[SongNote]
    # The D-50 patches of the notes, by name: their parameters, as
    # speech/backend/d50.py loads them.
    patches: dict[str, dict] = {}


Card = Annotated[
    TranscriptCard | NoticeCard | WeatherCard | GameCard | SongCard | BrightnessCard,
    Field(discriminator="type"),
]


# The payloads of the events the service publishes. The class name is the
# message's action.


class Payload(CamelModel):
    TOPIC: ClassVar[str]


class DialogStateChanged(Payload):
    TOPIC = "Recognition"
    state: DialogState
    # The capture the state belongs to, None when IDLE.
    dialog_request_id: str | None


class TranscriptUpdated(Payload):
    """The transcript so far, published whenever it changes."""

    TOPIC = "Recognition"
    dialog_request_id: str
    text: str


class IntentRecognized(Payload):
    """
    The capture asked for something the service can answer, such as the
    weather, published after CaptureFinished. The card with the answer
    follows.
    """

    TOPIC = "Recognition"
    dialog_request_id: str
    # Such as "GetWeather".
    intent: str
    # How well the utterance matched the intent's closest sample
    # utterance, a log probability per character, at most 0.
    score: float
    # The values of the sample's slots, such as {"city": "hamburg"}.
    slots: dict[str, str]


class CommandRecognized(Payload):
    """
    A COMMANDS capture heard a command, published as soon as it is
    recognized, without waiting for the end of the capture.
    """

    TOPIC = "Recognition"
    dialog_request_id: str
    # Such as "LEFT" or "RIGHT".
    command: str
    # How well the utterance matched the command, as IntentRecognized's.
    score: float
    # The transcript of the utterance so far.
    text: str


class EndOfSpeechDetected(Payload):
    TOPIC = "Recognition"
    dialog_request_id: str


class CaptureFinished(Payload):
    """The final transcript, which every capture ends with."""

    TOPIC = "Recognition"
    dialog_request_id: str
    # Empty when no speech was recognized.
    text: str
    reason: CaptureEndReason


class ShowCard(Payload):
    """
    Show the card, replacing the card shown. A card with the id of the
    card shown updates it.
    """

    TOPIC = "Display"
    card: Card


class ClearCard(Payload):
    """Dismiss the card with the id."""

    TOPIC = "Display"
    card_id: str


class StartAudioStream(Payload):
    """
    Stream the capture's audio to the WebSocket at url, as binary
    messages of 16 bit little-endian mono samples at sample_rate.
    """

    TOPIC = "Audio"
    stream_id: str
    url: str
    sample_rate: int
    encoding: Literal["LINEAR16"] = "LINEAR16"
    channels: Literal[1] = 1


class StopAudioStream(Payload):
    """
    Stop streaming. The service also closes the WebSocket. The client can
    end the capture itself by closing the WebSocket after the last audio,
    which unlike stop-capture never loses audio still on its way.
    """

    TOPIC = "Audio"
    stream_id: str


# Every event's payload.
EVENTS = (
    DialogStateChanged,
    TranscriptUpdated,
    EndOfSpeechDetected,
    CaptureFinished,
    IntentRecognized,
    CommandRecognized,
    ShowCard,
    ClearCard,
    StartAudioStream,
    StopAudioStream,
)


class Message(CamelModel):
    """The envelope of each event."""

    # Increases by one with each message the service sends.
    id: int
    topic: str
    action: str
    timestamp: datetime.datetime
    payload: SerializeAsAny[Payload]


# Requests and responses of the REST endpoints.


class StartCaptureRequest(CamelModel):
    initiator: Initiator


class StartCaptureResponse(CamelModel):
    dialog_request_id: str
    # Where to stream the audio, with audio input from the client.
    audio_stream: StartAudioStream | None


class StopCaptureRequest(CamelModel):
    # Stops only this capture, so a late request cannot stop a newer one.
    dialog_request_id: str | None = None


class StopCaptureResponse(CamelModel):
    dialog_request_id: str


class CardDismissedRequest(CamelModel):
    card_id: str


class State(CamelModel):
    dialog_state: DialogState
    dialog_request_id: str | None
    # The card shown, if any.
    card: Card | None


class Info(CamelModel):
    model: str
    sample_rate: int
    audio_input: AudioInputKind
    language_model: str | None
    # Whether the client can join a Wi-Fi network through /v1/wifi.
    wifi: bool = False


# Wi-Fi, see speech/backend/wifi.py.


class WifiState(enum.StrEnum):
    # Joined a network, with an IPv4 address.
    CONNECTED = "CONNECTED"
    # Joining a network, or joined and waiting for an address.
    CONNECTING = "CONNECTING"
    DISCONNECTED = "DISCONNECTED"
    # wpa_supplicant does not answer, such as while it restarts.
    UNAVAILABLE = "UNAVAILABLE"


class WifiStatus(CamelModel):
    state: WifiState
    # The network joined or being joined.
    ssid: str | None = None
    ip_address: str | None = None
    # Whether a network is saved, which the device joins when in reach.
    configured: bool = False


class WifiNetwork(CamelModel):
    ssid: str
    # Whether it needs a password.
    secured: bool
    # The signal strength in bars, from 0 to 4.
    signal: int = Field(ge=0, le=4)


class WifiNetworks(CamelModel):
    # The networks in reach, the strongest first, one per name.
    networks: list[WifiNetwork]


class WifiConnectRequest(CamelModel):
    ssid: str = Field(min_length=1)
    # Empty for an open network.
    password: str = ""

    @field_validator("ssid")
    @classmethod
    def ssid_fits(cls, ssid):
        if len(ssid.encode()) > 32:
            raise ValueError("A network name has at most 32 bytes.")
        return ssid

    @field_validator("password")
    @classmethod
    def password_is_wpa(cls, password):
        # WPA passphrases are 8 to 63 printable ASCII characters.
        if password and not (
            8 <= len(password) <= 63 and all(" " <= c <= "~" for c in password)
        ):
            raise ValueError("A Wi-Fi password has 8 to 63 ASCII characters.")
        return password


class WifiConnectError(enum.StrEnum):
    # The network refused the password.
    WRONG_PASSWORD = "WRONG_PASSWORD"
    # The network was not joined in time, such as when it is out of reach.
    NOT_JOINED = "NOT_JOINED"
    # Joined, and the network is saved, but no address came in time.
    NO_ADDRESS = "NO_ADDRESS"


class WifiConnectResponse(CamelModel):
    status: WifiStatus
    # Why it did not connect, null when it did.
    error: WifiConnectError | None = None
