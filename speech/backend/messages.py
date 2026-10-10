"""
The messages of the speech recognition service, see docs/backend-api.md.

All of them are JSON with camelCase field names.
"""

import datetime
import enum
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
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


Card = Annotated[TranscriptCard | NoticeCard | WeatherCard, Field(discriminator="type")]


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
