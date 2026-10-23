"""
The HTTP API of the speech recognition service, see docs/backend-api.md.
"""

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException, Response, WebSocket, status
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic.json_schema import models_json_schema

from speech.backend import d50
from speech.backend.engine import Busy, Unsupported
from speech.backend.messages import (
    EVENTS,
    Card,
    CardDismissedRequest,
    Info,
    Message,
    SongScore,
    StartCaptureRequest,
    StartCaptureResponse,
    State,
    StopCaptureRequest,
    StopCaptureResponse,
    WifiConnectRequest,
    WifiConnectResponse,
    WifiNetworks,
    WifiStatus,
)
from speech.backend.wifi import Unavailable

# The browser client in web/assistant, which shares web/recorder.js with
# serve.py's page.
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "web")


class RevalidatedFiles(StaticFiles):
    """
    Static files that browsers check for changes on every load, so the
    client's files never mix versions. Without a Cache-Control header
    Firefox reused an old copy of a script after the page changed.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(recognizer, microphone=None, wifi=None, songs=None):
    """
    Returns the service for the recognizer, an engine.Recognizer, which
    with a microphone, an engine.Microphone, listens to it while running.
    With wifi, a wifi.Wifi, the client can join a Wi-Fi network. With
    songs, a songs.Songs, it serves the notes its SongCards play.

    The endpoints are all async, so they run on the event loop, as the
    recognizer must.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app):
        recognizer.start()
        if microphone is not None:
            microphone.start(asyncio.get_running_loop())
        try:
            yield
        finally:
            if microphone is not None:
                microphone.stop()
            await recognizer.close()

    app = FastAPI(
        title="Speech recognizer",
        description="Streaming speech recognition with display cards. Events are "
        "sent on GET /v1/events, see docs/backend-api.md.",
        version="1",
        lifespan=lifespan,
    )

    @app.get("/v1/info")
    async def info() -> Info:
        """The model and how the service takes audio."""
        return Info(
            model=recognizer.model_name,
            sample_rate=recognizer.sample_rate,
            audio_input=recognizer.audio_input,
            language_model=recognizer.language_model,
            wifi=wifi is not None,
        )

    @app.get("/v1/state")
    async def state() -> State:
        """The dialog state and the card shown."""
        capture = recognizer.capture
        return State(
            dialog_state=recognizer.state,
            dialog_request_id=capture.id if capture is not None else None,
            card=recognizer.card,
        )

    @app.get("/v1/events", response_class=EventSourceResponse)
    async def events() -> AsyncIterator[ServerSentEvent]:
        """
        Server-sent events, one per message, named topic.action, with a
        Message as data. The payload schema is named after the action. The
        stream starts with the dialog state and the card shown, if any.
        """
        with recognizer.bus.subscribe(recognizer.snapshot()) as queue:
            while (message := await queue.get()) is not None:
                yield ServerSentEvent(
                    event=f"{message.topic}.{message.action}",
                    raw_data=message.model_dump_json(by_alias=True),
                )

    @app.post(
        "/v1/recognition/start-capture",
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            409: {"description": "Not IDLE"},
            422: {
                "description": "COMMANDS, and the service recognizes none, or "
                "WAKE_WORD, which only the wake word starts"
            },
        },
    )
    async def start_capture(request: StartCaptureRequest) -> StartCaptureResponse:
        """
        Starts capturing and recognizing speech. A HOLD_TO_TALK capture
        ends with stop-capture, a TAP_TO_TALK one also at the end of
        speech. With audio input from the client, stream the audio as
        audioStream says. A COMMANDS capture listens until stop-capture
        and reports each command it hears in a CommandRecognized event.
        """
        try:
            capture = recognizer.start_capture(request.initiator)
        except Busy as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None
        except Unsupported as error:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from None
        audio_stream = None
        if capture.stream_id is not None:
            audio_stream = recognizer.audio_stream_event(capture)
        return StartCaptureResponse(
            dialog_request_id=capture.id, audio_stream=audio_stream
        )

    @app.post(
        "/v1/recognition/stop-capture",
        status_code=status.HTTP_202_ACCEPTED,
        responses={409: {"description": "No capture to stop"}},
    )
    async def stop_capture(
        request: StopCaptureRequest | None = None,
    ) -> StopCaptureResponse:
        """
        Ends the capture after the audio received so far. The final
        transcript follows in a CaptureFinished event.
        """
        dialog_request_id = request.dialog_request_id if request is not None else None
        try:
            capture = recognizer.stop_capture(dialog_request_id)
        except Busy as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None
        return StopCaptureResponse(dialog_request_id=capture.id)

    @app.post(
        "/v1/intents/{intent}",
        responses={
            404: {"description": "No such intent"},
            409: {"description": "Not IDLE"},
        },
    )
    async def answer_intent(intent: str) -> Card:
        """
        Answers the intent as if it was heard, without slots, such as for a
        button that opens the synthesizer's keyboard: shows its card, in a
        ShowCard event as for speech, and returns it.
        """
        try:
            return await recognizer.answer(intent)
        except KeyError:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"No intent {intent}."
            ) from None
        except Busy as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None

    @app.post(
        "/v1/display/card-dismissed",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={409: {"description": "The card is not shown"}},
    )
    async def card_dismissed(request: CardDismissedRequest) -> Response:
        """
        The user dismissed the card, which the service then neither
        updates nor clears.
        """
        try:
            recognizer.card_dismissed(request.card_id)
        except Busy as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.websocket("/v1/audio/{stream_id}")
    async def audio_stream(websocket: WebSocket, stream_id: str):
        """
        Takes a capture's audio from the client, as binary messages of 16
        bit little-endian mono samples. The service closes the socket when
        the capture ends. Closing it from the client ends the capture after
        the audio sent. A stream that no capture takes is refused with
        HTTP 403.
        """
        capture = recognizer.open_stream(stream_id)
        if capture is None:
            await websocket.close()
            return
        await websocket.accept()
        ended = asyncio.create_task(capture.ended.wait())
        try:
            while True:
                receive = asyncio.create_task(websocket.receive())
                done, _ = await asyncio.wait(
                    {receive, ended}, return_when=asyncio.FIRST_COMPLETED
                )
                if receive not in done:
                    receive.cancel()
                    await websocket.close()
                    return
                message = receive.result()
                if message["type"] == "websocket.disconnect":
                    recognizer.end_stream(stream_id)
                    return
                data = message.get("bytes")
                if data is None or len(data) % 2:
                    await websocket.close(
                        status.WS_1003_UNSUPPORTED_DATA,
                        "Send binary messages of 16 bit samples.",
                    )
                    return
                recognizer.deliver(np.frombuffer(data, dtype="<i2"), stream_id)
        finally:
            ended.cancel()

    def require_wifi():
        if wifi is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "The service has no Wi-Fi.")

    @app.get("/v1/wifi", responses={404: {"description": "The service has no Wi-Fi"}})
    async def wifi_status() -> WifiStatus:
        """Whether a Wi-Fi network is joined, and which."""
        require_wifi()
        return await asyncio.to_thread(wifi.status)

    @app.post(
        "/v1/wifi/scan",
        responses={
            404: {"description": "The service has no Wi-Fi"},
            409: {"description": "A network is being joined"},
            503: {"description": "wpa_supplicant does not answer"},
        },
    )
    async def wifi_scan() -> WifiNetworks:
        """
        Scans for the Wi-Fi networks in reach, which takes up to some 10
        seconds.
        """
        require_wifi()
        if wifi.joining.locked():
            raise HTTPException(status.HTTP_409_CONFLICT, "A network is being joined.")
        try:
            networks = await asyncio.to_thread(wifi.scan)
        except Unavailable as error:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, str(error)
            ) from None
        return WifiNetworks(networks=networks)

    @app.post(
        "/v1/wifi/connect",
        responses={
            404: {"description": "The service has no Wi-Fi"},
            409: {"description": "Another network is being joined"},
            422: {"description": "Not a network name or WPA password"},
            503: {"description": "wpa_supplicant does not answer"},
        },
    )
    async def wifi_connect(request: WifiConnectRequest) -> WifiConnectResponse:
        """
        Joins the network, which takes up to a minute, and saves it in place
        of the others once it is joined. A network not joined is forgotten,
        and the networks saved before stay. The error says why it did not
        connect.
        """
        require_wifi()
        try:
            return await asyncio.to_thread(wifi.connect, request.ssid, request.password)
        except RuntimeError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None
        except Unavailable as error:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, str(error)
            ) from None

    @app.get(
        "/v1/speech/{speech_id}",
        response_class=StreamingResponse,
        responses={
            200: {"content": {"audio/L16": {}}},
            404: {"description": "Not the speech spoken last, or stopped"},
        },
    )
    async def speech(speech_id: str):
        """
        A Speak event's audio, 16 bit little-endian mono samples at its
        sampleRate, streamed as each sentence is synthesized. It ends after
        the last, or early when the speech is stopped.
        """
        spoken = recognizer.speech
        if spoken is None or spoken.id != speech_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such speech.")
        return StreamingResponse(
            spoken.audio(),
            media_type=f"audio/L16;rate={recognizer.voice.sample_rate};channels=1",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/v1/songs/{song_id}", responses={404: {"description": "No such song"}})
    async def song(song_id: str) -> SongScore:
        """A song's notes, which a SongCard plays."""
        score = songs.score(song_id) if songs is not None else None
        if score is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such song.")
        return score

    @app.get("/v1/patches/{patch}", responses={404: {"description": "No such patch"}})
    async def patch(patch: str) -> dict:
        """
        A Roland D-50 patch's parameters, of speech/backend/patches, which a
        KeyboardCard plays, see speech/backend/d50.py.
        """
        try:
            return d50.load(patch)
        except ValueError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such patch.") from None

    @app.get("/", include_in_schema=False)
    async def home():
        return RedirectResponse("/assistant/")

    app.mount("/", RevalidatedFiles(directory=WEB_DIR, html=True), name="web")

    def openapi():
        # The events are not responses, so add their schemas by hand.
        if app.openapi_schema is None:
            schema = FastAPI.openapi(app)
            _, definitions = models_json_schema(
                [(model, "serialization") for model in (Message, *EVENTS)],
                ref_template="#/components/schemas/{model}",
            )
            schema["components"]["schemas"].update(definitions["$defs"])
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi
    return app
