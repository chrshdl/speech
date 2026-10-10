"""
The HTTP API of the speech recognition service, see docs/backend-api.md.
"""

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException, Response, WebSocket, status
from fastapi.responses import RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic.json_schema import models_json_schema

from speech.backend.engine import Busy
from speech.backend.messages import (
    EVENTS,
    CardDismissedRequest,
    Info,
    Message,
    StartCaptureRequest,
    StartCaptureResponse,
    State,
    StopCaptureRequest,
    StopCaptureResponse,
)

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


def create_app(recognizer, microphone=None):
    """
    Returns the service for the recognizer, an engine.Recognizer, which
    with a microphone, an engine.Microphone, listens to it while running.

    The endpoints are all async, so they run on the event loop, as the
    recognizer must.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app):
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
        responses={409: {"description": "Not IDLE"}},
    )
    async def start_capture(request: StartCaptureRequest) -> StartCaptureResponse:
        """
        Starts capturing and recognizing speech. A HOLD_TO_TALK capture
        ends with stop-capture, a TAP_TO_TALK one also at the end of
        speech. With audio input from the client, stream the audio as
        audioStream says.
        """
        try:
            capture = recognizer.start_capture(request.initiator)
        except Busy as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None
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
