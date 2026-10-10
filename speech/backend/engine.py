"""
The speech recognition engine behind the service: captures speech from
the microphone or from the client, transcribes it as it arrives, and
publishes events and the display cards to show.
"""

import asyncio
import contextlib
import dataclasses
import datetime
import logging
import uuid

import numpy as np
import sounddevice

from speech import streaming
from speech.backend.messages import (
    AudioInputKind,
    CaptureEndReason,
    CaptureFinished,
    ClearCard,
    DialogState,
    DialogStateChanged,
    EndOfSpeechDetected,
    Initiator,
    Message,
    NoticeCard,
    ShowCard,
    StartAudioStream,
    StopAudioStream,
    TranscriptCard,
    TranscriptUpdated,
)

log = logging.getLogger(__name__)

# Ends a capture's audio, see Recognizer.stop_capture.
_STOP = object()


@dataclasses.dataclass
class Settings:
    """The time limits of a capture, in seconds."""

    # A pause after speech this long ends a TAP_TO_TALK capture.
    end_of_speech: float = 1.0
    # A TAP_TO_TALK capture that hears no speech for this long ends.
    no_speech_timeout: float = 8.0
    # The longest capture of any kind.
    max_capture: float = 60.0
    # Ends a capture when no audio arrives for this long.
    audio_timeout: float = 5.0
    # A final card is cleared after this long.
    card_timeout: float = 8.0


class Busy(Exception):
    """The request does not fit the dialog state."""


class EventBus:
    def __init__(self, max_queued=1000):
        """
        Sends each published message to every subscriber. A subscriber
        that falls max_queued messages behind is dropped.
        """
        self.max_queued = max_queued
        self.subscribers = set()
        self.next_id = 1

    def message(self, payload):
        message = Message(
            id=self.next_id,
            topic=payload.TOPIC,
            action=type(payload).__name__,
            timestamp=datetime.datetime.now(datetime.UTC),
            payload=payload,
        )
        self.next_id += 1
        return message

    def publish(self, payload):
        message = self.message(payload)
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                log.warning("Dropping an event subscriber that fell behind")
                self.subscribers.discard(queue)
                queue.get_nowait()
                queue.put_nowait(None)
        return message

    @contextlib.contextmanager
    def subscribe(self, initial=()):
        """
        Yields a queue of the messages published from now on, after the
        messages of the initial payloads. None in the queue ends it.
        """
        queue = asyncio.Queue(self.max_queued)
        for payload in initial:
            queue.put_nowait(self.message(payload))
        self.subscribers.add(queue)
        try:
            yield queue
        finally:
            self.subscribers.discard(queue)

    def close(self):
        """Ends every subscription."""
        for queue in self.subscribers:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)
        self.subscribers.clear()


@dataclasses.dataclass
class Capture:
    id: str
    initiator: Initiator
    transcriber: streaming.Transcriber
    audio: asyncio.Queue
    # With audio input from the client, the stream to read from and
    # whether a client is streaming it.
    stream_id: str | None = None
    streaming: bool = False
    # Set once the capture no longer takes audio.
    ended: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


class Recognizer:
    def __init__(
        self,
        model,
        new_search,
        audio_input,
        settings=None,
        model_name="",
        language_model=None,
    ):
        """
        Arguments:
            model: An OnnxModel or TorchModel, see streaming.load.
            new_search: Creates the decoder of a capture, see
                streaming.search_factory.
            audio_input (AudioInputKind): Where the audio comes from. From
                the MICROPHONE, deliver passes it in.
            settings (Settings): The time limits.
            model_name (str): The model's name for Info.
            language_model (str): The language model's name for Info, if
                new_search uses one.
        """
        self.model = model
        self.new_search = new_search
        self.audio_input = audio_input
        self.settings = settings or Settings()
        self.model_name = model_name
        self.language_model = language_model
        self.bus = EventBus()
        self.state = DialogState.IDLE
        self.capture = None
        # The card shown and the task that clears it.
        self.card = None
        self.card_timer = None
        # A card the client dismissed, which later updates must not show.
        self.dismissed = None

    @property
    def sample_rate(self):
        return self.model.sample_rate

    def snapshot(self):
        """The events that bring a new subscriber up to date."""
        events = [self._state_event()]
        if self.card is not None:
            events.append(ShowCard(card=self.card))
        return events

    def start_capture(self, initiator):
        """Starts a capture and returns it. Raises Busy unless IDLE."""
        if self.state != DialogState.IDLE:
            raise Busy(f"Cannot start a capture while {self.state}.")
        capture = Capture(
            id=uuid.uuid4().hex,
            initiator=initiator,
            transcriber=streaming.Transcriber(self.model, self.new_search()),
            audio=asyncio.Queue(),
        )
        if self.audio_input == AudioInputKind.CLIENT:
            capture.stream_id = uuid.uuid4().hex
        self.capture = capture
        log.info("Capture %s started with %s", capture.id, initiator)
        self._set_state(DialogState.LISTENING)
        self._show(self._transcript_card(capture, "", final=False))
        if capture.stream_id is not None:
            self.bus.publish(self.audio_stream_event(capture))
        capture.task = asyncio.create_task(self._run(capture))
        return capture

    def audio_stream_event(self, capture):
        return StartAudioStream(
            stream_id=capture.stream_id,
            url=f"/v1/audio/{capture.stream_id}",
            sample_rate=self.sample_rate,
        )

    def stop_capture(self, dialog_request_id=None):
        """
        Ends the capture after the audio delivered so far, and returns it.
        Raises Busy unless LISTENING to that capture.
        """
        capture = self.capture
        if self.state != DialogState.LISTENING or capture is None:
            raise Busy(f"No capture to stop while {self.state}.")
        if dialog_request_id is not None and dialog_request_id != capture.id:
            raise Busy(f"Capture {dialog_request_id} is not the current one.")
        capture.audio.put_nowait(_STOP)
        return capture

    def deliver(self, audio, stream_id=None):
        """
        Adds audio to the capture, if LISTENING to one from this source:
        the microphone, or the client stream with the id.
        """
        capture = self.capture
        if capture is None or capture.ended.is_set() or capture.stream_id != stream_id:
            return
        capture.audio.put_nowait(audio)

    def end_stream(self, stream_id):
        """
        The client closed the audio stream with the id, which ends its
        capture after the audio it sent, as stop-capture does.
        """
        capture = self.capture
        if (
            capture is not None
            and capture.stream_id == stream_id
            and not capture.ended.is_set()
        ):
            capture.audio.put_nowait(_STOP)

    def open_stream(self, stream_id):
        """
        Returns the capture that takes the client audio stream with the id,
        which then counts as streaming, or None if there is none or it
        already streams.
        """
        capture = self.capture
        if (
            capture is None
            or capture.stream_id != stream_id
            or capture.streaming
            or capture.ended.is_set()
        ):
            return None
        capture.streaming = True
        return capture

    def card_dismissed(self, card_id):
        """The client dismissed the card. Raises Busy if it is not shown."""
        if self.card is None or self.card.id != card_id:
            raise Busy(f"No card {card_id} is shown.")
        self._cancel_card_timer()
        self.card = None
        self.dismissed = card_id

    def shutdown(self):
        """
        Starts shutting down: ends the event subscriptions and the capture,
        whose audio stream then closes, so no connection keeps a server
        from stopping.
        """
        self.bus.close()
        if self.capture is not None:
            self.capture.audio.put_nowait(_STOP)

    async def close(self):
        """Ends the capture and the event subscriptions."""
        if self.capture is not None and self.capture.task is not None:
            self.capture.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.capture.task
        self._cancel_card_timer()
        self.bus.close()

    async def _run(self, capture):
        """Transcribes the capture's audio until the capture ends."""
        reason = CaptureEndReason.ERROR
        try:
            reason = await self._listen(capture)
        except Exception:
            log.exception("Capture %s failed", capture.id)
        finally:
            capture.ended.set()
        try:
            await self._finish(capture, reason)
        except Exception:
            log.exception("Capture %s failed to finish", capture.id)
            self.capture = None
            self._set_state(DialogState.IDLE)

    async def _listen(self, capture):
        settings = self.settings
        transcriber = capture.transcriber
        heard = 0.0
        text = ""
        while True:
            try:
                chunks = [
                    await asyncio.wait_for(capture.audio.get(), settings.audio_timeout)
                ]
            except TimeoutError:
                return CaptureEndReason.AUDIO_TIMEOUT
            # Take everything queued, so a slow step never builds a delay.
            while not capture.audio.empty():
                chunks.append(capture.audio.get_nowait())
            # Compared by identity, since the chunks are arrays.
            stops = [i for i, chunk in enumerate(chunks) if chunk is _STOP]
            stop = bool(stops)
            if stop:
                chunks = chunks[: stops[0]]

            if chunks:
                audio = np.concatenate(chunks)
                await asyncio.to_thread(transcriber.push, audio)
                heard += len(audio) / self.sample_rate
            if (new_text := transcriber.text().strip()) != text:
                text = new_text
                self.bus.publish(
                    TranscriptUpdated(dialog_request_id=capture.id, text=text)
                )
                self._show(self._transcript_card(capture, text, final=False))

            if stop:
                return CaptureEndReason.STOP_CAPTURE
            if capture.initiator == Initiator.TAP_TO_TALK:
                if text and transcriber.pause_seconds >= settings.end_of_speech:
                    self.bus.publish(EndOfSpeechDetected(dialog_request_id=capture.id))
                    return CaptureEndReason.END_OF_SPEECH
                if not text and heard >= settings.no_speech_timeout:
                    return CaptureEndReason.NO_SPEECH
            if heard >= settings.max_capture:
                return CaptureEndReason.MAX_DURATION

    async def _finish(self, capture, reason):
        """Decodes the final transcript and returns to IDLE."""
        self._set_state(DialogState.THINKING)
        if capture.stream_id is not None:
            self.bus.publish(StopAudioStream(stream_id=capture.stream_id))
        # Flush the frames held back for the lookahead, and let a language
        # model score the end of the utterance.
        await asyncio.to_thread(
            capture.transcriber.push, np.zeros(0, dtype=np.int16), True
        )
        text = capture.transcriber.text(final=True).strip()
        log.info("Capture %s finished with %s: %r", capture.id, reason, text)
        self.bus.publish(
            CaptureFinished(dialog_request_id=capture.id, text=text, reason=reason)
        )
        self._show(self._final_card(capture, text, reason))
        self._start_card_timer()
        self.capture = None
        self._set_state(DialogState.IDLE)

    def _transcript_card(self, capture, text, final):
        title = "You said" if final else "Listening…"
        return TranscriptCard(id=capture.id, title=title, text=text, is_final=final)

    def _final_card(self, capture, text, reason):
        if text:
            return self._transcript_card(capture, text, final=True)
        if reason == CaptureEndReason.AUDIO_TIMEOUT:
            title, body = "No audio", "The microphone sent no audio."
        elif reason == CaptureEndReason.ERROR:
            title, body = "Something went wrong", "Please try again."
        else:
            title, body = "Sorry, I didn't catch that", "Please try again."
        return NoticeCard(id=capture.id, title=title, text=body)

    def _show(self, card):
        """Shows the card, unless the client dismissed it."""
        if card.id == self.dismissed:
            return
        self._cancel_card_timer()
        self.card = card
        self.bus.publish(ShowCard(card=card))

    def _start_card_timer(self):
        if self.card is None:
            return
        card_id = self.card.id

        async def clear():
            await asyncio.sleep(self.settings.card_timeout)
            if self.card is not None and self.card.id == card_id:
                self.card = None
                self.card_timer = None
                self.bus.publish(ClearCard(card_id=card_id))

        self.card_timer = asyncio.create_task(clear())

    def _cancel_card_timer(self):
        if self.card_timer is not None:
            self.card_timer.cancel()
            self.card_timer = None

    def _state_event(self):
        dialog_request_id = self.capture.id if self.capture is not None else None
        return DialogStateChanged(state=self.state, dialog_request_id=dialog_request_id)

    def _set_state(self, state):
        self.state = state
        self.bus.publish(self._state_event())


class Microphone:
    def __init__(self, recognizer, device=None, chunk_ms=100):
        """
        Delivers the audio of a microphone to the recognizer, while
        started.

        Arguments:
            device: The microphone's name or index, by default the
                system default, see `python -m sounddevice`.
        """
        self.recognizer = recognizer
        self.device = device
        self.chunk_ms = chunk_ms
        self.stream = None

    def start(self, loop):
        """Opens the microphone. Its audio is delivered on the loop."""
        recognizer = self.recognizer

        def callback(indata, frames, time_info, status):
            if status:
                log.warning("Microphone: %s", status)
            loop.call_soon_threadsafe(recognizer.deliver, indata[:, 0].copy())

        self.stream = sounddevice.InputStream(
            samplerate=recognizer.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(recognizer.sample_rate * self.chunk_ms / 1e3),
            device=self.device,
            callback=callback,
        )
        self.stream.start()
        name = sounddevice.query_devices(self.stream.device)["name"]
        log.info("Listening on %s", name)

    def stop(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
