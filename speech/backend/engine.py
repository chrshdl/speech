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

from speech import wake_word as wake_word_module
from speech.backend.messages import (
    AudioInputKind,
    CaptureEndReason,
    CaptureFinished,
    ClearCard,
    CommandRecognized,
    DialogState,
    DialogStateChanged,
    EndOfSpeechDetected,
    Initiator,
    IntentRecognized,
    Message,
    NoticeCard,
    ShowCard,
    ShowPopup,
    Speak,
    StartAudioStream,
    StopAudioStream,
    StopSpeaking,
    TranscriptCard,
    TranscriptUpdated,
)

log = logging.getLogger(__name__)

# Ends a capture's audio, see Recognizer.stop_capture, or the capture at
# once, without an answer, see Recognizer.card_dismissed.
_STOP = object()
_CANCEL = object()
# A COMMANDS capture drops the audio of a transcript without speech once it
# is this long and this quiet, keeping the stream going.
_DROP_SILENCE = 2.0
_DROP_PAUSE = 1.0
# The longest rest of the word after the wake word's alignment that is
# also dropped from the transcript.
_WAKE_WORD_REST = 0.3
# With a voice activity detector, the speech a WAKE_WORD capture needs to
# hear after the wake word, in frames of speech, which start this long
# after the transcript of the wake word ends, as the sound can last longer.
_WAKE_WORD_SPEECH = 0.2
_WAKE_WORD_MARGIN = 0.25
# The captures that end at the end of speech.
_ENDS_BY_ITSELF = {Initiator.TAP_TO_TALK, Initiator.WAKE_WORD}
# How a capture can end and still ask for an intent.
_ANSWERABLE = {
    CaptureEndReason.END_OF_SPEECH,
    CaptureEndReason.STOP_CAPTURE,
    CaptureEndReason.MAX_DURATION,
}


@dataclasses.dataclass
class Settings:
    """The time limits of a capture, in seconds."""

    # A pause after speech this long ends a TAP_TO_TALK capture.
    end_of_speech: float = 1.0
    # A TAP_TO_TALK capture that hears no speech for this long ends.
    no_speech_timeout: float = 8.0
    # The longest TAP_TO_TALK or HOLD_TO_TALK capture.
    max_capture: float = 60.0
    # A pause after speech this long ends a command, in a COMMANDS
    # capture. Pauses within short commands, such as between "go" and
    # "left", were up to 0.44 seconds, at a median of 0.1.
    command_pause: float = 0.3
    # A command still going on after this long is matched as it is, and
    # the next starts.
    max_command: float = 3.0
    # The longest COMMANDS capture, such as a game.
    max_commands_capture: float = 1800.0
    # Ends a capture when no audio arrives for this long.
    audio_timeout: float = 5.0
    # A final card is cleared after this long.
    card_timeout: float = 8.0
    # A popup read at a glance, such as the volume just turned, is cleared
    # after this long.
    brief_card_timeout: float = 5.0
    # A card that answers an intent, such as the weather, is cleared after
    # this long, as it takes longer to read.
    answer_card_timeout: float = 30.0
    # The longest wait for the answer to an intent.
    answer_timeout: float = 10.0
    # The audio before the wake word was heard that a WAKE_WORD capture
    # starts with, which holds the wake word and what was said right
    # after it.
    wake_word_lookback: float = wake_word_module.LOOKBACK_SECONDS


class Busy(Exception):
    """The request does not fit the dialog state."""


class Unsupported(Exception):
    """The service cannot do what was requested."""


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
class WakeWord:
    """Starts a capture when the microphone's audio says the wake word."""

    # Hears the wake word, such as a speech.wake_word.Detector. Its
    # push(audio) adds audio and returns whether the wake word was heard,
    # score is the latest decision's, and reset() skips the decisions
    # until a window of new audio.
    detector: object
    # The wake word in the transcriber's labels, and the blank label,
    # with which the capture finds where the wake word ends, to transcribe
    # what was said after it.
    labels: np.ndarray
    blank: int
    # The transcriber's space label, or None if it has none.
    space: int | None = None
    # The lowest alignment score of the wake word in the capture's
    # transcript, see speech.wake_word.align, that confirms it was said.
    # Below it, the wake word was heard by mistake, and the capture ends
    # quietly. See examples/olivia/README.md.
    threshold: float = wake_word_module.ASR_THRESHOLD
    # Hears the wake word instead of the detector while a card plays on,
    # such as the radio, as the detector does, if given, such as a
    # speech.wake_word.SpeechModelDetector behind a GatedDetector: a
    # speakerphone's echo canceller distorts a voice over what it plays.
    playing_detector: object | None = None


@dataclasses.dataclass
class Capture:
    id: str
    initiator: Initiator
    # Such as a streaming.Transcriber.
    transcriber: object
    audio: asyncio.Queue
    # The audio of a WAKE_WORD capture from before the wake word was heard.
    lookback: np.ndarray | None = None
    # Whether anything was said, after the wake word of a WAKE_WORD capture.
    said: bool = False
    # With audio input from the client, the stream to read from and
    # whether a client is streaming it.
    stream_id: str | None = None
    streaming: bool = False
    # Set once the capture no longer takes audio.
    ended: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


@dataclasses.dataclass
class Speech:
    """An answer spoken, which clients stream as it is synthesized."""

    id: str
    # The card it speaks for.
    card_id: str
    text: str
    # The audio of each sentence synthesized so far.
    sentences: list[bytes] = dataclasses.field(default_factory=list)
    # Whether every sentence is synthesized, and whether it was stopped.
    synthesized: bool = False
    stopped: bool = False
    # Set, and replaced, at each change.
    changed: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    async def audio(self):
        """
        Yields the audio of each sentence, as soon as it is synthesized,
        until the last, or until the speech is stopped.
        """
        sent = 0
        while not self.stopped:
            if sent < len(self.sentences):
                yield self.sentences[sent]
                sent += 1
            elif self.synthesized:
                return
            else:
                await self.changed.wait()

    def notify(self):
        changed, self.changed = self.changed, asyncio.Event()
        changed.set()


class Recognizer:
    def __init__(
        self,
        new_transcriber,
        sample_rate,
        audio_input,
        settings=None,
        model_name="",
        language_model=None,
        intents=None,
        handlers=None,
        commands=None,
        voice=None,
        wake_word=None,
        vad=None,
    ):
        """
        Arguments:
            new_transcriber: Creates the transcriber of a capture, such as
                a streaming.Transcriber. Its push(audio, final) adds audio,
                its text(final) is the transcript so far, heard_speech says
                whether speech was heard, pause_seconds how long ago, and
                clear() starts a new transcript within the stream.
            sample_rate (int): The sample rate of the audio.
            audio_input (AudioInputKind): Where the audio comes from. From
                the MICROPHONE, deliver passes it in.
            settings (Settings): The time limits.
            model_name (str): The model's name for Info.
            language_model (str): The language model's name for Info, if
                the transcribers use one.
            intents (optional): Recognizes what a capture asks for, such as
                a speech.intents.IntentRecognizer.
            handlers (dict, optional): Answers each intent by name, with an
                object whose async card(card_id, slots) returns the card to
                show for the intent's slots, or None to show none, and whose
                async close() releases it. For a card that plays on, such
                as a RadioCard, its updates(card), if it has one, yields the
                card again as what plays changes. A handler whose
                stops_playing is true stops what plays.
            commands (optional): Recognizes the commands of a COMMANDS
                capture, as intents does the intents, as they are said:
                its stream() matches each, see IntentRecognizer.stream.
            voice (optional): Speaks the answers whose handler has a
                speech(card) that returns what to say, such as a
                voice.Voice. Its sample_rate is the audio's, and its
                sentences(text) yields each sentence's 16 bit samples.
            wake_word (WakeWord, optional): Starts a WAKE_WORD capture
                when the microphone's audio says the wake word, while
                IDLE, from start until close.
            vad (optional): Hears when speech ends, for the captures that
                end by themselves, such as a speech.vad.VadModel. Its
                stream(history=True) returns a stream whose push(audio)
                adds audio, whose silence_seconds is the time since speech
                ended, and whose speech_seconds(after) is the speech after
                a time in the stream.
        """
        self.new_transcriber = new_transcriber
        self._sample_rate = sample_rate
        self.audio_input = audio_input
        self.settings = settings or Settings()
        self.model_name = model_name
        self.language_model = language_model
        self.intents = intents
        self.handlers = handlers or {}
        self.commands = commands
        self.voice = voice
        self.bus = EventBus()
        self.state = DialogState.IDLE
        self.capture = None
        # The card shown and the task that clears it.
        self.card = None
        self.card_timer = None
        # A card the client dismissed, which later updates must not show.
        self.dismissed = None
        # The card that plays on beneath the cards shown over it, such as a
        # RadioCard, and the task that follows its updates.
        self.playing = None
        self.follow_task = None
        # The popup over the card shown, such as the volume's, and the task
        # that clears it.
        self.popup = None
        self.popup_timer = None
        # The answer spoken last, until it is stopped.
        self.speech = None
        self.wake_word = wake_word
        self.vad = vad
        if wake_word is not None and audio_input != AudioInputKind.MICROPHONE:
            raise ValueError("The wake word needs the microphone's audio.")
        # The microphone's latest audio, and its audio while IDLE, in which
        # the wake word is spotted.
        size = round(self.settings.wake_word_lookback * sample_rate)
        self.lookback = wake_word_module.RingBuffer(size)
        self.wake_audio = asyncio.Queue()
        # Set by a capture, after which the wake word is only heard in new
        # audio.
        self.wake_reset = False
        self.wake_task = None

    @property
    def sample_rate(self):
        return self._sample_rate

    def snapshot(self):
        """The events that bring a new subscriber up to date."""
        events = [self._state_event()]
        # What plays first, so a client plays it beneath the card shown.
        if self.playing is not None and (
            self.card is None or self.card.id != self.playing.id
        ):
            events.append(ShowCard(card=self.playing))
        if self.card is not None:
            events.append(ShowCard(card=self.card))
        if self.popup is not None:
            events.append(ShowPopup(card=self.popup))
        return events

    def start(self):
        """Starts listening for the wake word, if there is one."""
        if self.wake_word is not None and self.wake_task is None:
            self.wake_task = asyncio.create_task(self._spot_wake_word())

    def start_capture(self, initiator):
        """
        Starts a capture and returns it. Raises Busy unless IDLE, and
        Unsupported for a COMMANDS capture without commands to recognize,
        and for a WAKE_WORD capture, which only the wake word starts.
        """
        if self.state != DialogState.IDLE or self.capture is not None:
            raise Busy(f"Cannot start a capture while {self.state}.")
        if initiator == Initiator.COMMANDS and self.commands is None:
            raise Unsupported("This service recognizes no commands.")
        if initiator == Initiator.WAKE_WORD:
            raise Unsupported("A WAKE_WORD capture starts when the wake word is heard.")
        return self._start_capture(initiator)

    def _start_capture(self, initiator, lookback=None):
        capture = Capture(
            id=uuid.uuid4().hex,
            initiator=initiator,
            transcriber=self.new_transcriber(),
            audio=asyncio.Queue(),
            lookback=lookback,
        )
        if self.wake_word is not None:
            self.wake_reset = True
        if self.audio_input == AudioInputKind.CLIENT:
            capture.stream_id = uuid.uuid4().hex
        self.capture = capture
        log.info("Capture %s started with %s", capture.id, initiator)
        # Neither talked over nor heard by the microphone.
        self._stop_speaking()
        self._set_state(DialogState.LISTENING)
        # Commands steer what is shown, such as a game, rather than show
        # their transcript.
        if initiator != Initiator.COMMANDS:
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
        the microphone, or the client stream with the id. The microphone's
        audio while IDLE is searched for the wake word.
        """
        if self.wake_word is not None and stream_id is None:
            self.lookback.write(audio)
            if self.state == DialogState.IDLE and self.capture is None:
                self.wake_audio.put_nowait(audio)
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

    async def answer(self, name):
        """
        Answers the intent by name as if it was heard, without slots, such as
        for a button, shows its card and returns it, or None if it shows
        none. Raises Busy unless IDLE, and KeyError for an intent the service
        does not answer.
        """
        if self.handlers.get(name) is None:
            raise KeyError(name)
        if self.state != DialogState.IDLE:
            raise Busy(f"Cannot answer {name} while {self.state}.")
        card_id = uuid.uuid4().hex
        log.info("Answering %s for %s", name, card_id)
        card = await self._card(name, card_id, {})
        self._present(name, card, card_id, self.settings.answer_card_timeout)
        return card

    def card_dismissed(self, card_id):
        """
        The client dismissed the card, which, of the capture listening,
        ends it at once, without an answer. Raises Busy if it is neither
        shown nor plays beneath the card shown.
        """
        if self.popup is not None and self.popup.id == card_id:
            self._cancel_popup_timer()
            self.popup = None
            return
        shown = self.card is not None and self.card.id == card_id
        playing = self.playing is not None and self.playing.id == card_id
        if not (shown or playing):
            raise Busy(f"No card {card_id} is shown.")
        self.dismissed = card_id
        if playing:
            # The client stopped it already.
            self._stop_playing(publish=False)
        if shown:
            self._cancel_card_timer()
            self.card = None
            if self.speech is not None and self.speech.card_id == card_id:
                self._stop_speaking()
            # What plays beneath it shows again.
            if self.playing is not None:
                self._show(self.playing)
        capture = self.capture
        if (
            capture is not None
            and capture.id == card_id
            and self.state == DialogState.LISTENING
        ):
            capture.audio.put_nowait(_CANCEL)

    def shutdown(self):
        """
        Starts shutting down: ends the event subscriptions and the capture,
        whose audio stream then closes, so no connection keeps a server
        from stopping.
        """
        # Its audio stream ends, too.
        self._stop_speaking()
        self.bus.close()
        if self.capture is not None:
            self.capture.audio.put_nowait(_STOP)

    async def close(self):
        """
        Ends the capture, listening for the wake word, following what plays
        and the event subscriptions.
        """
        tasks = [self.wake_task, self.follow_task]
        if self.capture is not None:
            tasks.append(self.capture.task)
        for task in tasks:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._cancel_card_timer()
        self._cancel_popup_timer()
        self._stop_speaking()
        self.bus.close()
        for handler in self.handlers.values():
            # None for an intent the service cannot answer, such as the
            # brightness without a backlight.
            if handler is not None:
                await handler.close()

    async def _spot_wake_word(self):
        """Starts a WAKE_WORD capture when the audio while IDLE says the wake word."""
        wake = self.wake_word
        detector = wake.detector
        while True:
            chunks = [await self.wake_audio.get()]
            while not self.wake_audio.empty():
                chunks.append(self.wake_audio.get_nowait())
            hearing = wake.detector
            if self.playing is not None and wake.playing_detector is not None:
                hearing = wake.playing_detector
            if self.wake_reset or hearing is not detector:
                # The audio went to a capture, or the other detector, since,
                # so the window has a gap.
                hearing.reset()
                self.wake_reset = False
            detector = hearing
            heard = await asyncio.to_thread(detector.push, np.concatenate(chunks))
            if heard and self.state == DialogState.IDLE and self.capture is None:
                log.info(
                    "Heard the wake word, score %.3f%s",
                    detector.score,
                    ", while playing" if detector is wake.playing_detector else "",
                )
                self._start_capture(Initiator.WAKE_WORD, self.lookback.read())

    def _confirm_wake_word(self, capture):
        """
        Starts the WAKE_WORD capture's transcript after the wake word, if it
        says the wake word, and returns the seconds of audio before that, or
        None if it does not say it.
        """
        transcriber = capture.transcriber
        wake = self.wake_word
        probs = transcriber.probs()
        aligned = wake_word_module.align(probs, wake.labels, wake.blank)
        score = aligned[0] if aligned is not None else -np.inf
        if score < wake.threshold:
            log.info(
                "Capture %s does not say the wake word, score %.2f", capture.id, score
            )
            return None
        log.info("Capture %s says the wake word, score %.2f", capture.id, score)
        # Also the rest of the word, when the transcript spells the wake
        # word another way, such as "olivier".
        longest = round(_WAKE_WORD_REST / transcriber.frame_seconds)
        end = wake_word_module.word_end(
            probs, aligned[2], wake.blank, wake.space, longest
        )
        transcriber.drop(end)
        return end * transcriber.frame_seconds

    async def _run(self, capture):
        """Transcribes the capture's audio until the capture ends."""
        reason = CaptureEndReason.ERROR
        listen = self._listen
        if capture.initiator == Initiator.COMMANDS:
            listen = self._listen_commands
        try:
            reason = await listen(capture)
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

    async def _next_chunks(self, capture):
        """
        Waits for the capture's audio and returns all the chunks queued,
        and what ended the capture after them, _STOP or _CANCEL, or None.
        Raises TimeoutError when none arrives in time.
        """
        chunks = [
            await asyncio.wait_for(capture.audio.get(), self.settings.audio_timeout)
        ]
        # Take everything queued, so a slow step never builds a delay.
        while not capture.audio.empty():
            chunks.append(capture.audio.get_nowait())
        # Compared by identity, since the chunks are arrays.
        ends = [
            i for i, chunk in enumerate(chunks) if chunk is _STOP or chunk is _CANCEL
        ]
        if ends:
            return chunks[: ends[0]], chunks[ends[0]]
        return chunks, None

    async def _listen(self, capture):
        settings = self.settings
        transcriber = capture.transcriber
        # The audio after the capture started, so the time limits count
        # from when the wake word was heard.
        heard = 0.0
        text = ""
        # Hears when speech ends, if there is a voice activity detector, or
        # else the transcript's pauses do.
        voice = self.vad.stream(history=True) if self.vad is not None else None

        def push(audio):
            transcriber.push(audio)
            if voice is not None:
                voice.push(audio)

        # Whether the transcript starts where it should, after the wake word
        # of a WAKE_WORD capture, and the seconds of audio before that.
        started = capture.lookback is None
        after = 0.0
        if not started:
            await asyncio.to_thread(push, capture.lookback)
        while True:
            try:
                chunks, stop = await self._next_chunks(capture)
            except TimeoutError:
                return CaptureEndReason.AUDIO_TIMEOUT
            if stop is _CANCEL:
                return CaptureEndReason.CANCELLED
            if chunks:
                audio = np.concatenate(chunks)
                await asyncio.to_thread(push, audio)
                heard += len(audio) / self.sample_rate
            if not started:
                if heard < wake_word_module.CONFIRM_SECONDS and not stop:
                    continue
                after = await asyncio.to_thread(self._confirm_wake_word, capture)
                if after is None:
                    return CaptureEndReason.NO_WAKE_WORD
                started = True
            # Whether anything was said, after the wake word. The transcript
            # can keep a letter of the wake word, so with a voice activity
            # detector, it must also hear speech after the wake word.
            speech = transcriber.heard_speech
            if voice is not None and capture.lookback is not None:
                speech = speech and (
                    voice.speech_seconds(after + _WAKE_WORD_MARGIN) >= _WAKE_WORD_SPEECH
                )
            capture.said = speech
            if speech and (new_text := transcriber.text().strip()) != text:
                text = new_text
                self.bus.publish(
                    TranscriptUpdated(dialog_request_id=capture.id, text=text)
                )
                self._show(self._transcript_card(capture, text, final=False))

            if stop:
                return CaptureEndReason.STOP_CAPTURE
            if capture.initiator in _ENDS_BY_ITSELF:
                # Something was said, and then nothing.
                pause = transcriber.pause_seconds
                if voice is not None:
                    pause = voice.silence_seconds
                if speech and pause >= settings.end_of_speech:
                    self.bus.publish(EndOfSpeechDetected(dialog_request_id=capture.id))
                    return CaptureEndReason.END_OF_SPEECH
                if not speech and heard >= settings.no_speech_timeout:
                    return CaptureEndReason.NO_SPEECH
            if heard >= settings.max_capture:
                return CaptureEndReason.MAX_DURATION

    async def _listen_commands(self, capture):
        """
        Recognizes commands in a COMMANDS capture's audio until it is
        stopped. A pause ends each command, which is matched as it is said,
        so it is reported as soon as it is recognized, once.
        """
        settings = self.settings
        transcriber = capture.transcriber
        heard = 0.0
        # The audio of the transcript, which starts anew with each command,
        # and its match so far, which aligns each frame once.
        said = 0.0
        matcher = self.commands.stream()
        recognized = False
        while True:
            try:
                chunks, stop = await self._next_chunks(capture)
            except TimeoutError:
                return CaptureEndReason.AUDIO_TIMEOUT
            # A chunk at a time, so commands said one after another stay
            # apart also when their audio queued up.
            for chunk in chunks:
                await asyncio.to_thread(transcriber.push, chunk)
                heard += len(chunk) / self.sample_rate
                said += len(chunk) / self.sample_rate
                if transcriber.heard_speech:
                    ended = (
                        transcriber.pause_seconds >= settings.command_pause
                        or said >= settings.max_command
                    )
                    if not recognized:
                        if ended:
                            # Flush the frames held back for the
                            # lookahead, so all the command is matched.
                            await asyncio.to_thread(
                                transcriber.push, np.zeros(0, np.int16), True
                            )
                        recognized = await self._command(capture, matcher, final=ended)
                    if ended:
                        transcriber.clear()
                        said, recognized = 0.0, False
                        matcher = self.commands.stream()
                elif said >= _DROP_SILENCE and transcriber.pause_seconds >= _DROP_PAUSE:
                    transcriber.clear()
                    said = 0.0
                    matcher = self.commands.stream()
            if stop is _CANCEL:
                return CaptureEndReason.CANCELLED
            if stop:
                return CaptureEndReason.STOP_CAPTURE
            if heard >= settings.max_commands_capture:
                return CaptureEndReason.MAX_DURATION

    async def _command(self, capture, matcher, final):
        """
        Publishes the command the capture's transcript says, if it says
        one, and returns whether it does. The matcher, of the commands'
        stream, has matched the transcript so far.
        """
        transcriber = capture.transcriber
        match = await asyncio.to_thread(matcher.closest_heard, transcriber)
        if match is None or match.score < self.commands.threshold:
            return False
        text = transcriber.text(final).strip()
        log.info("Command %s %r, score %.2f", match.intent.name, text, match.score)
        self.bus.publish(
            CommandRecognized(
                dialog_request_id=capture.id,
                command=match.intent.name,
                score=match.score,
                text=text,
            )
        )
        return True

    async def _finish(self, capture, reason):
        """Decodes the final transcript and returns to IDLE."""
        if capture.initiator == Initiator.COMMANDS:
            # Its commands were all reported, and the card it steered stays.
            if capture.stream_id is not None:
                self.bus.publish(StopAudioStream(stream_id=capture.stream_id))
            log.info("Capture %s finished with %s", capture.id, reason)
            self.bus.publish(
                CaptureFinished(dialog_request_id=capture.id, text="", reason=reason)
            )
            self.capture = None
            self._set_state(DialogState.IDLE)
            return
        if reason == CaptureEndReason.NO_WAKE_WORD:
            # Heard by mistake, it ends quietly, without a transcript or notice.
            self._end_quietly(capture, reason, "")
            return
        if reason == CaptureEndReason.CANCELLED:
            if capture.stream_id is not None:
                self.bus.publish(StopAudioStream(stream_id=capture.stream_id))
            self._end_quietly(capture, reason, "")
            return
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
        if capture.initiator == Initiator.WAKE_WORD and not (text and capture.said):
            # Nothing was said after the wake word, so the capture ends
            # quietly, without a notice.
            self._end_quietly(capture, reason, text)
            return
        self.bus.publish(
            CaptureFinished(dialog_request_id=capture.id, text=text, reason=reason)
        )
        card, timeout, intent = await self._answer(capture, text, reason)
        self._present(intent, card, capture.id, timeout)
        self.capture = None
        self._set_state(DialogState.IDLE)

    async def _answer(self, capture, text, reason):
        """
        Returns the card to show at the end of the capture, how long to
        show it, and the intent it answers: the answer to the intent the
        capture asks for, or else the final transcript, for no intent.
        """
        match = None
        if text and self.intents is not None and reason in _ANSWERABLE:
            match = await asyncio.to_thread(
                self.intents.closest_heard, capture.transcriber
            )
        if match is None or match.score < self.intents.threshold:
            if match is not None:
                log.info(
                    "Capture %s is no intent, closest %s, score %.2f",
                    capture.id,
                    match.intent.name,
                    match.score,
                )
            card = self._final_card(capture, text, reason)
            return card, self.settings.card_timeout, None
        name = match.intent.name
        log.info(
            "Capture %s asked for %s %s, score %.2f",
            capture.id,
            name,
            match.slots,
            match.score,
        )
        self.bus.publish(
            IntentRecognized(
                dialog_request_id=capture.id,
                intent=name,
                score=match.score,
                slots=match.slots,
            )
        )
        card = await self._card(name, capture.id, match.slots)
        return card, self.settings.answer_card_timeout, name

    def _present(self, name, card, card_id, timeout):
        """
        Shows the card that answers the intent by name, or with no intent the
        capture's last, for so many seconds, speaks it and follows what it
        plays. With no card to show, the card with card_id, the capture's
        transcript, goes.
        """
        if card is None:
            self._uncover(card_id)
            return
        if card.id == self.dismissed:
            # Neither shown nor spoken, such as when the client dismissed the
            # capture's card while its answer was found.
            return
        if getattr(card, "POPUP", False):
            # Over the card shown, which stays, but for the capture's own.
            self._uncover(card_id)
            self._pop_up(card)
            return
        self._show(card)
        if name is not None:
            self._speak(name, card)
            self._follow(name, card)
        # A card such as a game's stays until the client dismisses it.
        if not getattr(card, "UNTIL_DISMISSED", False):
            self._start_card_timer(timeout)

    async def _card(self, name, card_id, slots):
        """
        The card of the handler of the intent by name, None for none, or a
        notice that it could not answer.
        """
        handler = self.handlers[name]
        if getattr(handler, "stops_playing", False):
            self._stop_playing()
        try:
            return await asyncio.wait_for(
                handler.card(card_id, slots),
                self.settings.answer_timeout,
            )
        except Exception:
            log.exception("%s could not be answered for %s", name, card_id)
            return NoticeCard(
                id=card_id,
                title="Sorry, I can't answer that right now",
                text="Please try again later.",
            )

    def _speak(self, name, card):
        """
        Speaks the card that answers the intent by name, if there is a voice
        and its handler says what to say for it.
        """
        say = getattr(self.handlers.get(name), "speech", None)
        text = say(card) if self.voice is not None and say is not None else None
        if not text:
            return
        self._stop_speaking()
        speech = Speech(id=uuid.uuid4().hex, card_id=card.id, text=text)
        self.speech = speech
        log.info("Speaking %s for %s: %r", speech.id, card.id, text)
        speech.task = asyncio.create_task(self._synthesize(speech))
        self.bus.publish(
            Speak(
                speech_id=speech.id,
                card_id=card.id,
                text=text,
                url=f"/v1/speech/{speech.id}",
                sample_rate=self.voice.sample_rate,
            )
        )

    async def _synthesize(self, speech):
        """Synthesizes the speech a sentence at a time, until stopped."""
        sentences = self.voice.sentences(speech.text)
        try:
            while not speech.stopped:
                audio = await asyncio.to_thread(next, sentences, None)
                if audio is None:
                    break
                speech.sentences.append(audio)
                speech.notify()
        except Exception:
            log.exception("Speech %s failed", speech.id)
        finally:
            speech.synthesized = True
            speech.notify()

    def _stop_speaking(self):
        """
        Stops the speech, also on clients still playing it after it was all
        synthesized, which end its audio stream.
        """
        speech = self.speech
        if speech is None:
            return
        self.speech = None
        speech.stopped = True
        speech.notify()
        self.bus.publish(StopSpeaking(speech_id=speech.id))

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
        if self.speech is not None and self.speech.card_id != card.id:
            self._stop_speaking()
        ended = None
        if (
            self.playing is not None
            and self.playing.id != card.id
            and getattr(card, "UNTIL_DISMISSED", False)
        ):
            # A card that stays, such as a song's or the radio asked for
            # again, ends what plays, cleared after the card shows, so a
            # client can play on what both play.
            ended = self.playing
            self._stop_playing(publish=False)
        if getattr(card, "PLAYS_ON", False):
            self.playing = card
        self.card = card
        self.bus.publish(ShowCard(card=card))
        if ended is not None:
            self.bus.publish(ClearCard(card_id=ended.id))

    def _pop_up(self, card):
        """
        Shows the card as a popup over the card shown, and clears it after
        brief_card_timeout, unless another popup replaces it sooner.
        """
        self._cancel_popup_timer()
        self.popup = card
        self.bus.publish(ShowPopup(card=card))

        async def clear():
            await asyncio.sleep(self.settings.brief_card_timeout)
            if self.popup is card:
                self.popup_timer = None
                self.popup = None
                self.bus.publish(ClearCard(card_id=card.id))

        self.popup_timer = asyncio.create_task(clear())

    def _cancel_popup_timer(self):
        if self.popup_timer is not None:
            self.popup_timer.cancel()
            self.popup_timer = None

    def _uncover(self, card_id):
        """
        The card with the id goes, if shown: the card that plays beneath it
        shows again, or else it is cleared.
        """
        if self.card is None or self.card.id != card_id:
            return
        self._cancel_card_timer()
        if self.playing is not None and self.playing.id != card_id:
            self._show(self.playing)
        else:
            self.card = None
            self.bus.publish(ClearCard(card_id=card_id))

    def _follow(self, name, card):
        """
        Follows the updates of the card, if it plays on, from the handler of
        the intent by name, such as the song on the radio.
        """
        updates = getattr(self.handlers.get(name), "updates", None)
        if updates is None or self.playing is not card:
            return
        self.follow_task = asyncio.create_task(self._follow_updates(updates(card)))

    async def _follow_updates(self, updates):
        try:
            async with contextlib.aclosing(updates):
                async for card in updates:
                    if self.playing is None or self.playing.id != card.id:
                        return
                    self.playing = card
                    if self.card is not None and self.card.id == card.id:
                        self.card = card
                        self.bus.publish(ShowCard(card=card))
        except Exception:
            log.exception("Following what plays failed")

    def _stop_playing(self, publish=True):
        """Stops what plays, and with publish, clears its card on the clients."""
        playing = self.playing
        if playing is None:
            return
        log.info("Stopped playing %s", playing.id)
        self.playing = None
        if self.follow_task is not None:
            self.follow_task.cancel()
            self.follow_task = None
        if self.card is not None and self.card.id == playing.id:
            self.card = None
        if publish:
            self.bus.publish(ClearCard(card_id=playing.id))

    def _end_quietly(self, capture, reason, text):
        """
        Ends the capture without showing a card, and its own goes, for what
        plays beneath it.
        """
        if reason == CaptureEndReason.NO_WAKE_WORD:
            log.info("Capture %s finished with %s", capture.id, reason)
        self.bus.publish(
            CaptureFinished(dialog_request_id=capture.id, text=text, reason=reason)
        )
        self._uncover(capture.id)
        self.capture = None
        self._set_state(DialogState.IDLE)

    def _start_card_timer(self, timeout):
        if self.card is None:
            return
        card_id = self.card.id

        async def clear():
            await asyncio.sleep(timeout)
            if self.card is not None and self.card.id == card_id:
                self.card_timer = None
                if self.speech is not None and self.speech.card_id == card_id:
                    self._stop_speaking()
                self._uncover(card_id)

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
