"""
Runs Whisper exported by export_whisper.py with ONNX Runtime. The .onnx
file holds the encoder, the decoder and the search, which ONNX Runtime's
WhisperBeamSearch runs, and the vocabulary to turn its tokens into text,
so it needs only NumPy and ONNX Runtime, as on a Raspberry Pi.

Whisper transcribes a whole utterance at once, so WhisperTranscriber
keeps the audio of a capture, finds the end of speech by the loudness of
the audio, and transcribes it when the capture ends. While the capture
goes on, it transcribes the audio so far in the background, for a
partial transcript.
"""

import functools
import json
import logging
import threading

import numpy as np
import onnxruntime

# The version of the metadata export_whisper.py writes.
FORMAT_VERSION = 1
SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
# Whisper hears at most 30 seconds.
CHUNK = 30 * SAMPLE_RATE
# The tokens a transcript can have, at most.
MAX_TOKENS = 224


def is_whisper(session):
    """Whether an ONNX Runtime session runs a model from export_whisper.py."""
    return session.get_modelmeta().custom_metadata_map.get("model_type") == "whisper"


class WhisperModel:
    def __init__(self, session):
        """
        Arguments:
            session (onnxruntime.InferenceSession): Of a model from
                export_whisper.py, see is_whisper.
        """
        meta = session.get_modelmeta().custom_metadata_map
        if int(meta.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("The model is from another version of export_whisper.py.")
        self.session = session
        self.vocabulary = json.loads(meta["vocabulary"])
        self.end = int(meta["end_of_text"])
        # The tokens the transcript starts with: a prompt, if any, then the
        # start of the transcript without timestamps.
        self.prefix = np.array([json.loads(meta["prefix"])], np.int32)
        self.filters = mel_filters(int(meta["n_mels"]))
        self.sample_rate = SAMPLE_RATE

    def transcribe(self, audio, run_options=None):
        """
        The text of 16 bit mono audio at 16 kHz, its first 30 seconds.
        Setting run_options.terminate from another thread stops it with an
        onnxruntime error.
        """
        features = log_mel(audio.astype(np.float32) / 32768, self.filters)
        sequences = self.session.run(
            ["sequences"],
            {
                "input_features": features[None],
                "max_length": np.array([self.prefix.shape[1] + MAX_TOKENS], np.int32),
                "min_length": np.array([0], np.int32),
                "num_beams": np.array([1], np.int32),
                "num_return_sequences": np.array([1], np.int32),
                "length_penalty": np.array([1.0], np.float32),
                "repetition_penalty": np.array([1.0], np.float32),
                "decoder_input_ids": self.prefix,
            },
            run_options,
        )[0]
        tokens = sequences[0, 0, self.prefix.shape[1] :]
        return self.decode(tokens).strip()

    def decode(self, tokens):
        """The text of tokens, up to the end of the text."""
        pieces = []
        for token in tokens.tolist():
            if token >= self.end:
                # The end of the text and the other special tokens.
                break
            pieces.append(self.vocabulary[token])
        data = bytes(BYTE_OF_CHAR[c] for c in "".join(pieces))
        return data.decode("utf-8", errors="replace")


def _bytes_to_unicode():
    """
    GPT-2's map from bytes to the printable characters its vocabulary
    spells them with, which Whisper's tokenizer shares.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = list(printable)
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return dict(zip(printable, map(chr, chars), strict=True))


BYTE_OF_CHAR = {c: b for b, c in _bytes_to_unicode().items()}


@functools.cache
def mel_filters(n_mels, sample_rate=SAMPLE_RATE, n_fft=N_FFT):
    """
    Whisper's mel filterbank, librosa.filters.mel with Slaney's mel scale
    and area normalization, with shape (n_mels, n_fft // 2 + 1).
    """

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        log = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
        return np.where(f >= 1000.0, log, f / (200.0 / 3))

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
        return np.where(m >= 15.0, log, m * (200.0 / 3))

    frequencies = np.linspace(0, sample_rate / 2, 1 + n_fft // 2)
    edges = mel_to_hz(
        np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2)
    )
    widths = np.diff(edges)
    ramps = edges[:, None] - frequencies[None, :]
    rising = -ramps[:-2] / widths[:-1, None]
    falling = ramps[2:] / widths[1:, None]
    weights = np.maximum(0, np.minimum(rising, falling))
    weights *= (2.0 / (edges[2:] - edges[:-2]))[:, None]
    return weights.astype(np.float32)


def log_mel(audio, filters):
    """
    Whisper's input features of float audio, padded or cut to 30 seconds,
    with shape (n_mels, 3000).
    """
    padded = np.zeros(CHUNK, np.float32)
    padded[: min(len(audio), CHUNK)] = audio[:CHUNK]
    padded = np.pad(padded, N_FFT // 2, mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1]
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = filters @ power[:-1].T
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)


log = logging.getLogger(__name__)


class WhisperTranscriber:
    # The loudness of the audio is measured in frames of this many seconds.
    FRAME = 0.03
    # Frames this much louder than the quietest are speech, but at least
    # MIN_SPEECH dBFS.
    ABOVE_NOISE = 15.0
    MIN_SPEECH = -50.0
    # This much speech in all counts as having heard speech, rather than a
    # click or a cough.
    SPEECH_SECONDS = 0.25

    def __init__(self, model, partial_seconds=1.0):
        """
        Transcribes one capture with a WhisperModel, with the interface of
        streaming.Transcriber that speech.backend.engine uses.

        Arguments:
            model (WhisperModel): Transcribes the audio.
            partial_seconds (float): Transcribes the audio so far in the
                background after each this much more audio, once speech
                was heard, for a partial transcript. 0 for none.
        """
        self.model = model
        self.partial_seconds = partial_seconds
        self.frame = int(self.FRAME * model.sample_rate)
        self.audio = []
        self.samples = 0
        self.leftover = np.zeros(0, np.int16)
        self.noise = None
        self.speech_frames = 0
        self.quiet_frames = 0
        self.transcript = ""
        # The latest partial transcript, and the background transcription
        # with the options that stop it.
        self.partial = ""
        self.partial_at = 0
        self.worker = None
        self.run_options = None

    def push(self, audio, final=False):
        """
        Adds audio, and with final transcribes all of it, stopping a partial
        transcription still running.
        """
        self.audio.append(audio)
        self.samples += len(audio)
        samples = np.concatenate([self.leftover, audio])
        n = len(samples) // self.frame
        self.leftover = samples[n * self.frame :]
        for frame in samples[: n * self.frame].reshape(n, self.frame):
            self._measure(frame)
        if final:
            self._stop_partial()
            if self.heard_speech:
                self.transcript = self.model.transcribe(np.concatenate(self.audio))
        elif self._partial_due():
            self._start_partial()

    def _partial_due(self):
        interval = self.partial_seconds * self.model.sample_rate
        return (
            self.partial_seconds > 0
            and self.heard_speech
            and self.samples - self.partial_at >= interval
            and (self.worker is None or not self.worker.is_alive())
        )

    def _start_partial(self):
        audio = np.concatenate(self.audio)
        self.partial_at = self.samples
        run_options = onnxruntime.RunOptions()

        def transcribe():
            try:
                text = self.model.transcribe(audio, run_options)
            except Exception:
                if not run_options.terminate:
                    log.exception("A partial transcription failed")
                return
            if not run_options.terminate:
                self.partial = text

        self.run_options = run_options
        self.worker = threading.Thread(target=transcribe, daemon=True)
        self.worker.start()

    def _stop_partial(self):
        if self.worker is not None:
            self.run_options.terminate = True
            self.worker.join()
            self.worker = None

    def _measure(self, frame):
        power = np.mean(np.square(frame, dtype=np.float64))
        level = 10 * np.log10(power / 32768**2 + 1e-12)
        # The quietest level so far, rising slowly, so the noise can change.
        self.noise = level if self.noise is None else min(level, self.noise + 0.01)
        if level >= max(self.noise + self.ABOVE_NOISE, self.MIN_SPEECH):
            self.speech_frames += 1
            self.quiet_frames = 0
        else:
            self.quiet_frames += 1

    @property
    def heard_speech(self):
        return self.speech_frames * self.FRAME >= self.SPEECH_SECONDS

    @property
    def pause_seconds(self):
        """The time since the last speech."""
        return self.quiet_frames * self.FRAME

    def text(self, final=False):
        """
        Whisper's transcript after the final push, else the latest partial
        transcript.
        """
        return self.transcript if final else self.partial
