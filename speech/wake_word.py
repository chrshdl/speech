"""
Spots a wake word, such as "olivia", in a stream of audio, with a small
keyword spotting model of "Hello Edge: Keyword Spotting on
Microcontrollers" (Zhang et al., 2017), see speech.models.ds_cnn.

The model classifies each second of audio, from 10 MFCCs of 40 ms frames
every 20 ms, as silence, other speech or the wake word. A Detector runs
it on the last second every 100 ms, and hears the wake word when its
probability, averaged over the last few windows, reaches a threshold.

Needs only NumPy and ONNX Runtime, not PyTorch, so it suits small
devices such as a Raspberry Pi.
"""

import collections
import json
import math

import numpy as np

from speech.intents import WILDCARD, _states, viterbi

# The version of the graph's inputs, outputs and metadata that
# speech.models.ds_cnn.export writes, bumped when they change.
FORMAT_VERSION = 1

# The features of Hello Edge: 10 MFCCs of 40 mel bands from 20 Hz to
# 4 kHz, of 40 ms frames every 20 ms.
WINDOW_MS = 40
STEP_MS = 20
MEL_BANDS = 40
COEFFICIENTS = 10
LOW_HZ = 20
HIGH_HZ = 4000
# The audio of each decision.
CLIP_SECONDS = 1.0

# How often the Detector decides, how many decisions it averages, and the
# lowest average probability of the wake word, see examples/olivia.
HOP_SECONDS = 0.1
SMOOTHING = 3
THRESHOLD = 0.8
# With a voice activity detector, see GatedDetector: the audio before speech
# started that the Detector takes, and how long after it ended it runs on.
LEAD_SECONDS = 1.0
HANGOVER_SECONDS = 1.0
# A speech model confirms each time the Detector hears the wake word: it
# transcribes the audio of LOOKBACK_SECONDS before then and CONFIRM_SECONDS
# after, which holds the end of the wake word, as the Detector often hears
# it before its end, and what the model's lookahead holds back, and the wake word's alignment in its output, see
# align, must score at least ASR_THRESHOLD, see examples/olivia.
LOOKBACK_SECONDS = 2.0
CONFIRM_SECONDS = 1.0
ASR_THRESHOLD = -2.0
# The SpeechModelDetector aligns the wake word to the speech model's output
# of the last SPOT_WINDOW_SECONDS every SPOT_HOP_SECONDS, and hears it at a
# score of SPOT_THRESHOLD, stricter than the confirmation, as nothing
# confirms it but the capture's own confirmation by the same model. With
# the stride 3 model after epoch 4, -1.25 missed 10% of the recorded test
# clips, none of 15 said over the radio playing on a Jabra SPEAK 510, and
# heard it 3 times in the 5.4 hours of LibriSpeech's dev-clean.
SPOT_WINDOW_SECONDS = 1.5
SPOT_HOP_SECONDS = 0.2
SPOT_THRESHOLD = -1.25


def mel_filterbank(sample_rate, fft_size, bands=MEL_BANDS, low=LOW_HZ, high=HIGH_HZ):
    """
    Triangular filters evenly spaced on the mel scale, with shape
    (fft_size // 2 + 1, bands), which map a power spectrum to mel bands.
    """

    def mel(hz):
        return 1127 * np.log1p(np.asarray(hz) / 700)

    edges = np.linspace(mel(low), mel(high), bands + 2)
    bins = mel(np.arange(fft_size // 2 + 1) * sample_rate / fft_size)
    left, center, right = edges[:-2], edges[1:-1], edges[2:]
    rising = (bins[:, None] - left) / (center - left)
    falling = (right - bins[:, None]) / (right - center)
    return np.maximum(0, np.minimum(rising, falling)).astype(np.float32)


def dct_matrix(bands=MEL_BANDS, coefficients=COEFFICIENTS):
    """The orthonormal DCT-II of the log mel bands, with shape (bands, coefficients)."""
    n = np.arange(bands)
    k = np.arange(coefficients)
    dct = np.cos(np.pi / bands * (n[:, None] + 0.5) * k)
    dct *= math.sqrt(2 / bands)
    dct[:, 0] /= math.sqrt(2)
    return dct.astype(np.float32)


class Mfcc:
    def __init__(self, sample_rate):
        """Computes the MFCCs of 16 bit audio at the sample rate."""
        self.sample_rate = sample_rate
        self.window = int(WINDOW_MS * sample_rate / 1e3)
        self.hop = int(STEP_MS * sample_rate / 1e3)
        self.fft_size = 1 << (self.window - 1).bit_length()
        n = np.arange(self.window)
        self.hann = (0.5 - 0.5 * np.cos(2 * np.pi * n / self.window)).astype(np.float32)
        self.filterbank = mel_filterbank(sample_rate, self.fft_size)
        self.dct = dct_matrix()

    def frames(self, samples):
        """The number of frames of so many samples."""
        return max((samples - self.window) // self.hop + 1, 0)

    def __call__(self, audio):
        """The MFCCs of the audio's frames, with shape (frames, COEFFICIENTS)."""
        n = self.frames(len(audio))
        if n == 0:
            return np.zeros((0, COEFFICIENTS), np.float32)
        audio = np.asarray(audio, np.float32) / 32768
        frames = np.lib.stride_tricks.sliding_window_view(audio, self.window)
        spectrum = np.fft.rfft(
            frames[: n * self.hop : self.hop] * self.hann, self.fft_size
        )
        power = spectrum.real**2 + spectrum.imag**2
        return np.log(power @ self.filterbank + 1e-6) @ self.dct


class MfccStream:
    def __init__(self, sample_rate):
        """
        Computes MFCCs incrementally as audio arrives. The frames returned
        for consecutive chunks are those of the whole audio.
        """
        self.mfcc = Mfcc(sample_rate)
        self.buffer = np.zeros(0, np.int16)

    def push(self, audio):
        """Adds the next chunk of 16 bit samples and returns the new frames."""
        audio = np.concatenate([self.buffer, audio])
        n = self.mfcc.frames(len(audio))
        self.buffer = audio[n * self.mfcc.hop :]
        return self.mfcc(audio[: (n - 1) * self.mfcc.hop + self.mfcc.window])


class WakeWordModel:
    def __init__(self, path, threads=1):
        """
        Loads a model exported by speech.models.ds_cnn.export, run with
        the threads, one by default, as each decision is little work.
        """
        from speech.onnx_model import open_session

        self.session = open_session(path, threads)
        meta = self.session.get_modelmeta().custom_metadata_map
        if int(meta.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(
                "The model is from another version of speech.models.ds_cnn."
            )
        self.labels = json.loads(meta["labels"])
        self.keyword = meta["keyword"]
        self.keyword_index = self.labels.index(self.keyword)
        self.sample_rate = int(meta["sample_rate"])
        self.mfcc = Mfcc(self.sample_rate)
        self.clip_frames = self.mfcc.frames(round(CLIP_SECONDS * self.sample_rate))

    def probs(self, windows):
        """
        The probabilities of the labels for windows of MFCCs with shape
        (windows, clip_frames, COEFFICIENTS).
        """
        windows = np.asarray(windows, np.float32)
        return self.session.run(None, {"mfcc": windows})[0]

    def scores(self, audio, hop_seconds=HOP_SECONDS, batch=512):
        """
        The probability of the wake word in each window of a whole stream
        of 16 bit audio that a Detector decides on, every hop_seconds.
        """
        frames = self.mfcc(audio)
        hop = hop_frames(self, hop_seconds)
        ends = np.arange(self.clip_frames, len(frames) + 1, hop)
        scores = []
        for i in range(0, len(ends), batch):
            windows = np.stack(
                [frames[end - self.clip_frames : end] for end in ends[i : i + batch]]
            )
            scores.append(self.probs(windows)[:, self.keyword_index])
        return np.concatenate(scores) if scores else np.zeros(0, np.float32)


def hop_frames(model, hop_seconds):
    """The MFCC frames between two decisions."""
    return max(1, round(hop_seconds * 1e3 / STEP_MS))


def decision_end(model, decision, hop_seconds=HOP_SECONDS):
    """The sample after the window of a decision, counted from 0 in a stream."""
    frames = model.clip_frames + decision * hop_frames(model, hop_seconds)
    return (frames - 1) * model.mfcc.hop + model.mfcc.window


def blocked_hops(model, hop_seconds, smoothing):
    """
    The decisions after one that heard the wake word that are skipped,
    until every window averaged starts after the window it was heard in,
    so one utterance of it is heard once.
    """
    return math.ceil(model.clip_frames / hop_frames(model, hop_seconds)) + smoothing - 1


def detections(scores, threshold, smoothing, blocked):
    """
    The indexes of the decisions that hear the wake word in a stream,
    given the wake word's probability in each window, as a Detector
    decides, see blocked_hops for blocked.
    """
    found = []
    recent = collections.deque(maxlen=smoothing)
    skip = 0
    for i, score in enumerate(scores):
        recent.append(score)
        if skip > 0:
            skip -= 1
        elif len(recent) == smoothing and sum(recent) / smoothing >= threshold:
            found.append(i)
            skip = blocked
    return found


class Detector:
    def __init__(
        self, model, threshold=THRESHOLD, smoothing=SMOOTHING, hop_seconds=HOP_SECONDS
    ):
        """
        Hears the wake word in a stream of audio with a WakeWordModel.

        Arguments:
            threshold (float): The lowest probability of the wake word,
                averaged over smoothing decisions, that hears it.
            smoothing (int): The number of decisions averaged.
            hop_seconds (float): The time between decisions.
        """
        self.model = model
        self.threshold = threshold
        self.smoothing = smoothing
        self.hop = hop_frames(model, hop_seconds)
        self.blocked = blocked_hops(model, hop_seconds, smoothing)
        self.sample_rate = model.sample_rate
        self.mfcc = MfccStream(model.sample_rate)
        self.frames = np.zeros((0, COEFFICIENTS), np.float32)
        # Frames until the next decision.
        self.next = model.clip_frames
        self.recent = collections.deque(maxlen=smoothing)
        self.skip = 0
        # The average probability of the latest decision.
        self.score = 0.0

    def push(self, audio):
        """
        Adds the next chunk of 16 bit samples, and returns whether the
        wake word was heard in it.
        """
        frames = np.concatenate([self.frames, self.mfcc.push(audio)])
        clip = self.model.clip_frames
        ends = list(range(self.next, len(frames) + 1, self.hop))
        heard = False
        if ends:
            windows = np.stack([frames[end - clip : end] for end in ends])
            for score in self.model.probs(windows)[:, self.model.keyword_index]:
                self.recent.append(float(score))
                self.score = sum(self.recent) / len(self.recent)
                if self.skip > 0:
                    self.skip -= 1
                elif (
                    len(self.recent) == self.smoothing and self.score >= self.threshold
                ):
                    heard = True
                    self.skip = self.blocked
            self.next = ends[-1] + self.hop
        # Keep the frames from the start of the next window.
        drop = max(0, self.next - clip)
        self.frames = frames[drop:]
        self.next -= drop
        return heard

    def reset(self):
        """
        Skips the decisions until a window of new audio, such as after
        the wake word was heard and the audio went elsewhere for a while.
        """
        self.skip = self.blocked


class GatedDetector:
    def __init__(
        self,
        model,
        vad,
        threshold=THRESHOLD,
        smoothing=SMOOTHING,
        hop_seconds=HOP_SECONDS,
        lead_seconds=LEAD_SECONDS,
        hangover_seconds=HANGOVER_SECONDS,
        new_detector=None,
    ):
        """
        Hears the wake word as a Detector does, but runs it only while a
        voice activity detector hears speech, so that no other sound sets
        it off, and it takes no compute in silence.

        When speech starts, a new Detector takes the last lead_seconds of
        audio, which hold the start of the speech, as the voice activity
        detector hears it only after a few frames, and then the audio as it
        comes, until hangover_seconds after the speech ended, which holds
        the decisions on its end.

        Arguments:
            model (WakeWordModel): The wake word model, or with new_detector,
                anything with the audio's sample_rate.
            vad (speech.vad.VadModel): The voice activity detector.
            threshold, smoothing, hop_seconds: As the Detector's.
            new_detector (optional): Creates the detector to run instead of a
                Detector of the model, such as a SpeechModelDetector.
        """
        self.model = model
        self.vad = vad
        self.new_detector = new_detector or (
            lambda: Detector(model, threshold, smoothing, hop_seconds)
        )
        self.lead = RingBuffer(round(lead_seconds * model.sample_rate))
        self.hangover = round(hangover_seconds * model.sample_rate)
        self.reset()

    def reset(self):
        """
        Waits for speech to start again, such as after the wake word was
        heard and the audio went elsewhere for a while.
        """
        self.voice = self.vad.stream()
        self.detector = None
        # The samples of audio until the detector stops, once speech ended.
        self.left = 0
        self.score = 0.0

    def push(self, audio):
        """
        Adds the next chunk of 16 bit samples, and returns whether the
        wake word was heard in it.
        """
        self.lead.write(audio)
        was_speech = self.voice.speech
        self.voice.push(audio)
        if self.voice.speech or was_speech:
            self.left = self.hangover
        heard = False
        if self.detector is None:
            if self.voice.speech:
                self.detector = self.new_detector()
                heard = self.detector.push(self.lead.read())
        else:
            heard = self.detector.push(audio)
        if self.detector is not None:
            self.score = self.detector.score
            if not self.voice.speech:
                self.left -= len(audio)
                if self.left <= 0:
                    self.detector = None
        return heard


class SpeechModelDetector:
    def __init__(
        self,
        transcriber,
        labels,
        blank,
        threshold=SPOT_THRESHOLD,
        window_seconds=SPOT_WINDOW_SECONDS,
        hop_seconds=SPOT_HOP_SECONDS,
    ):
        """
        Hears the wake word with the speech model, as a Detector does with
        the wake word model, for where that misses it: a speakerphone's echo
        canceller, such as a Jabra SPEAK 510's, removes what it plays, such
        as the radio, from the microphone's audio, and with it much of a
        voice over it, which the wake word model then hardly ever hears as
        the wake word, while the speech model, trained on masked
        spectrograms, still does. Every hop_seconds of audio, the wake word
        is aligned to the model's output of the last window_seconds, see
        align, and heard at a score of threshold. It costs as much compute
        as transcribing, so run it behind a GatedDetector.

        Arguments:
            transcriber: A new speech.streaming.Transcriber, whose
                push(audio) adds audio, whose probs() are its output so
                far, whose frame_seconds is the time of a frame, and whose
                model's sample_rate is the audio's.
            labels (ndarray): The wake word in the model's labels.
            blank (int): The model's blank label.
        """
        self.transcriber = transcriber
        self.labels = labels
        self.blank = blank
        self.threshold = threshold
        self.window = round(window_seconds / transcriber.frame_seconds)
        self.hop = round(hop_seconds * transcriber.model.sample_rate)
        # The decisions skipped after hearing it, until the window holds
        # new audio, so the same word is heard once.
        self.blocked = math.ceil(window_seconds / hop_seconds)
        self.pushed = 0
        self.skip = 0
        # The latest alignment's score, the Detector's probability.
        self.score = float("-inf")

    def push(self, audio):
        """
        Adds the next chunk of 16 bit samples, and returns whether the
        wake word was heard in it.
        """
        self.transcriber.push(audio)
        self.pushed += len(audio)
        if self.pushed < self.hop:
            return False
        self.pushed %= self.hop
        aligned = align(
            self.transcriber.probs()[-self.window :], self.labels, self.blank
        )
        self.score = aligned[0] if aligned is not None else float("-inf")
        if self.skip > 0:
            self.skip -= 1
            return False
        if self.score < self.threshold:
            return False
        self.skip = self.blocked
        return True

    def reset(self):
        """Skips the decisions until a window of new audio, as a Detector's."""
        self.skip = self.blocked


class RingBuffer:
    def __init__(self, size):
        """Keeps the last size samples of a stream of 16 bit audio."""
        self.buffer = np.zeros(size, np.int16)
        self.end = 0
        self.filled = 0

    def write(self, audio):
        audio = np.asarray(audio, np.int16)[-len(self.buffer) :]
        size = len(self.buffer)
        first = min(len(audio), size - self.end)
        self.buffer[self.end : self.end + first] = audio[:first]
        self.buffer[: len(audio) - first] = audio[first:]
        self.end = (self.end + len(audio)) % size
        self.filled = min(size, self.filled + len(audio))

    def read(self):
        """The samples kept, oldest first."""
        return np.roll(self.buffer, -self.end)[len(self.buffer) - self.filled :]


def align(probs, labels, blank):
    """
    Finds a word, such as the wake word, in a CTC model's output.

    Arguments:
        probs (ndarray): The output probabilities, with shape (frames,
            labels + 1).
        labels (ndarray): The word's labels.
        blank (int): The blank label.

    Returns the mean log probability per label of the word's best
    alignment, relative to the best path, so at most 0, and the frame
    where it starts and the frame after it, or None if there are too few
    frames.
    """
    # Any audio can come before and after it.
    sequence = np.concatenate([[WILDCARD], labels, [WILDCARD]])
    if len(probs) < len(sequence):
        return None
    with np.errstate(divide="ignore"):
        log_probs = np.log(probs)
    relative = log_probs - log_probs.max(axis=1, keepdims=True)
    score, path = viterbi(relative, [sequence], blank, path=True)
    if not np.isfinite(score[0]):
        return None
    # The states of the word's labels, from its first to its last.
    states = _states(sequence, blank)
    said = np.flatnonzero((path >= 3) & (path <= len(states) - 4))
    return float(score[0]) / len(labels), int(said[0]), int(said[-1]) + 1


def word_end(probs, frame, blank, space, longest, pause=4):
    """
    The frame after the end of the word a CTC model's output spells at
    the frame, such as the rest of the wake word that its alignment left,
    when it was spelled another way: the next space, or pause of so many
    blank frames, at most longest frames later.

    Arguments:
        probs (ndarray): The output probabilities, with shape (frames,
            labels + 1).
        blank (int): The blank label.
        space (int): The space label, or None if the model has none.
    """
    best = probs[frame : frame + longest].argmax(axis=1)
    end = frame
    blanks = 0
    for i, label in enumerate(best.tolist()):
        if label == space or blanks >= pause:
            break
        if label == blank:
            blanks += 1
        else:
            end, blanks = frame + i + 1, 0
    return end
