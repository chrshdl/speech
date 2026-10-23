"""
Voice activity detection with Silero VAD, github.com/snakers4/silero-vad,
a small recurrent network exported to ONNX, which gives the probability
that each 32 ms frame of 16 kHz audio is speech. Speech starts when most
of the last frames are speech, and ends when few are, so a single frame
neither starts nor ends it.

Needs only NumPy and ONNX Runtime, not PyTorch, so it suits small
devices such as a Raspberry Pi.
"""

import collections

import numpy as np

SAMPLE_RATE = 16000
# The samples of each frame, and of the frame before it that the model
# also takes, as Silero VAD's own wrapper gives it.
FRAME = 512
CONTEXT = 64
# The lowest probability of a frame of speech.
THRESHOLD = 0.5
# Speech starts when at least START of the last WINDOW frames are speech,
# and ends when at most END are.
WINDOW = 10
START = 7
END = 3


class VadModel:
    def __init__(self, path, threads=1):
        """
        Loads Silero VAD's silero_vad.onnx, version 5, run with the
        threads, one by default, as each frame is little work.
        """
        from speech.onnx_model import open_session

        self.session = open_session(path, threads)
        inputs = {i.name for i in self.session.get_inputs()}
        if inputs != {"input", "state", "sr"}:
            raise ValueError(f"{path} is not Silero VAD version 5.")
        self.sample_rate = SAMPLE_RATE

    def stream(self, **kwargs):
        """A VadStream of the model, see there for the arguments."""
        return VadStream(self, **kwargs)


class VadStream:
    def __init__(
        self,
        model,
        threshold=THRESHOLD,
        window=WINDOW,
        start=START,
        end=END,
        history=False,
    ):
        """
        Hears speech in a stream of 16 bit audio at 16 kHz with a VadModel.

        Arguments:
            threshold (float): The lowest probability of a frame of speech.
            window (int): The frames counted.
            start (int): The frames of speech among them that start speech.
            end (int): The frames of speech among them, at most, that end it.
            history (bool): Whether to keep which frames are speech, for
                speech_seconds, which a stream that runs all along must not.
        """
        self.model = model
        self.threshold = threshold
        self.start = start
        self.end = end
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)
        self.buffer = np.zeros(0, np.float32)
        self.recent = collections.deque(maxlen=window)
        # Whether it hears speech, whether it ever did, and the frames
        # since it last did.
        self.speech = False
        self.heard_speech = False
        self.quiet_frames = 0
        # Whether each frame is speech, with history.
        self.frames = [] if history else None

    @property
    def silence_seconds(self):
        """The time since speech ended, 0 while speech goes on."""
        return self.quiet_frames * FRAME / SAMPLE_RATE

    def speech_seconds(self, after=0.0):
        """The seconds of the frames of speech after the time in the stream."""
        first = max(0, int(after * SAMPLE_RATE / FRAME))
        return sum(self.frames[first:]) * FRAME / SAMPLE_RATE

    def push(self, audio):
        """
        Adds the next chunk of 16 bit samples, and returns the
        probability of speech of each frame it completes.
        """
        audio = np.concatenate([self.buffer, np.asarray(audio, np.float32) / 32768])
        count = len(audio) // FRAME
        self.buffer = audio[count * FRAME :]
        probs = []
        for i in range(count):
            frame = audio[i * FRAME : (i + 1) * FRAME]
            prob, self.state = self.model.session.run(
                None,
                {
                    "input": np.concatenate([self.context, frame])[None],
                    "state": self.state,
                    "sr": np.array(SAMPLE_RATE, np.int64),
                },
            )
            self.context = frame[-CONTEXT:]
            prob = float(prob[0, 0])
            probs.append(prob)
            frame_speech = prob >= self.threshold
            if self.frames is not None:
                self.frames.append(frame_speech)
            self.recent.append(frame_speech)
            heard = sum(self.recent)
            if not self.speech and heard >= self.start:
                self.speech = self.heard_speech = True
            elif self.speech and heard <= self.end:
                self.speech = False
            self.quiet_frames = 0 if self.speech else self.quiet_frames + 1
        return probs
