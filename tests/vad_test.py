import numpy as np

from speech import vad, wake_word


class ScriptedSession:
    """Answers each frame with the next of the probabilities."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.inputs = []

    def run(self, outputs, inputs):
        self.inputs.append(inputs)
        return np.array([[self.probs.pop(0)]], np.float32), inputs["state"] + 1


class ScriptedModel:
    sample_rate = vad.SAMPLE_RATE

    def __init__(self, probs):
        self.session = ScriptedSession(probs)

    def stream(self, **kwargs):
        return vad.VadStream(self, **kwargs)


def test_speech_starts_and_ends_by_most_frames():
    # 6 frames of speech among 10 do not start speech, 7 do. It ends once
    # at most 3 of the last 10 are speech.
    probs = [0.9] * 6 + [0.1] * 4 + [0.9] * 7 + [0.1] * 4 + [0.1] * 10
    model = ScriptedModel(probs)
    stream = model.stream()
    heard = []
    for i in range(len(probs)):
        stream.push(np.zeros(vad.FRAME, np.int16))
        heard.append(stream.speech)
    assert not any(heard[:16])
    assert heard[16] and stream.heard_speech
    # The 4th frame of silence leaves 6 of speech, the 7th 3.
    assert heard[17:23] == [True] * 6
    assert heard[23] is False
    assert stream.silence_seconds == (len(probs) - 23) * vad.FRAME / vad.SAMPLE_RATE


def test_frames_carry_their_context_and_state():
    model = ScriptedModel([0.1] * 3)
    stream = model.stream()
    audio = np.arange(3 * vad.FRAME, dtype=np.int16)
    # Chunks that split frames give the same frames.
    assert len(stream.push(audio[:700])) == 1
    assert len(stream.push(audio[700:])) == 2
    first, second, third = model.session.inputs
    assert first["input"].shape == (1, vad.CONTEXT + vad.FRAME)
    np.testing.assert_array_equal(first["input"][0, : vad.CONTEXT], 0)
    np.testing.assert_allclose(
        second["input"][0, : vad.CONTEXT],
        audio[vad.FRAME - vad.CONTEXT : vad.FRAME] / 32768,
    )
    assert third["state"].max() == 2
    assert int(first["sr"]) == 16000


class FakeVoice:
    """Hears speech in each chunk that starts loud."""

    def __init__(self):
        self.speech = False

    def push(self, audio):
        self.speech = bool(len(audio)) and audio[0] > 1000


class FakeVad:
    sample_rate = 16000

    def stream(self):
        return FakeVoice()


class CountingDetector:
    def __init__(self, log):
        self.log = log
        self.score = 0.5

    def push(self, audio):
        self.log.append(len(audio))
        return False


class FakeModel:
    sample_rate = 16000


def test_gated_detector_runs_only_around_speech():
    log, made = [], []

    def new_detector():
        made.append(len(log))
        return CountingDetector(log)

    gated = wake_word.GatedDetector(
        FakeModel(), FakeVad(), hangover_seconds=0.25, new_detector=new_detector
    )
    quiet, loud = np.zeros(1600, np.int16), np.full(1600, 5000, np.int16)
    for _ in range(12):
        gated.push(quiet)
    assert not made
    # Speech starts: the detector takes the last second, then each chunk,
    # until the hangover after speech ended.
    gated.push(loud)
    assert made == [0] and log == [16000]
    gated.push(loud)
    for _ in range(5):
        gated.push(quiet)
    assert log == [16000, 1600, 1600, 1600, 1600]
    assert gated.detector is None
    # The next speech gets a new detector, and reset waits for speech again.
    gated.push(loud)
    assert len(made) == 2
    gated.reset()
    assert gated.detector is None
