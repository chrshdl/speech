import numpy as np
import shared
import torch

from speech import loader, streaming
from speech.models import CTC
from speech.models.ctc_decoder import PRUNE, BeamSearch, decode
from speech.models.word_lm import WordLM
from speech.utils import wave


def test_transcriber():
    torch.manual_seed(0)
    np.random.seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    config = shared.model_config | {
        "encoder": shared.model_config["encoder"] | {"lookahead": 2}
    }
    model = CTC(preproc.input_dim, preproc.vocab_size, config)
    shared.randomize_lookahead(model)
    model.set_eval()

    audio, sample_rate = wave.array_from_wave("test0.wav")
    assert streaming.model_sample_rate(preproc) == sample_rate

    # Feed 100 ms chunks, then flush.
    transcriber = streaming.Transcriber(model, preproc, sample_rate)
    for start in range(0, len(audio), 1600):
        transcriber.push(audio[start : start + 1600])
    transcriber.push(audio[:0], final=True)

    # Greedy decoding of the whole file gives the same labels.
    features = torch.from_numpy(preproc.preprocess("test0.wav", "")[0])
    with torch.no_grad():
        probs = model.forward_impl(features.unsqueeze(0), softmax=True)
    best = probs[0].argmax(dim=1).tolist()
    assert transcriber.labels() == model.max_decode(best, model.blank)

    # The pause counts the blank frames after the last label.
    trailing = len(best) - max(i for i, p in enumerate(best) if p != model.blank) - 1
    assert transcriber.pause_seconds == trailing * transcriber.frame_seconds
    assert transcriber.frame_seconds == 0.02

    transcriber.clear()
    assert transcriber.labels() == []
    assert transcriber.pause_seconds == 0


def test_transcriber_with_lm():
    torch.manual_seed(0)
    np.random.seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    model = CTC(preproc.input_dim, preproc.vocab_size, shared.model_config)
    model.set_eval()
    lm = WordLM(["hello world", "hello hi", "world"])
    scorer = lm.scorer(preproc.char_to_int)

    audio, sample_rate = wave.array_from_wave("test1.wav")
    search = BeamSearch(model.blank, 8, scorer, PRUNE)
    transcriber = streaming.Transcriber(model, preproc, sample_rate, search)
    for start in range(0, len(audio), 1600):
        transcriber.push(audio[start : start + 1600])
    transcriber.push(audio[:0], final=True)

    # Streaming the beam search gives the offline result.
    features = torch.from_numpy(preproc.preprocess("test1.wav", "")[0])
    with torch.no_grad():
        probs = model.forward_impl(features.unsqueeze(0), softmax=True)[0].numpy()
    expected = decode(probs, 8, model.blank, scorer, PRUNE)[0]
    assert transcriber.labels(final=True) == list(expected)

    # Clearing starts a new utterance.
    transcriber.clear()
    assert transcriber.labels() == []


def test_take_line():
    torch.manual_seed(0)
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    model = CTC(preproc.input_dim, preproc.vocab_size, shared.model_config)
    model.set_eval()
    audio, sample_rate = wave.array_from_wave("test0.wav")
    transcriber = streaming.Transcriber(model, preproc, sample_rate)

    # No line before any text.
    assert transcriber.take_line(0.0) is None
    transcriber.push(audio, final=True)
    text = transcriber.text().strip()
    assert text

    # A longer pause than heard keeps the line going.
    assert transcriber.take_line(transcriber.pause_seconds + 1) is None
    assert transcriber.take_line(transcriber.pause_seconds) == text
    assert transcriber.labels() == []
