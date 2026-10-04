import numpy as np
import shared
import torch

import stream
from speech import loader
from speech.models import CTC
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
    assert stream.model_sample_rate(preproc) == sample_rate

    # Feed 100 ms chunks, then flush.
    transcriber = stream.Transcriber(model, preproc, sample_rate)
    for start in range(0, len(audio), 1600):
        transcriber.push(audio[start : start + 1600])
    transcriber.push(audio[:0], final=True)

    # Greedy decoding of the whole file gives the same labels.
    features = torch.from_numpy(preproc.preprocess("test0.wav", "")[0])
    with torch.no_grad():
        probs = model.forward_impl(features.unsqueeze(0), softmax=True)
    best = probs[0].argmax(dim=1).tolist()
    assert transcriber.labels == model.max_decode(best, model.blank)

    # The pause counts the blank frames after the last label.
    trailing = len(best) - max(i for i, p in enumerate(best) if p != model.blank) - 1
    assert transcriber.pause_seconds == trailing * transcriber.frame_seconds
    assert transcriber.frame_seconds == 0.02

    transcriber.clear()
    assert transcriber.labels == []
    assert transcriber.pause_seconds == 0
