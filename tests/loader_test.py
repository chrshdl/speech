import numpy as np

from speech import loader
from speech.utils import wave


def test_dataset():
    batch_size = 2
    data_json = "test.json"
    preproc = loader.Preprocessor(data_json)
    dataset = loader.AudioDataset(data_json, preproc, batch_size)

    # Num chars plus start and end tokens
    assert preproc.vocab_size == 11
    s_idx = preproc.vocab_size - 1
    assert preproc.int_to_char[s_idx] == preproc.START

    inputs, _ = dataset[0]

    # Inputs should be time x frequency
    assert inputs.shape[1] == preproc.input_dim
    assert inputs.dtype == np.float32

    # Correct number of examples
    assert len(dataset.data) == 8


def test_loader():

    batch_size = 2
    data_json = "test.json"
    preproc = loader.Preprocessor(data_json)
    ldr = loader.make_loader(data_json, preproc, batch_size, num_workers=0)

    # Test that batches are properly sorted by size
    for inputs, labels in ldr:
        assert inputs[0].shape == inputs[1].shape


def test_specgram_stream():
    audio, sample_rate = wave.array_from_wave("test0.wav")
    full = loader.log_specgram(audio, sample_rate)

    # Push random chunk sizes, including empty and tiny ones.
    rng = np.random.RandomState(0)
    stream = loader.SpecgramStream(sample_rate)
    frames = []
    start = 0
    while start < len(audio):
        end = start + rng.randint(0, 2000)
        frames.append(stream.push(audio[start:end]))
        start = end

    frames = np.concatenate(frames)
    assert frames.shape == full.shape
    # scipy computes in float32 for int16 audio, so the log of quiet bins
    # carries some rounding noise.
    assert np.allclose(frames, full, atol=1e-3)
