import numpy as np
import torch

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


def test_spec_augment():
    rng = np.random.RandomState(0)
    features = rng.randn(200, 161).astype(np.float32) + 5
    args = {
        "freq_masks": 2,
        "freq_width": 27,
        "time_masks": 2,
        "time_width": 40,
        "time_ratio": 0.1,
    }

    torch.manual_seed(0)
    masked = loader.spec_augment(features, **args)
    assert masked.shape == features.shape
    assert not np.shares_memory(masked, features)

    # Masked bands and spans are zero, everything else is unchanged.
    zero_bins = np.all(masked == 0, axis=0)
    zero_frames = np.all(masked == 0, axis=1)
    assert zero_bins.sum() <= 2 * 27
    # Time spans are capped at 10% of the 200 frames.
    assert zero_frames.sum() <= 2 * 20
    kept = ~zero_bins[None, :] & ~zero_frames[:, None]
    assert np.array_equal(masked[kept], features[kept])
    assert (masked[~kept] == 0).all()

    # The masks come from torch's generator.
    torch.manual_seed(0)
    assert np.array_equal(loader.spec_augment(features, **args), masked)

    # Without masks nothing changes.
    none = dict(args, freq_masks=0, time_masks=0)
    assert np.array_equal(loader.spec_augment(features, **none), features)
