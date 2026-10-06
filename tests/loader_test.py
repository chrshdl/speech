import math

import numpy as np
import pytest
import soundfile
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


def test_volume():
    audio, _ = wave.array_from_wave("test0.wav")

    # A quiet level sets the peak, a loud one clips at full scale.
    quiet = loader.volume(audio, dbfs=[-6, -6])
    assert np.abs(quiet).max() == pytest.approx(32768 * 10 ** (-6 / 20), rel=1e-4)
    loud = loader.volume(audio, dbfs=[6, 6])
    clipped = (loud >= 32767) | (loud <= -32768)
    assert clipped.any()
    gain = 32768 * 10 ** (6 / 20) / np.abs(audio).max()
    assert np.allclose(loud[~clipped], audio[~clipped] * gain, rtol=1e-5)

    assert loader.volume(audio, dbfs=[6, 6], p=0) is audio
    silent = np.zeros(100, dtype=np.int16)
    assert loader.volume(silent, dbfs=[6, 6]) is silent


def test_pitch():
    # A smooth bump around bin 80, like a formant.
    bins = np.arange(161)
    features = np.tile(np.exp(-(((bins - 80) / 3) ** 2)), (10, 1)).astype(np.float32)

    # A lower pitch moves the peak down and fills the top with silence.
    lower = loader.pitch(features, factor=[0.5, 0.5])
    assert lower.shape == features.shape
    assert abs(lower[0].argmax() - 40) <= 1
    assert (lower[:, 81:] == features.min()).all()

    # A higher pitch moves it up.
    assert abs(loader.pitch(features, factor=[1.5, 1.5])[0].argmax() - 120) <= 1
    assert np.array_equal(loader.pitch(features, factor=[1, 1]), features)
    assert loader.pitch(features, factor=[0.5, 0.5], p=0) is features


def test_tempo():
    features = np.random.RandomState(0).randn(100, 161).astype(np.float32)
    assert loader.tempo(features, factor=[2, 2]).shape == (50, 161)
    assert loader.tempo(features, factor=[0.5, 0.5]).shape == (200, 161)
    assert np.array_equal(loader.tempo(features, factor=[1, 1]), features)
    assert loader.tempo(features, factor=[2, 2], p=0) is features


def test_augmented_dataset():
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    augment = {
        "noise": [
            {"source": ".", "snr": [5, 15], "p": 0.5},
            {"source": "test.json", "snr": [5, 15], "layers": [2, 3], "p": 0.5},
        ],
        "reverb": {"delay": [2, 18], "decay": [0.55, 0.85], "p": 0.5},
        "volume": {"dbfs": [-13, 7], "p": 0.5},
        "pitch": {"factor": [0.9, 1.1]},
        "tempo": {"factor": [0.9, 1.1]},
        "spec_augment": {
            "freq_masks": 2,
            "freq_width": 15,
            "time_masks": 2,
            "time_width": 10,
        },
    }
    plain = loader.AudioDataset("test.json", preproc, 2)
    augmented = loader.AudioDataset("test.json", preproc, 2, augment)

    torch.manual_seed(0)
    inputs, targets = augmented[0]
    clean_inputs, clean_targets = plain[0]
    assert targets == clean_targets
    assert inputs.shape[1] == clean_inputs.shape[1]
    assert 0.85 * len(clean_inputs) <= len(inputs) <= 1.15 * len(clean_inputs)

    # The same seed repeats the augmentation, another seed changes it.
    torch.manual_seed(0)
    assert np.array_equal(augmented[0][0], inputs)
    torch.manual_seed(1)
    other = augmented[0][0]
    assert other.shape != inputs.shape or not np.array_equal(other, inputs)


def deepspeech_reverb(audio, delay, decay, rate):
    # DeepSpeech 0.9.3's reverb loop, without its final normalization.
    decay = 10 ** (-decay / 20)
    result = np.copy(audio)
    primes = [17, 19, 23, 29, 31]
    for delay_prime in primes:
        layer = np.copy(audio)
        n_delay = math.floor(delay * (delay_prime / primes[0]) * rate / 1000.0)
        n_delay = max(16, n_delay)
        for w_index in range(math.floor(len(audio) / n_delay)):
            w1 = w_index * n_delay
            w2 = (w_index + 1) * n_delay
            width = min(len(audio) - w2, n_delay)
            layer[w2 : w2 + width] += decay * layer[w1 : w1 + width]
        result += layer
    return result


def test_reverb():
    audio, rate = wave.array_from_wave("test0.wav")
    audio = audio.astype(np.float64)
    for delay, decay in [(2.0, 0.55), (10.0, 0.7), (18.0, 0.85)]:
        ours = loader.reverb(audio, delay=[delay, delay], decay=[decay, decay])
        theirs = deepspeech_reverb(audio, delay, decay, rate)
        theirs *= np.abs(audio).max() / np.abs(theirs).max()
        assert np.allclose(ours, theirs, rtol=1e-4, atol=1e-2)
        # The peak level is kept.
        assert np.abs(ours).max() == pytest.approx(np.abs(audio).max())
    assert loader.reverb(audio, delay=[10, 10], decay=[0.7, 0.7], p=0) is audio


def test_noise(tmp_path):
    audio, _ = wave.array_from_wave("test0.wav")
    audio = audio.astype(np.float32)

    # White noise at another sample rate and shorter than the audio.
    rng = np.random.RandomState(0)
    soundfile.write(tmp_path / "white.wav", rng.uniform(-0.5, 0.5, 4000), 8000)
    torch.manual_seed(0)
    noisy = loader.noise(audio, source=str(tmp_path), snr=[10, 10])
    assert noisy.shape == audio.shape
    assert np.abs(noisy).max() == pytest.approx(np.abs(audio).max())

    # Undo the peak scaling, then measure the added noise against the audio.
    scale = np.dot(noisy, audio) / np.dot(audio, audio)
    added = noisy / scale - audio
    snr = 10 * np.log10(np.mean(audio**2) / np.mean(added**2))
    assert snr == pytest.approx(10, abs=0.3)

    # Babble mixes several recordings from a dataset.
    babble = loader.noise(audio, source="test.json", snr=[5, 5], layers=[3, 3])
    assert babble.shape == audio.shape
    assert not np.array_equal(babble, audio)

    assert loader.noise(audio, source="test.json", snr=[5, 5], p=0) is audio
    silent = np.zeros(100, dtype=np.float32)
    assert loader.noise(silent, source="test.json", snr=[5, 5]) is silent
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        loader.NoiseSource(str(tmp_path / "empty"))


def test_from_epoch():
    preproc = loader.Preprocessor("test.json", start_and_end=False)
    volume = {"dbfs": [-6, -6]}
    augment = {
        "noise": [
            {"source": ".", "snr": [5, 5]},
            {"source": "test.json", "snr": [5, 5], "layers": [2, 2], "from_epoch": 1},
        ],
        "reverb": {"delay": [10, 10], "decay": [0.7, 0.7], "from_epoch": 2},
        "volume": volume,
    }
    dataset = loader.AudioDataset("test.json", preproc, 2, augment)
    early = loader.AudioDataset(
        "test.json", preproc, 2, {"noise": augment["noise"][:1], "volume": volume}
    )

    def item(ds):
        torch.manual_seed(0)
        return ds[0][0]

    # Before their from_epoch, augmentations are skipped and draw no
    # random numbers, so the others augment exactly as without them.
    assert list(dataset.active) == ["noise", "volume"]
    assert len(dataset.active["noise"]) == 1
    assert np.array_equal(item(dataset), item(early))

    dataset.set_epoch(1)
    assert len(dataset.active["noise"]) == 2
    assert "from_epoch" not in dataset.active["noise"][1]
    assert not np.array_equal(item(dataset), item(early))

    dataset.set_epoch(2)
    assert list(dataset.active) == ["noise", "reverb", "volume"]
    assert "from_epoch" not in dataset.active["reverb"]
    item(dataset)
