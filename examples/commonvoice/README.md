[Mozilla Common Voice] is crowd-sourced read speech recorded by volunteers on
their own microphones, with many accents. Version 26.0 of English has 2,785
validated hours from 100,172 speakers, as 88 GB of 48 kHz MP3, released under
CC0. Its range of speakers, microphones and rooms complements the clean
audiobooks of LibriSpeech for speech in the real world.

## Setup

Common Voice is distributed through the [Mozilla Data Collective]. Download the
latest Common Voice Scripted Speech release for English there, 26.0 from June
2026 or newer. Downloading
means agreeing not to try to identify the speakers. Extract it, then from the
top level directory run

```
uv run examples/commonvoice/preprocess.py <path_to_extracted>/en \
    examples/commonvoice/data
```

where `en` is the directory that holds `clips/` and the `.tsv` files. This
writes `train.json`, `dev.json` and `test.json` to `examples/commonvoice/data`,
with the clips converted to 16 kHz FLAC in `examples/commonvoice/data/clips`.
The conversion runs on every CPU core, and an interrupted run continues where
it stopped. The FLAC clips take about twice the space of the MP3s, so plan for
about 260 GB in total, or delete the MP3s afterwards.

The script

- normalizes each transcript to the characters of the models: lowercase
  letters, the apostrophe and the space. It removes accents and punctuation,
  turns hyphens into spaces, and keeps apostrophes inside words and after a
  plural s. Sentences with other characters, such as digits, are dropped
  rather than guessed at, and counted in the summary it prints.
- trains on all validated clips except those of the dev and test speakers and
  those of the dev and test sentences, which gives far more data than the
  official train split without leaking dev or test material into training.
  `--train-from train` uses the official train split instead.

## Train

The configs train a streaming CTC model of 31 million parameters, five
unidirectional GRU layers of 1,024 units with a 200 ms lookahead, on Common
Voice and all 960 hours of LibriSpeech, about 3,750 hours. They augment the
training audio as Mozilla's DeepSpeech 0.9 did: background noise, babble,
reverb, volume, pitch and tempo, plus SpecAugment. Besides Common Voice,
prepare LibriSpeech, about 60 GB, and the noise recordings:

```
uv run examples/librispeech/download.py examples/librispeech/data --sets \
    train-clean-100 train-clean-360 train-other-500 dev-clean test-clean
uv run examples/librispeech/preprocess.py examples/librispeech/data --sets \
    train-clean-100 train-clean-360 train-other-500 dev-clean test-clean
uv run examples/noise/download.py examples/noise/data
```

With Common Voice's FLAC clips, plan for about 250 GB of free space, or about
350 GB until its archive and MP3s are deleted.

Before a run that takes days, measure the speed on the machine with
`benchmark.py`. It times training steps on real batches for each batch size
and precision, and estimates the hours per epoch from the prepared training
set:

```
uv run benchmark.py examples/commonvoice/ctc_streaming_3750h_config.json \
    --batch-sizes 16,32,64 --precisions none,bf16
```

### On an Apple M4 Max

`ctc_streaming_3750h_config.json` uses bfloat16, batches of 32, 8 data loader
workers and 6 epochs. Keep the Mac awake while it trains:

```
caffeinate -i uv run train.py examples/commonvoice/ctc_streaming_3750h_config.json
```

On an 8 GB M1 MacBook Air the model trained at 40 seconds of audio per second
in bfloat16, which is 94 hours per epoch. The M4 Max has about five to six
times its GPU throughput, so expect roughly 15 to 20 hours per epoch and four
to five days for the run, and check it with `benchmark.py`. Training saves the
model after every epoch and continues after a stop with `--resume`. Each
epoch sees 37 times the audio of an epoch on `train-clean-100`, so a few
epochs go a long way.

### On an NVIDIA GPU

`ctc_streaming_3750h_gpu_config.json` is meant for a GPU with bfloat16, such
as an A100 or H100, with batches of 32, 16 data loader workers and 12 epochs:

```
uv run train.py examples/commonvoice/ctc_streaming_3750h_gpu_config.json
```

It has not been run yet, so measure its speed with `benchmark.py` before
committing to a full run.

[Mozilla Common Voice]: https://commonvoice.mozilla.org
[Mozilla Data Collective]: https://mozilladatacollective.com
