[Mozilla Common Voice] is crowd-sourced read speech recorded by volunteers on
their own microphones, with many accents. Version 26.0 of English has 2,785
validated hours from 100,172 speakers, as 88 GB of 48 kHz MP3, released under
CC0. Its range of speakers, microphones and rooms complements the clean
audiobooks of LibriSpeech for speech in the real world.

## Setup

Common Voice is distributed through the [Mozilla Data Collective]. Download the
latest Common Voice Scripted Speech release for English there. Downloading
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

`ctc_streaming_config.json` trains a streaming CTC model of 31 million
parameters on Common Voice and all 960 hours of LibriSpeech. Prepare
LibriSpeech, about 60 GB, with

```
uv run examples/librispeech/download.py examples/librispeech/data --sets \
    train-clean-100 train-clean-360 train-other-500 dev-clean test-clean
uv run examples/librispeech/preprocess.py examples/librispeech/data --sets \
    train-clean-100 train-clean-360 train-other-500 dev-clean test-clean
```

then train with

```
uv run train.py examples/commonvoice/ctc_streaming_config.json
```

The config is meant for an NVIDIA GPU with bfloat16, such as an A100 or H100,
and has not been run yet. Measure its speed for an hour on the GPU you plan to
use before committing to a full run.

[Mozilla Common Voice]: https://commonvoice.mozilla.org
[Mozilla Data Collective]: https://mozilladatacollective.com
