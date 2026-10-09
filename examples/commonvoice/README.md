[Mozilla Common Voice] is crowd-sourced read speech recorded by volunteers on
their own microphones, with many accents. Version 26.0 of English has 2,785
validated hours from 100,172 speakers, as 88 GB of 48 kHz MP3, released under
CC0. Its range of speakers, microphones and rooms complements the clean
audiobooks of LibriSpeech for speech in the real world.

## Setup

The configs train on Common Voice and all 960 hours of LibriSpeech, about
3,750 hours, with background noise for augmentation. Run these steps from the
top level directory. Plan for about 350 GB of free space while preparing the
data, and about 250 GB once the Common Voice archive and MP3s are deleted.

### 1. Install

Clone the repository and install its dependencies with [uv]:

```
git clone https://github.com/chrshdl/speech.git
cd speech
uv sync
```

### 2. LibriSpeech

Download and index all of LibriSpeech, about 60 GB. The archives are extracted
while they download, so they never take space on disk.

```
uv run examples/librispeech/download.py examples/librispeech/data --sets \
    train-clean-100 train-clean-360 train-other-500 dev-clean test-clean
uv run examples/librispeech/preprocess.py examples/librispeech/data --sets \
    train-clean-100 train-clean-360 train-other-500 dev-clean test-clean
```

### 3. Noise

Download the background noise recordings, which takes seconds:

```
uv run examples/noise/download.py examples/noise/data
```

### 4. Common Voice

Common Voice is distributed through the [Mozilla Data Collective]. Sign in
there, open Common Voice Scripted Speech for English, 26.0 from June 2026 or
newer, accept its terms and download it, about 88 GB. Downloading means
agreeing not to try to identify the speakers.

Extract the archive and find the directory that holds `clips/` and the `.tsv`
files:

```
mkdir -p ~/cv
tar -xzf <downloaded archive> -C ~/cv
find ~/cv -name validated.tsv
```

Then convert it, with the directory `find` printed, which ends in `/en`:

```
uv run examples/commonvoice/preprocess.py <directory with validated.tsv> \
    examples/commonvoice/data
```

This writes `train.json`, `dev.json` and `test.json` to
`examples/commonvoice/data`, with the clips converted to 16 kHz FLAC in
`examples/commonvoice/data/clips`. The conversion runs on every CPU core, and
an interrupted run continues where it stopped when run again. At the end it
prints the clips and hours of each split. Once they look right, delete the
archive and `~/cv` to free about 90 to 180 GB.

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

### 5. Benchmark

Before a run that takes days, measure the speed on the machine with
`benchmark.py`. It times training steps on real batches for each batch size
and precision, and estimates the hours per epoch from the prepared training
set:

```
uv run benchmark.py examples/commonvoice/ctc_streaming_3750h_config.json \
    --batch-sizes 16,32,64 --precisions none,bf16
```

If another batch size or precision is clearly faster than the config's, change
`batch_size` or `mixed_precision` in the config.

## Train

The configs train a streaming CTC model of 31 million parameters, five
unidirectional GRU layers of 1,024 units with a 200 ms lookahead. Like Deep
Speech 2, it uses batch normalization, a clipped ReLU and large batches,
without which a model this deep stopped learning in its first epoch. Unlike
Deep Speech 2, it trains in random order from the start, without SortaGrad,
since the shortest Common Voice clips do not let a new model learn. See
"Training deep models" in the main README.

The configs augment the training audio as Mozilla's DeepSpeech 0.9 did:
background noise, babble, reverb, volume, pitch and tempo, plus SpecAugment.
DeepSpeech added noise to 90% of the clips, while Deep Speech 2 found that "too
much noise augmentation tends to make optimization difficult" and added it to
40%. Here noise goes to 30% and babble to 10%, so 37% of the clips get at least
one. The noise, babble and reverb start with the second epoch, as
`"from_epoch" : 1`, since a CTC model first has to learn which frames belong
to which characters, which noise makes harder. One epoch over 3,750 hours
should be enough for that.

### On an Apple M4 Max

`ctc_streaming_3750h_config.json` uses bfloat16, batches of 32, 8 data loader
workers and 6 epochs. Each optimizer step averages 8 batches, so the model
learns from batches of 256 clips, with gradients clipped at a norm of 400.
Adam's learning rate starts at 2e-4 and is divided by
1.2 after every epoch, as in Deep Speech 2, to 8.0e-5 in the last. Keep the
Mac awake while it trains:

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

To check early that the model learns, watch `avg_loss` in the progress bar,
a running average of the loss per batch. In random order it does not depend
on where the epoch is, and it should fall steadily within the first half
hour. A first run that never learned stayed at about 210 to 230. With
SortaGrad the raw loss climbs with the clip length even while the model
improves, so `loss_per_char.py` prints that run's loss per character
instead.

### On an NVIDIA GPU

`ctc_streaming_3750h_gpu_config.json` is meant for a GPU with bfloat16, such
as an A100 or H100, with batches of 32 accumulated into 256 as above, 16 data
loader workers and 12 epochs, over which the learning rate falls from 2e-4 to
2.7e-5:

```
uv run train.py examples/commonvoice/ctc_streaming_3750h_gpu_config.json
```

It has not been run yet, so measure its speed with `benchmark.py` before
committing to a full run.

[Mozilla Common Voice]: https://commonvoice.mozilla.org
[Mozilla Data Collective]: https://mozilladatacollective.com
[uv]: https://docs.astral.sh/uv/
