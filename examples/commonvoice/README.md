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
    train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean
uv run examples/librispeech/preprocess.py examples/librispeech/data --sets \
    train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean
```

dev-other, the harder half of LibriSpeech's dev speakers, is a dev set of
the bigram config. A setup without it adds it with the same commands and
`--sets dev-other`.

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

### 5. Speech Commands

Download the clips of single words, such as "left" and "right", which the
configs also train on, see `examples/speech_commands/README.md`. It takes
about 2.5 GB and a few minutes:

```
uv run examples/speech_commands/prepare.py examples/speech_commands/data
```

### 6. Benchmark

Before a run that takes days, measure the speed on the machine with
`benchmark.py`. For each batch size and precision it times the training step
on clips from the shortest to the longest of the prepared training set, with
the accumulation that keeps the config's batches of 256. It estimates the
hours per epoch from the real batches, and reports the peak memory against
what the GPU holds and how fast the data loader keeps up:

```
uv run benchmark.py examples/commonvoice/ctc_streaming_3750h_config.json \
    --batch-sizes 32,64,128 --precisions none,bf16
```

If another batch size or precision is clearly faster than the config's and
fits in memory, change `batch_size`, `accumulate` or `mixed_precision` in the
config. Once an epoch has run, `--epoch-hours` scales the estimates to its
measured time.

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

`ctc_streaming_3750h_config.json` uses bfloat16, batches of 64, 8 data loader
workers and 6 epochs. Each optimizer step averages 4 batches, so the model
learns from batches of 256 clips, with gradients clipped at a norm of 400.
Adam's learning rate starts at 2e-4 and is divided by
1.2 after every epoch, as in Deep Speech 2, to 8.0e-5 in the last. Keep the
Mac awake while it trains:

```
caffeinate -i uv run train.py examples/commonvoice/ctc_streaming_3750h_config.json
```

On an 8 GB M1 MacBook Air the model trained at 40 seconds of audio per second
in bfloat16, which is 94 hours per epoch. On an M4 Max an epoch took 14 hours
in batches of 32, accumulated 8 times. In batches of 64, `benchmark.py`
estimates 11 hours, with a measured peak of 15 GB of the GPU's 32 GB working
set. Batches of 128 would save about another hour, but their longest batches,
of 30 second clips that tempo augmentation stretches by up to 11%, would need
more memory than the working set. Training saves the model after every epoch
and continues after a stop with `--resume`, also with another batch size that
keeps the batches of 256. Each epoch sees 37 times the audio of an epoch on
`train-clean-100`, so a few epochs go a long way.

### Speech Commands

Besides sentences, the configs train on the 85,000 clips of single words
of Speech Commands, 23 hours, so the model also learns words said on
their own, such as the commands of the backend's game. They are 5% of
the clips and 0.6% of the audio, so they add about 2,650 batches, 5%, to
each epoch. The dev set leaves them out, so the dev CER stays comparable
from epoch to epoch; `examples/speech_commands/evaluate.py` measures the
words.

A run that started without them can add them after any epoch, such as
after epoch 1, to train on them from epoch 2:

1. Let the epoch finish. Once the log shows `Epoch 2, augmentations: …`,
   the model and the training state of epoch 1 are saved.
2. Stop the run with Ctrl+C. Its progress into epoch 2 is lost.
3. Update the repository and download Speech Commands, see setup step 5.
4. Resume with the same config, which now lists Speech Commands:

   ```
   caffeinate -i uv run train.py examples/commonvoice/ctc_streaming_3750h_config.json --resume
   ```

From then on, an epoch has 53,171 batches rather than 50,520.

### More datasets

`examples/myst` prepares children's speech and `examples/voxpopuli`
speeches with many accents, about 200 and 530 hours once filtered. A run
can add them after any epoch the same way: prepare them, add their
`train.json` to the config's `train_set`, such as
`"examples/myst/data/train.json"`, and resume. Their dev sets stay out of
`dev_set`, so the dev CER stays comparable. Measure what they add on the
speech the model is for, such as a child's recordings with
`examples/speech_commands/evaluate.py --data`, not only on the dev set.

To check early that the model learns, watch `avg_loss` in the progress bar,
a running average of the loss per batch. In random order it does not depend
on where the epoch is, and it should fall steadily within the first half
hour. A first run that never learned stayed at about 210 to 230. With
SortaGrad the raw loss climbs with the clip length even while the model
improves, so `loss_per_char.py` prints that run's loss per character
instead.

### On an NVIDIA GPU

`ctc_streaming_3750h_gpu_config.json` is meant for a GPU with bfloat16, such
as an A100 or H100, with batches of 32 accumulated into 256, 16 data
loader workers and 12 epochs, over which the learning rate falls from 2e-4 to
2.7e-5:

```
uv run train.py examples/commonvoice/ctc_streaming_3750h_gpu_config.json
```

It has not been run yet, so measure its speed with `benchmark.py` before
committing to a full run.

### Bigrams

`ctc_streaming_3750h_bigrams_config.json` is the config with
`"bigrams" : true`, which outputs pairs of characters, so the transcripts
fit the model's 40 ms frames better, see "Bigram outputs" in the main
README. It also differs in:

- **Epochs.** It trains for 8 epochs rather than 6, over which the
  learning rate falls from 2e-4 to 5.6e-5, the same rate in each epoch as
  the run of characters.
- **VoxPopuli.** It also trains on the European Parliament speeches of
  VoxPopuli, with many accents and more spontaneous than audiobooks,
  which need preparing first, see `examples/voxpopuli/README.md`. The
  children's speech of MyST can join after any epoch, see "More datasets"
  above.
- **dev-other.** LibriSpeech dev-other, the closest public stand-in for
  hard speech, is a third dev set, see setup step 2.

Each dev set's loss, CER and WER are reported apart, such as
`dev_wer/librispeech/dev-clean` in TensorBoard. Compare those with the run
of characters epoch by epoch, as its `dev_cer` over all dev sets, which
selects the best model, has no dev-other. It saves to its own directory:

```
caffeinate -i uv run train.py examples/commonvoice/ctc_streaming_3750h_bigrams_config.json
```

The configs start at a learning rate of 2e-4, found with batches of 32,
where 5e-4 saturated the gates of the GRU. With batches of 256 a higher
rate may learn faster. Before a run, `trials.py` compares them: setups J,
K, L and M train the model of bigrams at 2e-4, 3e-4, 4e-4 and 6e-4 for 400
steps each, as the run trains, and report the dev CER and the share of
saturated gates every 100 steps. They train on
`examples/commonvoice/data/trial_train_set.json`, the random sample of the
training clips that `trial_noaug.py` writes:

```
uv run examples/commonvoice/trials.py --setups J,K,L,M
```

Each trains on 1,600 batches of 64, about 40 minutes at the speed of the
run of characters, 0.67 batches per second. Choose the highest rate whose
CER falls fastest while few of its gates saturate.

[Mozilla Common Voice]: https://commonvoice.mozilla.org
[Mozilla Data Collective]: https://mozilladatacollective.com
[uv]: https://docs.astral.sh/uv/
