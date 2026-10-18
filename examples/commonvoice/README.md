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

### Stride 3

`ctc_streaming_3750h_stride3_config.json` strides 3 over time instead of
4, a frame every 30 ms, so fast speech has frames to spare for its
characters, see "Frames per character" in the main README. It also differs
in:

- **Epochs.** It trains for 8 epochs rather than 6, over which the
  learning rate falls from 2e-4 to 5.6e-5, the same rate in each epoch as
  the run at a stride of 4.
- **VoxPopuli.** It also trains on the European Parliament speeches of
  VoxPopuli, with many accents and more spontaneous than audiobooks,
  which need preparing first, see `examples/voxpopuli/README.md`. The
  children's speech of MyST can join after any epoch, see "More datasets"
  above.
- **dev-other.** LibriSpeech dev-other, the closest public stand-in for
  hard speech, is a third dev set, see setup step 2.

The GRU computes a third more frames, so measure the hours per epoch with
`benchmark.py` first. Each dev set's loss, CER and WER are reported apart,
such as `dev_wer/librispeech/dev-clean` in TensorBoard. Compare those with
the run at a stride of 4 epoch by epoch, as its `dev_cer` over all dev
sets, which selects the best model, has no dev-other. It saves to its own
directory:

```
uv run benchmark.py examples/commonvoice/ctc_streaming_3750h_stride3_config.json
caffeinate -i uv run train.py examples/commonvoice/ctc_streaming_3750h_stride3_config.json
```

`examples/commonvoice/inspect_run.py` shows whether a run learns, from
its training loss and the output of a saved model on a few dev clips:

```
uv run examples/commonvoice/inspect_run.py \
    examples/commonvoice/models/ctc_streaming_3750h_stride3 --epoch 0
```

`examples/commonvoice/data_report.py` compares a run's epochs on each dev
set, also on SLURP's requests to a voice assistant, which are what the
device hears, written by `examples/slurp/prepare.py`. Its `score` decodes a
random sample of each training source with a saved model and compares the
sources: loss per character, CER, speaking rate, durations, and the clips
whose transcript cannot fit the model's output frames. It writes
`report.html` with the worst clips of each source to listen to, which tells
mislabeled data from hard speech:

```
uv run examples/slurp/prepare.py
uv run examples/commonvoice/data_report.py eval \
    examples/commonvoice/models/ctc_streaming_3750h_stride3 --epochs 0,1,2 \
    --sets examples/slurp/data/slurp-test-headset.json \
    examples/slurp/data/slurp-test-far.json
uv run examples/commonvoice/data_report.py score \
    examples/commonvoice/models/ctc_streaming_3750h_stride3 --epoch 2
```

Its `clean` scores every clip of one source in batches and writes a page
of clips from each band of the highest losses, to hear where the
transcripts stop matching the audio, as VoxPopuli's sometimes do: they are
the Parliament's edited record, such as "returning financial markets" for
"financial markets are stabilizing". Given that band's lowest loss, it
writes the source without those clips and the ones that cannot fit:

```
uv run examples/commonvoice/data_report.py clean \
    examples/commonvoice/models/ctc_streaming_3750h_stride3 --epoch 2 \
    examples/voxpopuli/data/train.json
uv run examples/commonvoice/data_report.py clean \
    examples/commonvoice/models/ctc_streaming_3750h_stride3 --epoch 2 \
    examples/voxpopuli/data/train.json --max-loss 4
```

With 15 of 300 SLURP clips given each other's transcripts, the stride 3
model after epoch 2 ranked 13 of them among the 21 highest losses, and the
other 2 could not fit; `--max-loss 4` dropped all 15 and 9 of the other
285, which it found hard.

`examples/commonvoice/whisper_check.py` checks the transcripts with
Whisper, trained on other data than these models, for a person to review
the clips whose transcript Whisper does not hear, such as the many Speech
Commands clips whose word is cut off. Its `check` transcribes each clip
once, on the GPU, and keeps the result next to the dataset. Then training
pauses after each epoch for the review, given the checked datasets in the
config's data:

```
uv run --with transformers --with accelerate \
    examples/commonvoice/whisper_check.py check \
    examples/speech_commands/data/train.json examples/voxpopuli/data/train.json
```

```json
"review": {
    "sets": ["examples/speech_commands/data/train.json",
             "examples/voxpopuli/data/train.json"]
}
```

After each epoch but the last, `train.py` serves the clips whose CER
against Whisper's transcript is over 0.15 (`"min_cer"`) at
http://127.0.0.1:8765/ (`"port"`), one at a time, playing each: `O` keeps
it, Whisper misheard it, and `N` leaves it out of training, Whisper is
right. Each verdict is saved at once to `<dataset>-review.json`, so only
new clips come up in a later review, and the next epoch starts once
Continue is pressed, without the clips reviewed NOK. `whisper_check.py
review` serves the same review without training, and `write` writes a
dataset without the NOK clips.

#### Reviewing a run, step by step

1. Check the dataset with Whisper, and let it finish before training, as
   both use the GPU. It prints how many clips fall in each band of CER at
   the end:

   ```
   uv run --with transformers --with accelerate \
       examples/commonvoice/whisper_check.py check \
       examples/speech_commands/data/train.json
   ```

   Whisper pads every clip to 30 seconds, so a one-second clip costs as
   much as a 30-second one: Speech Commands' 85,000 clips took 6 to 7
   hours on an Apple M4 Max at the default `--batch-size 16`, and
   VoxPopuli's 181,000 clips take about twice as long, a job for a night.
   A larger `--batch-size`, such as 64, may be faster. Stopping loses at
   most the batch in progress: a run started again skips the checked
   clips.

2. Add the checked datasets to the config's data, such as after
   `"num_workers"`, keeping the comma before the next key:

   ```json
   "review" : {
       "sets" : ["examples/speech_commands/data/train.json"]
   },
   ```

   `train.py` reads it when it starts, so a run already training needs a
   restart with `--resume` to pause for the review; stop it after an
   epoch, as stopping within one loses that epoch.

3. Review first, without training, so the next epoch already trains
   without the NOK clips rather than with them for a whole epoch:

   ```
   uv run examples/commonvoice/whisper_check.py review \
       examples/speech_commands/data/train.json
   ```

   and open http://127.0.0.1:8765/. The queue need not be finished: press
   Continue whenever training should start. The verdicts are kept, and a
   later review shows only the clips not judged yet.

4. Continue the run:

   ```
   caffeinate -i uv run train.py \
       examples/commonvoice/ctc_streaming_3750h_stride3_config.json --resume
   ```

   It prints "Leaving out N clips reviewed as NOK." when it starts, and
   after each epoch it pauses at the same page for the clips not judged
   yet, until Continue is pressed.

### Bigram trials

A model of bigrams at a stride of 4, the config at a stride of 4 with
`"bigrams" : true` and the data of the stride 3 config, did not learn: after
its first epoch its training loss was still about 200, as in its first
steps, and it gave "th s the the the" for every dev clip, a CER of 0.76,
where the run of characters reached 0.28. Deep Speech 2 trained bigrams
the same way as characters, but chose each model's learning rate from
1e-4 to 6e-4 for the fastest convergence. `trials.py` compares setups of
the model of bigrams, trained as the run trained, and reports the dev CER
and the share of saturated gates every 100 steps:

| Setup | What | CER on Common Voice at step 800 | Saturated gates |
|---|---|---|---|
| J, K, L, M | Learning rates of 2e-4, 3e-4, 4e-4 and 6e-4 | 0.83, 0.81, 0.80, 0.81 | 43%, 53%, 53%, 60% |
| N | 2e-4, with a CTC loss of characters on a second output layer added to the loss of the bigrams, which only training uses | 0.78 | 34% |
| O | Deep Speech 2's optimizer: SGD with Nesterov momentum of 0.99, at 3e-4 | 0.78 | 24% |

They train on `examples/commonvoice/data/trial_train_set.json`, the random
sample of 96,000 training clips that `trial_noaug.py` writes, for 800
steps of 256 clips, which take the sample about twice, as 375 steps fit in
it. Each takes about 80 minutes on an M4 Max:

```
uv run examples/commonvoice/trials.py --setups J,K,L,M,N,O --steps 800
```

None started: the loss of each stopped falling after about 300 steps,
where the run of characters had lost 27% of its loss between steps 200
and 800, and higher learning rates saturated more of the gates. SGD
saturated the fewest. The trials had no setup of characters, though, which
in an earlier trial in batches of 32 also reached a CER of 0.77 in 400
steps.

[Mozilla Common Voice]: https://commonvoice.mozilla.org
[Mozilla Data Collective]: https://mozilladatacollective.com
[uv]: https://docs.astral.sh/uv/
