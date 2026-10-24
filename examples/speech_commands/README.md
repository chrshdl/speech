The [Speech Commands] dataset by Pete Warden, version 0.02: 105,829 clips
of one second, each of one of 35 words said alone, such as "left",
"right", "yes", "no", "stop" and the digits, by 2,618 speakers on their
own microphones. The 3,750 hour configs in `examples/commonvoice` train on
it besides their sentences, so the model also learns words said on their
own, as for the game's commands.

## Setup

From the top level directory run

```
uv run examples/speech_commands/prepare.py examples/speech_commands/data
```

This streams the 2.4 GB archive, extracts the clips, about 2.4 GB of
16 kHz WAV, without the background noise, which `examples/noise` uses, and
writes `train.json`, `dev.json` and `test.json` with each clip's word as its
text. The dev and test sets are the official ones, so no speaker is in two
sets:

| Set | Clips | Hours |
|---|---|---|
| train | 84,843 | 23.1 |
| dev | 9,981 | 2.7 |
| test | 11,005 | 3.0 |

Each word has 1,557 to 4,052 clips, "left" 3,801 and "right" 3,778.

## Evaluate

`evaluate.py` measures how a model recognizes the test set's words, each
in half a second of silence, as a word within a stream:

```
uv run examples/speech_commands/evaluate.py model.onnx
```

`--threshold` sets the lowest score of a command, as `backend.py`'s
`--command-threshold`. Several, comma separated, compare from one pass over
the clips, such as `--threshold=-1.5,-1.25,-1.0` with an equals sign, as
the values start with a minus.

It prints how many clips are closest to their own of the 35 words, how
many are transcribed exactly as the word, and how many of the clips of the
games' commands, "left", "right", "up" and "down", the commands recognize,
as `backend.py` does, also of each, and of the other words it takes for
one. The tables below are from before "up" and "down" became commands, for
Snake, and count only "left" and "right", which they still recognize as
before, see `docs/backend-api.md`. For the 3,750 hour model after its
first three epochs, of which the third also trained on Speech Commands'
training clips, at the threshold of -1.5:

| | Epoch 0 | Epoch 1 | Epoch 2 |
|---|---|---|---|
| Closest of the words | 82.0% | 85.9% | 87.4% |
| Transcript is the word | 22.8% | 33.7% | 38.0% |
| Commands recognized | 603 of 808 | 694 of 808 | 701 of 808 |
| In the wrong direction | 1 | 1 | 0 |
| Other words taken for a command | 6 of 10,197 | 14 of 10,197 | 34 of 10,197 |

After epoch 2 the model took more than twice as many other words for a
command, so the default threshold became -1.25. For epoch 2's int8 ONNX
model, on the test set and on the kid's clips recorded with
`record.py`:

| Threshold | Commands recognized | Other words taken | Child's commands | Child's other words |
|---|---|---|---|---|
| -1.5 | 701 of 808 | 34 of 10,197 | 20 of 24 | 0 of 12 |
| -1.25 | 678 of 808 | 21 of 10,197 | 20 of 24 | 0 of 12 |
| -1.0 | 651 of 808 | 10 of 10,197 | 16 of 24 | 0 of 12 |
| -0.8 | 621 of 808 | 4 of 10,197 | 16 of 24 | 0 of 12 |
| -0.5 | 556 of 808 | 0 of 10,197 | 14 of 24 | 0 of 12 |

-1.25 takes a third fewer other words for a command, at no cost for the
child. A higher threshold costs the child's commands sooner than those of
the test set, whose speakers are mostly adults.

The model knows the words, but trained on sentences it rarely spells one
said alone right: "left" became "laft" or "lef", and only 16% of the
"left" clips were transcribed as "left".

## Record a speaker

`record.py` records someone, such as a child, saying the commands and
other words, to measure how a model recognizes that speaker:

```
uv run examples/speech_commands/record.py
```

It prompts 12 times each for "left" and "right" and 12 other words, such
as "yay" and "oh no", in random order. For each, press Enter, and the
word is said within 2.5 seconds; `r` records the last again. It writes the
clips and `clips.json` to `examples/speech_commands/data/recorded`, which
`evaluate.py --data` takes.

## License

Speech Commands is released under the [CC BY 4.0] license, so models
trained with it must credit the dataset:

> Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary
> Speech Recognition. arXiv:1804.03209.

[Speech Commands]: https://arxiv.org/abs/1804.03209
[CC BY 4.0]: https://creativecommons.org/licenses/by/4.0/
