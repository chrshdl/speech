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

It prints how many clips are closest to their own of the 35 words, how
many are transcribed exactly as the word, and how many of the "left" and
"right" clips the game's commands recognize, as `backend.py` does, and of
the other words it takes for one. For the 3,750 hour model after its
first epoch, which had not heard these clips:

| | Epoch 0 |
|---|---|
| Closest of the words | 82.0% |
| Transcript is the word | 22.8% |
| Commands recognized | 641 of 808 |
| In the wrong direction | 0 |
| Other words taken for a command | 20 of 10,197 |

The model knows the words, but trained on sentences it rarely spells one
said alone right: "left" became "laft" or "lef", and only 16% of the
"left" clips were transcribed as "left".

## License

Speech Commands is released under the [CC BY 4.0] license, so models
trained with it must credit the dataset:

> Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary
> Speech Recognition. arXiv:1804.03209.

[Speech Commands]: https://arxiv.org/abs/1804.03209
[CC BY 4.0]: https://creativecommons.org/licenses/by/4.0/
