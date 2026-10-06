[LibriSpeech] is about 1,000 hours of read English audiobooks at 16 kHz,
released under CC BY 4.0. The streaming CTC models, the language model and the
decoding of this repository were developed on it.

## Setup

From the top level directory, download and index the subsets. The audio stays
FLAC and the archives are extracted while they download, so they never take
space on disk.

```
uv run examples/librispeech/download.py examples/librispeech/data \
    --sets train-clean-100 dev-clean test-clean
uv run examples/librispeech/preprocess.py examples/librispeech/data \
    --sets train-clean-100 dev-clean test-clean
```

This writes `train-clean-100.json`, `dev-clean.json` and `test-clean.json` to
`examples/librispeech/data/LibriSpeech`. Add `train-clean-360` and
`train-other-500` to both commands for all 960 hours.

## Train

```
uv run train.py examples/librispeech/ctc_streaming_100h_config.json
```

trains a streaming CTC model of 6.7 million parameters, four unidirectional GRU
layers of 512 units with a 200 ms lookahead, on `train-clean-100`, tuning on
`dev-clean`. It uses mild SpecAugment and halves the learning rate when the dev
loss stops improving. On an 8 GB M1 MacBook Air each epoch took 1.6 hours, so
the 15 epochs took about 25 hours. `--resume` continues a stopped run.

`ctc_streaming_config.json` and `ctc_streaming_specaug_config.json` are the
earlier experiments on 10 hours, with three GRU layers of 384 units. Their data
is `dev-clean` and `test-clean` without four speakers, 5338, 5694, 1221 and
5639, which form their dev set. As they train on audio from `test-clean`, they
have no test result.

To decode with the language model, train it as described in the top level
`README.md` and pass it with `--lm`:

```
uv run eval.py examples/librispeech/models/ctc_streaming_100h \
    examples/librispeech/data/LibriSpeech/test-clean.json \
    --lm examples/librispeech/models/lm-librispeech.npz
```

## Results

Greedy decoding without the LM, and the prefix beam search with a beam of 32
and the bigram LM trained on the LibriSpeech LM corpus, with the default
weight 0.5 and word bonus 4.

| Config | Training data | Dev CER | Dev WER | Test CER | Test WER |
|---|---|---|---|---|---|
| `ctc_streaming_config.json` | 10 h | 0.309 | 0.768 | | |
| with the LM | | 0.290 | 0.671 | | |
| `ctc_streaming_specaug_config.json` | 10 h | 0.326 | 0.799 | | |
| with the LM | | 0.310 | 0.707 | | |
| `ctc_streaming_100h_config.json` | `train-clean-100` | 0.208 | 0.577 | 0.202 | 0.569 |
| with the LM | | **0.190** | **0.480** | **0.184** | **0.470** |

The dev set of the 10 hour configs is their four held out speakers, so it is
not comparable with `dev-clean`. The test set is `test-clean`.

- Ten times the data lowered the CER by a third. The 10 hour model overfit,
  with its dev loss flat from epoch 14 while its training loss kept falling,
  but the dev loss of the 100 hour model fell in all 15 epochs, so the
  learning rate was never lowered and more epochs would help.
- SpecAugment with more dropout stopped the 10 hour model from overfitting but
  slowed its learning more than it helped within 30 epochs.
- The LM lowers the WER of the 100 hour model by 17%, more than the 13% of the
  10 hour model, since it helps most where the acoustic model spells a word
  almost right. Retuning the decoding on `dev-clean` for this model found its
  best WER at weight 0.3 and word bonus 2, only 0.001 below the defaults.

[LibriSpeech]: https://www.openslr.org/12
