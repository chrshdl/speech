# speech

Speech is an open-source package to build end-to-end models for automatic
speech recognition. Sequence-to-sequence models with attention,
Connectionist Temporal Classification and the RNN Sequence Transducer
are currently supported.

The goal of this software is to facilitate research in end-to-end models for
speech recognition. The models are implemented in PyTorch.

The software requires Python 3.13 or later.

## Install

Install [uv](https://docs.astral.sh/uv/), then from the top level directory run:

```
uv sync
```

This creates a virtual environment with PyTorch and the other requirements and
installs the `speech` package into it. If you need a PyTorch build for a
specific CUDA version, see the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

You can verify the install was successful by running the tests.

```
uv run pytest
```

## Run 

To train a model run
```
uv run train.py <path_to_config>
```

After the model is done training you can evaluate it with

```
uv run eval.py <path_to_model> <path_to_data_json>
```

To see the available options for each script use `-h`: 

```
uv run {train, eval}.py -h
```

## Streaming

A bidirectional encoder needs the whole utterance before it can output
anything. For streaming recognition, use a unidirectional RNN followed by a
lookahead convolution as in [Deep Speech 2]. Set `"bidirectional" : false` and
`"lookahead" : <frames>` in the model's `encoder` config. The lookahead is
counted in encoder frames, after the stride of the convolutional front end. For
example, with 10 ms input frames and a total stride of 2, a lookahead of 10 sees
200 ms of future audio. See `examples/timit/ctc_streaming_config.json`.

To encode audio as it arrives, call `model.encode_stream` with each chunk of
input frames and the state it returned on the previous call. Pass `final=True`
with the last chunk. The streamed outputs match `model.encode` on the full input.

To transcribe live from the microphone with a trained CTC model run

```
uv run stream.py <path_to_model> --mic
```

The transcript grows as you speak, and a pause starts a new line. Stop with
Ctrl+C. Use `--mic-device` to pick a microphone from `uv run python -m
sounddevice`. Given audio files or a dataset json instead of `--mic`,
`stream.py` feeds the audio chunk by chunk and checks the result against
decoding the whole file at once.

### Language model

CTC models can decode with a word bigram language model in the prefix beam
search, both when streaming and in `eval.py`. Train one from the text of a
dataset, then pass it with `--lm`:

```
uv run python -m speech.models.word_lm examples/librispeech/data/train.json lm.json
uv run stream.py <path_to_model> --mic --lm lm.json
uv run eval.py <path_to_model> <path_to_data_json> --lm lm.json
```

A word is scored once it is complete, as `lm_weight * log P(word | previous
word) + word_bonus`. While streaming, the transcript can change as more audio
arrives, and a pause scores the end of the sentence before starting a new
line. The defaults for `--lm-weight`, `--word-bonus` and `--beam-size` were
tuned on the LibriSpeech dev set, where the LM lowered the WER of the
streaming CTC model from 0.80 to 0.74.

[Deep Speech 2]: https://arxiv.org/abs/1512.02595

## Examples

For examples of model configurations and datasets, visit the examples
directory. Each example dataset should have instructions and/or scripts for
downloading and preparing the data. There should also be one or more model
configurations available. The results for each configuration will documented in
each examples corresponding `README.md`.
