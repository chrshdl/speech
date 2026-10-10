# Whisper

[Whisper](https://github.com/openai/whisper) recognizes speech far better
than the models trained here so far, also the names of places, so
`backend.py` can run it instead, until a streaming model trained here
recognizes as well. See the Whisper section of `docs/backend-api.md`.

Export a model to a single `.onnx` file, which needs Hugging Face
transformers in the version ONNX Runtime's exporter supports:

```
uv run --with transformers==4.48.3 --with onnxscript export_whisper.py base.en
```

This writes `models/whisper-base.en.onnx`, 216 MB, and checks it on
`tests/test0.wav`. `tiny.en` is faster but less accurate, 136 MB. The
file includes a prompt, by default "What's the weather in Hamburg?",
which makes Whisper more likely to write the words of the intents, such as
"weather" rather than "whether". Change it with `--prompt`.

Run the service with it, on the Pi with `--no-default-groups`, as the
file needs only ONNX Runtime:

```
uv run backend.py examples/whisper/models/whisper-base.en.onnx
```
