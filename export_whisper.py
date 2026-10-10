"""
Exports OpenAI's Whisper to a single .onnx file for backend.py, which runs
it with ONNX Runtime and speech.whisper_model, without PyTorch. Exporting
needs Hugging Face transformers in the version ONNX Runtime's exporter
supports:

    uv run --with transformers==4.48.3 --with onnxscript export_whisper.py base.en
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx

from speech import onnx_model, whisper_model
from speech.utils import wave

DEFAULT_PROMPT = "What's the weather in Hamburg?"


def export(name, out, prompt, int8=True):
    """
    Exports openai/whisper-<name>, such as base.en, to out, with the
    metadata speech.whisper_model needs.
    """
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    repo = f"openai/whisper-{name}"
    with tempfile.TemporaryDirectory() as tmp:
        # The encoder, the decoder and the search in one graph, which takes
        # the tokens to start the transcript with.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "onnxruntime.transformers.models.whisper.convert_to_onnx",
                "--model_name_or_path",
                repo,
                "--output",
                tmp,
                "--precision",
                "int8" if int8 else "fp32",
                "--provider",
                "cpu",
                "--use_forced_decoder_ids",
                # Hugging Face's cache rather than ./cache_models.
                "--cache_dir",
                HF_HUB_CACHE,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        exported = [f for f in os.listdir(tmp) if f.endswith("_beamsearch.onnx")]
        if not exported:
            # Its output is long and has harmless errors, such as of the
            # check it runs when the datasets package is installed, so it
            # is shown only on failure.
            sys.exit(f"Exporting {repo} failed:\n{result.stdout}{result.stderr}")
        model = onnx.load(os.path.join(tmp, exported[0]))

    tokenizer = WhisperTokenizer.from_pretrained(repo)
    features = WhisperFeatureExtractor.from_pretrained(repo)
    vocabulary = [""] * len(tokenizer)
    for token, i in tokenizer.get_vocab().items():
        vocabulary[i] = token

    def special(token):
        return tokenizer.convert_tokens_to_ids(token)

    prefix = []
    if prompt:
        prefix += [special("<|startofprev|>")]
        prefix += tokenizer.encode(" " + prompt.strip(), add_special_tokens=False)
    prefix += [special("<|startoftranscript|>")]
    if not name.endswith(".en"):
        prefix += [special("<|en|>"), special("<|transcribe|>")]
    prefix += [special("<|notimestamps|>")]
    metadata = {
        "model_type": "whisper",
        "format_version": whisper_model.FORMAT_VERSION,
        "name": name,
        "vocabulary": json.dumps(vocabulary),
        "end_of_text": special("<|endoftext|>"),
        "prefix": json.dumps(prefix),
        "n_mels": features.feature_size,
    }
    onnx.helper.set_model_props(model, {k: str(v) for k, v in metadata.items()})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    onnx.save(model, out)
    return features


def check(out, features, audio_file):
    """
    Checks that speech.whisper_model computes the features as Whisper's
    feature extractor does, and returns its transcript of the audio file.
    """
    whisper = whisper_model.WhisperModel(onnx_model.open_session(out))
    audio, sample_rate = wave.array_from_wave(audio_file)
    assert sample_rate == whisper.sample_rate, "The check needs 16 kHz audio."
    samples = audio.astype(np.float32) / 32768
    expected = features(samples, sampling_rate=sample_rate, return_tensors="np")
    computed = whisper_model.log_mel(samples, whisper.filters)
    difference = np.abs(expected.input_features[0] - computed).max()
    assert difference < 1e-3, f"The features differ by {difference}."
    return whisper.transcribe(audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Whisper to a .onnx file for backend.py."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="base.en",
        help="The Whisper model, such as tiny.en, base.en or small.en.",
    )
    parser.add_argument(
        "--out",
        help="The .onnx file to write, by default "
        "examples/whisper/models/whisper-<model>.onnx.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text like what is said, which makes Whisper more likely to write "
        "its words, such as 'weather' rather than 'whether'. Empty for none.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Keep the weights in float32 rather than int8.",
    )
    parser.add_argument(
        "--check-audio",
        default="tests/test0.wav",
        help="A 16 kHz audio file to transcribe after exporting, as a check.",
    )
    args = parser.parse_args()
    out = args.out or os.path.join(
        "examples", "whisper", "models", f"whisper-{args.model}.onnx"
    )
    features = export(args.model, out, args.prompt, int8=not args.fp32)
    print(f"Wrote {out}, {os.path.getsize(out) / 1e6:.0f} MB")
    print(f"{args.check_audio}: {check(out, features, args.check_audio)!r}")
