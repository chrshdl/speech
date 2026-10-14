import argparse
import os

import numpy as np

import speech
from speech import export
from speech.onnx_model import OnnxModel
from speech.torch_model import TorchModel
from speech.utils.io import get_names


def check(model, preproc, path, seconds=10, chunk=17):
    """
    Streams random features through the exported model in chunks and
    returns the largest difference of its output probabilities from
    PyTorch's on the whole input.
    """
    rng = np.random.default_rng(0)
    frames = seconds * 100
    features = preproc.mean + preproc.std * rng.standard_normal(
        (frames, preproc.input_dim)
    )
    features = features.astype(np.float32)
    expected = TorchModel(model, preproc).probs(features)

    exported = OnnxModel(path)
    state, probs = None, []
    for start in range(0, frames, chunk):
        final = start + chunk >= frames
        p, state = exported.stream(features[start : start + chunk], state, final)
        probs.append(p)
    probs = np.concatenate(probs)
    assert probs.shape == expected.shape, (probs.shape, expected.shape)
    return np.abs(probs - expected).max()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a streaming CTC model to ONNX, to run it with "
        "stream.py or serve.py without PyTorch, such as on a Raspberry Pi."
    )
    parser.add_argument("model", help="A path to a stored model.")
    parser.add_argument(
        "--out",
        help="The .onnx file to write, by default the saved model's name with "
        "-int8.onnx or -fp32.onnx, such as best_model-int8.onnx.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Keep the GRU weights in float32 instead of int8, four times "
        "larger and slower but exactly the PyTorch model.",
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--last",
        action="store_true",
        help="Last saved model instead of best on dev set.",
    )
    which.add_argument(
        "--epoch",
        type=int,
        help="The model saved after this epoch, counted from 0 as train.py "
        "prints them, instead of the best on the dev set.",
    )
    args = parser.parse_args()

    tag = None if args.last or args.epoch is not None else "best"
    model, preproc = speech.load(args.model, tag=tag, epoch=args.epoch)
    model.set_eval()
    precision = "fp32" if args.fp32 else "int8"
    model_path = get_names(args.model, tag, args.epoch)[0]
    out = args.out or f"{model_path}-{precision}.onnx"
    export.export(model, preproc, out, int8=not args.fp32)
    print(f"Wrote {out}, {os.path.getsize(out) / 1e6:.1f} MB")
    diff = check(model, preproc, out)
    print(f"Largest difference of the output probabilities from PyTorch: {diff:.2g}")
