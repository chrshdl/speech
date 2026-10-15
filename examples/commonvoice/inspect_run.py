"""
Shows whether a training run learns: its training loss over the logged
batches, in steps of 5%, and the greedy output of one of its saved models
on a few dev clips with the share of blank frames. A run that never
learned keeps its loss flat at about 210 to 230 and outputs blanks on
nearly every frame. Trains nothing. Run it from the repository root:

    uv run examples/commonvoice/inspect_run.py \\
        examples/commonvoice/models/ctc_streaming_3750h_stride3 --epoch 0
"""

import argparse

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import speech
from speech.models.ctc_decoder import best_path
from speech.utils.dataset import read_data_json

DEV = "examples/librispeech/data/LibriSpeech/dev-clean.json"


def loss_curve(path, parts=20):
    """The mean and lowest training loss of each of parts runs of batches."""
    events = EventAccumulator(path, size_guidance={"scalars": 0})
    events.Reload()
    loss = np.array([e.value for e in events.Scalars("train_loss")])
    size = max(1, len(loss) // parts)
    return [
        (start, loss[start : start + size].mean(), loss[start : start + size].min())
        for start in range(0, len(loss), size)
    ]


def greedy_outputs(model, preproc, dataset, clips):
    """The transcript, greedy output and share of blank frames of each clip."""
    model.set_eval()
    results = []
    for d in read_data_json(dataset)[:clips]:
        x = preproc.preprocess(d["audio"], d["text"])[0]
        with torch.no_grad():
            probs = model.forward_impl(torch.from_numpy(x).unsqueeze(0), softmax=True)
        top = probs[0].argmax(1).cpu().numpy()
        hyp = "".join(preproc.decode(best_path(top, model.blank)))
        results.append((d["text"], hyp, float(np.mean(top == model.blank))))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="A directory that train.py saved to.")
    parser.add_argument(
        "--epoch",
        type=int,
        help="The model saved after this epoch, counted from 0, instead of "
        "the best on the dev set.",
    )
    parser.add_argument("--dataset", default=DEV, help="The clips to transcribe.")
    parser.add_argument("--clips", type=int, default=6)
    args = parser.parse_args()

    curve = loss_curve(args.model)
    print("Training loss, mean and lowest of each 5% of the logged batches:")
    for start, mean, low in curve:
        print(f"  from batch {start:>7}: {mean:6.1f}  (lowest {low:6.1f})")

    tag = None if args.epoch is not None else "best"
    model, preproc = speech.load(args.model, tag=tag, epoch=args.epoch)
    print(f"\n{len(preproc.char_to_int)} labels. Greedy output:")
    for ref, hyp, blank in greedy_outputs(model, preproc, args.dataset, args.clips):
        print(f"  {blank:.0%} of the frames blank")
        print(f"  REF {ref}\n  HYP {hyp}")


if __name__ == "__main__":
    main()
