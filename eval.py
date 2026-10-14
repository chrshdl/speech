import argparse
import json

import tqdm

import speech
from speech.features import log_specgram
from speech.models.ctc_decoder import BEAM_SIZE, PRUNE, TOP_K, decode
from speech.models.word_lm import LM_WEIGHT, UNK_PENALTY, WORD_BONUS, WordLM
from speech.utils import wave
from speech.utils.dataset import read_data_json


def eval_loop(model, ldr, decoder_args):
    import torch

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            preds = model.infer(batch, **decoder_args)
            all_preds.extend(preds)
            all_labels.extend(batch[1])
    return list(zip(all_labels, all_preds))


def eval_onnx(model, dataset_json, decoder_args):
    """
    Decodes each example with an OnnxModel as CTC.infer does, and returns
    (label, prediction) pairs of character lists.
    """
    beam_size = decoder_args.get("beam_size", 1)
    lm = decoder_args.get("lm")
    prune = decoder_args.get("prune")
    top_k = decoder_args.get("top_k")
    results = []
    for datum in tqdm.tqdm(read_data_json(dataset_json)):
        audio, sample_rate = wave.array_from_wave(datum["audio"])
        if sample_rate != model.sample_rate:
            raise ValueError(
                f"{datum['audio']} is {sample_rate} Hz, the model needs "
                f"{model.sample_rate} Hz."
            )
        probs = model.probs(log_specgram(audio, sample_rate))
        labels = decode(probs, beam_size, model.blank, lm, prune, top_k)[0]
        results.append((list(datum["text"]), list(model.decode(labels))))
    return results


def run(
    model_path,
    dataset_json,
    batch_size=8,
    tag="best",
    out_file=None,
    decoder_args=None,
    lm_path=None,
    lm_weight=LM_WEIGHT,
    word_bonus=WORD_BONUS,
    unk_penalty=UNK_PENALTY,
):

    # The beam search and LM options only apply to CTC models.
    decoder_args = dict(decoder_args or {})

    def scorer(char_to_int):
        lm = WordLM.load(lm_path)
        return lm.scorer(char_to_int, lm_weight, word_bonus, unk_penalty)

    if model_path.endswith(".onnx"):
        from speech.onnx_model import OnnxModel

        model = OnnxModel(model_path)
        if lm_path is not None:
            decoder_args["lm"] = scorer(model.char_to_int)
        results = eval_onnx(model, dataset_json, decoder_args)
    else:
        from speech import loader

        device = speech.best_device()
        model, preproc = speech.load(model_path, tag=tag)
        ldr = loader.make_loader(dataset_json, preproc, batch_size)
        model.to(device)
        model.set_eval()
        if lm_path is not None:
            decoder_args["lm"] = scorer(preproc.char_to_int)
        results = eval_loop(model, ldr, decoder_args)
        results = [
            (preproc.decode(label), preproc.decode(pred)) for label, pred in results
        ]
    cer = speech.compute_cer(results)
    wer = speech.compute_wer(results)
    print(f"CER {cer:.3f} WER {wer:.3f}")

    if out_file is not None:
        with open(out_file, "w") as fid:
            for label, pred in results:
                res = {"prediction": pred, "label": label}
                json.dump(res, fid)
                fid.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval a speech model.")

    parser.add_argument(
        "model",
        help="A path to a stored model, or a CTC model exported to .onnx by export.py.",
    )
    parser.add_argument("dataset", help="A json file with the dataset to evaluate.")
    parser.add_argument(
        "--last",
        action="store_true",
        help="Last saved model instead of best on dev set.",
    )
    parser.add_argument("--save", help="Optional file to save predicted results.")
    decoding = parser.add_argument_group("CTC decoding")
    decoding.add_argument(
        "--beam-size",
        type=int,
        help=f"Beam size for the prefix beam search, by default 1, or {BEAM_SIZE} "
        "with an LM.",
    )
    decoding.add_argument(
        "--prune",
        type=float,
        default=PRUNE,
        help="With a beam size or LM, skip labels with a lower log probability "
        "in a frame.",
    )
    decoding.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="With a beam size or LM, try at most this many of the most likely "
        "labels of a frame.",
    )
    decoding.add_argument(
        "--lm", help="A word LM .npz file from speech.models.word_lm."
    )
    decoding.add_argument(
        "--lm-weight", type=float, default=LM_WEIGHT, help="Scales the LM scores."
    )
    decoding.add_argument(
        "--word-bonus",
        type=float,
        default=WORD_BONUS,
        help="Score added per word, which offsets the LM's preference for fewer words.",
    )
    decoding.add_argument(
        "--unk-penalty",
        type=float,
        default=UNK_PENALTY,
        help="Score added per word outside the LM's vocabulary.",
    )
    args = parser.parse_args()

    decoder_args = {}
    if args.beam_size is not None or args.lm is not None:
        decoder_args["beam_size"] = args.beam_size or BEAM_SIZE
        decoder_args["prune"] = args.prune
        decoder_args["top_k"] = args.top_k
    run(
        args.model,
        args.dataset,
        tag=None if args.last else "best",
        out_file=args.save,
        decoder_args=decoder_args,
        lm_path=args.lm,
        lm_weight=args.lm_weight,
        word_bonus=args.word_bonus,
        unk_penalty=args.unk_penalty,
    )
