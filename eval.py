import argparse
import json

import torch
import tqdm

import speech
from speech import loader
from speech.models.ctc_decoder import BEAM_SIZE, PRUNE
from speech.models.word_lm import LM_WEIGHT, UNK_PENALTY, WORD_BONUS, WordLM


def eval_loop(model, ldr, decoder_args):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            preds = model.infer(batch, **decoder_args)
            all_preds.extend(preds)
            all_labels.extend(batch[1])
    return list(zip(all_labels, all_preds))


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

    device = speech.best_device()

    model, preproc = speech.load(model_path, tag=tag)
    ldr = loader.make_loader(dataset_json, preproc, batch_size)

    model.to(device)
    model.set_eval()

    # The beam search and LM options only apply to CTC models.
    decoder_args = dict(decoder_args or {})
    if lm_path is not None:
        lm = WordLM.load(lm_path)
        decoder_args["lm"] = lm.scorer(
            preproc.char_to_int, lm_weight, word_bonus, unk_penalty
        )

    results = eval_loop(model, ldr, decoder_args)
    results = [(preproc.decode(label), preproc.decode(pred)) for label, pred in results]
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

    parser.add_argument("model", help="A path to a stored model.")
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
        "--lm", help="A word LM json file from speech.models.word_lm."
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
