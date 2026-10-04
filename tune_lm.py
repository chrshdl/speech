import argparse
import itertools
import json
import multiprocessing

import numpy as np
import torch
import tqdm

import speech
from speech import loader
from speech.models.ctc_decoder import PRUNE, decode
from speech.models.word_lm import WordLM

# Colors for the plot: the chart surface, ink and diverging scale in
# light and dark mode. Blue means a lower WER than decoding without the LM,
# red a higher one, and gray no change.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#e4e3df",
        "better": "#1c5cab",
        "neutral": "#f0efec",
        "worse": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "grid": "#383835",
        "better": "#3987e5",
        "neutral": "#383835",
        "worse": "#e66767",
    },
}

# Set in each worker process, so the model outputs and the LM are sent
# once per worker rather than once per setting.
_worker = {}


def _init_worker(outputs, char_to_int, blank, lm_path, beam_size, prune):
    _worker.update(
        outputs=outputs,
        char_to_int=char_to_int,
        int_to_char={v: k for k, v in char_to_int.items()},
        blank=blank,
        lm=WordLM.load(lm_path),
        beam_size=beam_size,
        prune=prune,
    )


def _evaluate(setting):
    """Decodes the dev set with one LM setting, or without an LM if None."""
    w = _worker
    scorer = None
    if setting is not None:
        scorer = w["lm"].scorer(w["char_to_int"], *setting)
    results = []
    for probs, text in w["outputs"]:
        labels = decode(probs, w["beam_size"], w["blank"], scorer, w["prune"])[0]
        results.append((list(text), [w["int_to_char"][i] for i in labels]))
    return setting, speech.compute_cer(results), speech.compute_wer(results)


def model_outputs(model_path, dataset_json, tag):
    """Runs the model once over the dataset."""
    model, preproc = speech.load(model_path, tag=tag)
    model.set_eval()
    outputs = []
    with torch.no_grad():
        for d in tqdm.tqdm(loader.read_data_json(dataset_json), desc="model"):
            x = preproc.preprocess(d["audio"], d["text"])[0]
            x = torch.from_numpy(x).unsqueeze(0)
            probs = model.forward_impl(x, softmax=True)[0].numpy()
            outputs.append((probs, d["text"]))
    return outputs, preproc.char_to_int, model.blank


def search(args):
    outputs, char_to_int, blank = model_outputs(args.model, args.dataset, args.tag)
    grid = [None, *itertools.product(args.weights, args.bonuses, args.unk_penalties)]
    init = (outputs, char_to_int, blank, args.lm, args.beam_size, args.prune)
    results = []
    with multiprocessing.Pool(args.jobs, _init_worker, init) as pool:
        for setting, cer, wer in tqdm.tqdm(
            pool.imap_unordered(_evaluate, grid), total=len(grid), desc="decode"
        ):
            if setting is None:
                results.append({"lm": False, "cer": cer, "wer": wer})
            else:
                weight, bonus, unk = setting
                results.append(
                    {
                        "lm": True,
                        "lm_weight": weight,
                        "word_bonus": bonus,
                        "unk_penalty": unk,
                        "cer": cer,
                        "wer": wer,
                    }
                )
    return {"beam_size": args.beam_size, "results": results}


def report(search_results):
    results = search_results["results"]
    base = next(r for r in results if not r["lm"])
    tuned = sorted((r for r in results if r["lm"]), key=lambda r: r["wer"])
    print(f"Without the LM: CER {base['cer']:.3f} WER {base['wer']:.3f}")
    print("Best settings:")
    for r in tuned[:5]:
        print(
            f"  --lm-weight {r['lm_weight']} --word-bonus {r['word_bonus']} "
            f"--unk-penalty {r['unk_penalty']}: CER {r['cer']:.3f} WER {r['wer']:.3f}"
        )


def plot(search_results, path, theme_name):
    """
    Plots the WER over the LM weight and word bonus at the best unknown-word
    penalty, as a 3D surface and as a heatmap with the values.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    theme = THEMES[theme_name]
    results = search_results["results"]
    base = next(r for r in results if not r["lm"])["wer"]
    tuned = [r for r in results if r["lm"]]
    best = min(tuned, key=lambda r: r["wer"])
    tuned = [r for r in tuned if r["unk_penalty"] == best["unk_penalty"]]
    weights = sorted({r["lm_weight"] for r in tuned})
    bonuses = sorted({r["word_bonus"] for r in tuned})
    wer = np.full((len(weights), len(bonuses)), np.nan)
    for r in tuned:
        wer[weights.index(r["lm_weight"]), bonuses.index(r["word_bonus"])] = r["wer"]
    change = wer - base

    cmap = LinearSegmentedColormap.from_list(
        "change", [theme["better"], theme["neutral"], theme["worse"]]
    )
    # Scale the colors to the best improvement, so the region worth reading
    # stands out, and clip the much worse settings.
    limit = 2 * abs(best["wer"] - base)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": theme["primary"],
            "axes.labelcolor": theme["secondary"],
            "xtick.color": theme["secondary"],
            "ytick.color": theme["secondary"],
            "axes.edgecolor": theme["grid"],
        }
    )
    fig = plt.figure(figsize=(12, 5.2), facecolor=theme["surface"])
    fig.suptitle(
        f"Dev WER by LM weight α and word bonus β "
        f"(unknown-word penalty {best['unk_penalty']:g}, beam "
        f"{search_results['beam_size']})",
        color=theme["primary"],
        fontsize=12,
    )

    # The 3D surface: height is the WER, color its change from no LM.
    ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor=theme["surface"])
    # Draw in the order added, so the best point stays on top.
    ax.computed_zorder = False
    x, y = np.meshgrid(bonuses, weights)
    ax.plot_surface(
        x,
        y,
        wer,
        facecolors=cmap(norm(change)),
        edgecolor=theme["surface"],
        linewidth=0.5,
        shade=False,
    )
    ax.plot_surface(x, y, np.full_like(wer, base), color=theme["secondary"], alpha=0.12)
    ax.plot(
        [best["word_bonus"]] * 2,
        [best["lm_weight"]] * 2,
        [best["wer"], base],
        color=theme["primary"],
        linewidth=1.5,
    )
    ax.scatter(
        best["word_bonus"],
        best["lm_weight"],
        best["wer"],
        s=60,
        color=theme["primary"],
        edgecolor=theme["surface"],
        linewidth=2,
        depthshade=False,
    )
    ax.set_xticks(bonuses, [f"{b:g}" for b in bonuses])
    ax.set_yticks(weights, [f"{w:g}" for w in weights])
    ax.set_xlabel("word bonus β", labelpad=6)
    ax.set_ylabel("LM weight α", labelpad=6)
    ax.set_zlabel("WER", labelpad=10)
    ax.tick_params(axis="z", pad=6)
    ax.view_init(elev=24, azim=-130)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(theme["surface"])
        axis._axinfo["grid"]["color"] = theme["grid"]
        axis._axinfo["grid"]["linewidth"] = 0.8
    ax.set_title(
        f"height: WER, gray plane: without the LM ({base:.3f})",
        color=theme["secondary"],
        fontsize=10,
    )

    # The heatmap: the same grid with the values, the best one ringed.
    ax = fig.add_subplot(1, 2, 2, facecolor=theme["surface"])
    image = ax.imshow(change, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_xticks(range(len(bonuses)), [f"{b:g}" for b in bonuses])
    ax.set_yticks(range(len(weights)), [f"{w:g}" for w in weights])
    ax.set_xlabel("word bonus β")
    ax.set_ylabel("LM weight α")
    for i, j in itertools.product(range(len(weights)), range(len(bonuses))):
        if np.isnan(wer[i, j]):
            continue
        strong = abs(change[i, j]) > 0.6 * limit
        ink = "#ffffff" if strong else theme["primary"]
        ax.text(j, i, f"{wer[i, j]:.3f}", ha="center", va="center", color=ink)
    i, j = weights.index(best["lm_weight"]), bonuses.index(best["word_bonus"])
    ax.add_patch(
        plt.Rectangle(
            (j - 0.5, i - 0.5),
            1,
            1,
            fill=False,
            edgecolor=theme["primary"],
            linewidth=2,
        )
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, extend="both")
    bar.set_label("WER change from no LM")
    bar.outline.set_visible(False)
    ax.set_title(
        f"best α {best['lm_weight']:g}, β {best['word_bonus']:g}: "
        f"WER {best['wer']:.3f} (no LM {base:.3f})",
        color=theme["secondary"],
        fontsize=10,
    )

    fig.savefig(path, dpi=150, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)


def floats(text):
    return [float(v) for v in text.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search the LM weight, word bonus and unknown-word "
        "penalty for decoding a CTC model with a word LM."
    )
    parser.add_argument("model", nargs="?", help="A path to a stored CTC model.")
    parser.add_argument("dataset", nargs="?", help="The dev set json to tune on.")
    parser.add_argument("--lm", help="The word LM .npz file.")
    parser.add_argument(
        "--weights", type=floats, default="0.1,0.2,0.3,0.5,0.8", help="LM weights α."
    )
    parser.add_argument(
        "--bonuses", type=floats, default="0,1,2,3,4,5", help="Word bonuses β."
    )
    parser.add_argument(
        "--unk-penalties",
        type=floats,
        default="0,-3,-6",
        help="Unknown-word penalties.",
    )
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--prune", type=float, default=PRUNE)
    parser.add_argument(
        "--jobs", type=int, default=3, help="Decoding processes to run at once."
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Last saved model instead of best on dev set.",
    )
    parser.add_argument("--results", help="A json file to save the results to.")
    parser.add_argument(
        "--from-results",
        help="Plot and report saved results instead of searching again.",
    )
    parser.add_argument(
        "--plot",
        help="A .png file for the plot. A -dark.png variant is saved next to it.",
    )
    args = parser.parse_args()

    if args.from_results is not None:
        with open(args.from_results) as fid:
            search_results = json.load(fid)
    else:
        if args.model is None or args.dataset is None or args.lm is None:
            parser.error("give a model, a dataset and --lm, or --from-results")
        args.tag = None if args.last else "best"
        search_results = search(args)
        if args.results is not None:
            with open(args.results, "w") as fid:
                json.dump(search_results, fid, indent=1)

    report(search_results)
    if args.plot is not None:
        plot(search_results, args.plot, "light")
        plot(search_results, args.plot.replace(".png", "-dark.png"), "dark")
