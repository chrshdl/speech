"""
Prints a training run's loss per character over its first epoch. The loss
of a batch grows with the length of its transcripts, and SortaGrad orders
the first epoch from short clips to long ones, so the raw loss climbs even
while the model improves. This rebuilds the batch order from the config
and divides each batch's loss by its average transcript length. Run it
from the repository root with the run's config.
"""

import argparse
import glob
import json
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from speech import loader


def main(config_path, rows):
    with open(config_path) as fid:
        config = json.load(fid)
    data_cfg, opt_cfg = config["data"], config["optimizer"]
    assert data_cfg.get("sortagrad"), "Only runs with SortaGrad have a known order."
    events = sorted(
        glob.glob(os.path.join(config["save_path"], "events.out.tfevents.*"))
    )
    accumulator = EventAccumulator(events[0], size_guidance={"scalars": 0})
    accumulator.Reload()
    loss = np.array([e.value for e in accumulator.Scalars("train_loss")])

    batch_size = opt_cfg["batch_size"]
    dataset = loader.AudioDataset(data_cfg["train_set"], None, batch_size)
    order = list(loader.BatchRandomSampler(dataset, batch_size, sortagrad=True))
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    loss = loss[: len(batches)]
    batches = batches[: len(loss)]
    chars = np.array(
        [np.mean([len(dataset.data[i]["text"]) for i in b]) for b in batches]
    )
    secs = np.array(
        [np.mean([dataset.data[i]["duration"] for i in b]) for b in batches]
    )

    step = max(1, len(loss) // rows)
    print(
        f"{'steps':>13s} {'clip s':>7s} {'chars':>6s} {'raw loss':>9s} {'per char':>9s}"
    )
    for a in range(0, len(loss), step):
        b = min(a + step, len(loss))
        print(
            f"{a:6d}-{b:6d} {np.median(secs[a:b]):7.1f} {np.median(chars[a:b]):6.0f}"
            f" {np.median(loss[a:b]):9.1f} {np.median(loss[a:b] / chars[a:b]):9.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="The run's training config.")
    parser.add_argument("--rows", type=int, default=15, help="Rows to print.")
    args = parser.parse_args()
    main(args.config, args.rows)
