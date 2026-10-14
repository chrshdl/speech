"""
Writes a trial config to check whether the 3,750 hour run can learn without
augmentation. It is the full run's config with every augmentation off,
trained for one epoch on a random sample of 96,000 training clips, 3,000
batches of 32, and evaluated on 2,000 dev clips. Run it from the repository
root, then train with the command it prints.
"""

import json
import random

CONFIG = "examples/commonvoice/ctc_streaming_3750h_config.json"
TRIAL = "examples/commonvoice/trial_noaug_config.json"
SAMPLES = {"train_set": 96000, "dev_set": 2000}

random.seed(0)
with open(CONFIG) as fid:
    cfg = json.load(fid)

for split, n in SAMPLES.items():
    lines = []
    for path in cfg["data"][split]:
        with open(path) as fid:
            lines.extend(fid)
    random.shuffle(lines)
    path = f"examples/commonvoice/data/trial_{split}.json"
    with open(path, "w") as fid:
        fid.writelines(lines[:n])
    cfg["data"][split] = path
    print(f"Wrote {path} with {min(n, len(lines))} of {len(lines)} clips.")

# Noise and reverb already start at epoch 1, so the one epoch has no
# augmentation at all.
for name in ["volume", "pitch", "tempo", "spec_augment"]:
    cfg["data"][name]["from_epoch"] = 1
cfg["optimizer"]["epochs"] = 1
cfg["save_path"] = "examples/commonvoice/models/trial_noaug"

with open(TRIAL, "w") as fid:
    json.dump(cfg, fid, indent=4)
print(f"Wrote {TRIAL}. Train with:")
print(f"caffeinate -i uv run train.py {TRIAL}")
