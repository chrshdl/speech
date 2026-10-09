import argparse
import json
import math
import os
import random
import signal
import sys
import time

import torch
import torch.optim
import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import speech
from speech import loader, models
from speech.utils.io import atomic_write

# Everything needed to resume training, saved after each epoch.
STATE = "train_state"

# The mixed_precision settings and the types they compute in.
PRECISIONS = {None: None, "bf16": torch.bfloat16, "fp16": torch.float16}


class MixedPrecision:
    """
    Runs the forward pass in bfloat16 or float16, which GPUs compute much
    faster with their tensor cores, or in float32 without a mode. The
    losses stay in float32. float16 has a small range, so its gradients
    are scaled up to keep small values from rounding to zero.
    """

    def __init__(self, mode, device):
        if mode not in PRECISIONS:
            raise ValueError(f"mixed_precision must be bf16 or fp16, not {mode}")
        self.dtype = PRECISIONS[mode]
        self.device_type = device.type
        self.scaler = torch.amp.GradScaler(device.type, enabled=mode == "fp16")

    def autocast(self):
        return torch.autocast(
            self.device_type, dtype=self.dtype, enabled=self.dtype is not None
        )


def run_epoch(
    model,
    optimizer,
    precision,
    train_ldr,
    writer,
    it,
    avg_loss,
    accumulate=1,
    grad_clip=200,
):
    """
    Trains for one epoch and returns the iteration and the average loss.
    Each optimizer step averages the gradients of `accumulate` batches, a
    larger batch than fits in memory at once, and rescales them to a norm
    of at most `grad_clip`. A last partial group of batches still makes a
    step. The iteration counts batches.
    """
    model_t = 0.0
    data_t = 0.0
    end_t = time.time()
    grad_norm = 0.0
    n = len(train_ldr)
    optimizer.zero_grad()
    tq = tqdm.tqdm(train_ldr)
    for i, batch in enumerate(tq):
        start_t = time.time()
        with precision.autocast():
            loss = model.loss(batch)
        precision.scaler.scale(loss / accumulate).backward()
        loss = loss.item()

        if (i + 1) % accumulate == 0 or i + 1 == n:
            # Clip the true gradients, not the scaled ones.
            precision.scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
            # Skips the step if float16 gradients overflowed.
            precision.scaler.step(optimizer)
            precision.scaler.update()
            optimizer.zero_grad()

        # The MPS allocator caches freed blocks for every batch shape it
        # sees, which on a small machine pushes everything else to swap.
        if model.device.type == "mps":
            torch.mps.empty_cache()

        prev_end_t = end_t
        end_t = time.time()
        model_t += end_t - start_t
        data_t += start_t - prev_end_t

        exp_w = 0.99
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
        writer.add_scalar("train_loss", loss, it)
        tq.set_postfix(
            iter=it,
            loss=loss,
            avg_loss=avg_loss,
            grad_norm=grad_norm,
            model_time=model_t,
            data_time=data_t,
        )
        it += 1

    return it, avg_loss


def eval_dev(model, ldr, preproc, precision):
    losses = []
    all_preds = []
    all_labels = []

    model.set_eval()

    with torch.no_grad(), precision.autocast():
        for batch in tqdm.tqdm(ldr):
            preds = model.infer(batch)
            loss = model.loss(batch)
            losses.append(loss.item())
            all_preds.extend(preds)
            all_labels.extend(batch[1])

    model.set_train()

    loss = sum(losses) / len(losses)
    results = [
        (preproc.decode(l), preproc.decode(p)) for l, p in zip(all_labels, all_preds)
    ]
    cer = speech.compute_cer(results)
    print(f"Dev: Loss {loss:.3f}, CER {cer:.3f}")
    return loss, cer


def save_state(
    path, model, optimizer, schedulers, precision, epoch, it, avg_loss, best_so_far
):
    """
    Saves the state to resume training from after the given number of
    epochs, including the random number generators, which shuffle the
    batches and drive dropout.
    """
    rng = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        rng["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    state = {
        "epoch": epoch,
        "it": it,
        "avg_loss": avg_loss,
        "best_so_far": best_so_far,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedulers": {name: s.state_dict() for name, s in schedulers.items()},
        "scaler": precision.scaler.state_dict(),
        "rng": rng,
    }
    atomic_write(path, lambda tmp: torch.save(state, tmp))


def load_state(path, model, optimizer, schedulers, precision):
    """
    Restores a state saved with save_state. Returns the completed epochs,
    the iteration, the average loss and the best dev CER so far.
    """
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    saved = state.get("schedulers")
    if saved is None:
        # States saved before lr_anneal hold only the lr_decay schedule.
        saved = {"lr_decay": state.get("scheduler")}
    for name, scheduler in schedulers.items():
        if saved.get(name) is not None:
            scheduler.load_state_dict(saved[name])
    if state.get("scaler"):
        precision.scaler.load_state_dict(state["scaler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if "mps" in rng and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng["mps"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state["epoch"], state["it"], state["avg_loss"], state["best_so_far"]


def logged_progress(save_path):
    """
    Returns the completed epochs and the best dev CER from the
    TensorBoard log, for checkpoints saved without a training state.
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    events = EventAccumulator(save_path)
    events.Reload()
    if "dev_cer" not in events.Tags()["scalars"]:
        return 0, math.inf
    dev_cer = events.Scalars("dev_cer")
    return max(e.step for e in dev_cer) + 1, min(e.value for e in dev_cer)


def make_schedulers(optimizer, opt_cfg):
    """
    Returns the learning rate schedules a config's optimizer sets, by
    name. lr_anneal divides the learning rate by a constant factor after
    every epoch, as Deep Speech 2 did with 1.2. lr_decay multiplies it by
    a factor when the dev loss stops improving for `patience` epochs.
    Both can be used together.
    """
    schedulers = {}
    if "lr_anneal" in opt_cfg:
        schedulers["lr_anneal"] = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=1 / opt_cfg["lr_anneal"]
        )
    if "lr_decay" in opt_cfg:
        decay = opt_cfg["lr_decay"]
        schedulers["lr_decay"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=decay["factor"],
            patience=decay["patience"],
            min_lr=decay.get("min_lr", 0.0),
        )
    return schedulers


def step_schedulers(schedulers, dev_loss):
    """Steps the schedules at the end of an epoch."""
    for name, scheduler in schedulers.items():
        if name == "lr_decay":
            scheduler.step(dev_loss)
        else:
            scheduler.step()


def run(config, device, resume=False):

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    save_path = config["save_path"]
    state_path = os.path.join(save_path, STATE)

    # Loaders. A resumed run needs the saved preprocessor, as a new one
    # would normalize the features with statistics from other samples.
    batch_size = opt_cfg["batch_size"]
    if resume:
        saved_model, preproc = speech.load(save_path)
    else:
        preproc = loader.Preprocessor(
            data_cfg["train_set"], start_and_end=data_cfg["start_and_end"]
        )
    workers = data_cfg.get("num_workers", 4)
    # Only the training data is augmented.
    augment = {
        name: data_cfg[name] for name in loader.AUGMENTATIONS if name in data_cfg
    }
    sortagrad = data_cfg.get("sortagrad", False)
    train_ldr = loader.make_loader(
        data_cfg["train_set"],
        preproc,
        batch_size,
        workers,
        augment=augment,
        sortagrad=sortagrad,
    )
    dev_ldr = loader.make_loader(data_cfg["dev_set"], preproc, batch_size, workers)

    # Model
    model_class = getattr(models, model_cfg["class"])
    model = model_class(preproc.input_dim, preproc.vocab_size, model_cfg)
    model.to(device)

    # Optimizer
    if opt_cfg.get("type", "sgd") == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["learning_rate"])
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg["learning_rate"],
            momentum=opt_cfg["momentum"],
        )

    schedulers = make_schedulers(optimizer, opt_cfg)

    precision = MixedPrecision(config.get("mixed_precision"), device)

    start_epoch, it, avg_loss, best_so_far = 0, 0, 0.0, math.inf
    if resume and os.path.exists(state_path):
        start_epoch, it, avg_loss, best_so_far = load_state(
            state_path, model, optimizer, schedulers, precision
        )
    elif resume:
        print(
            f"No {STATE} in {save_path}, resuming from the last saved model "
            "with a new optimizer state. The progress comes from the "
            "TensorBoard log."
        )
        model.load_state_dict(saved_model.state_dict())
        start_epoch, best_so_far = logged_progress(save_path)
        it = start_epoch * len(train_ldr)
    if resume:
        print(
            f"Resuming after epoch {start_epoch - 1}, the best dev CER so far "
            f"is {best_so_far:.3f}."
        )

    writer = SummaryWriter(save_path)
    run_state = (it, avg_loss)
    for e in range(start_epoch, opt_cfg["epochs"]):
        start = time.time()

        # Set before the loader starts its workers, which copy the dataset.
        train_ldr.dataset.set_epoch(e)
        active = ", ".join(train_ldr.dataset.active) or "none"
        order = "shortest first (SortaGrad)" if sortagrad and e == 0 else "random"
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {e}, augmentations: {active}, batch order: {order}, "
            f"learning rate: {lr:.3g}"
        )

        run_state = run_epoch(
            model,
            optimizer,
            precision,
            train_ldr,
            writer,
            *run_state,
            accumulate=opt_cfg.get("accumulate", 1),
            grad_clip=opt_cfg.get("grad_clip", 200),
        )

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))

        dev_loss, dev_cer = eval_dev(model, dev_ldr, preproc, precision)

        # Log for tensorboard
        writer.add_scalar("dev_loss", dev_loss, e)
        writer.add_scalar("dev_cer", dev_cer, e)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], e)
        writer.flush()

        step_schedulers(schedulers, dev_loss)

        speech.save(model, preproc, save_path)

        # Save the best model on the dev set
        if dev_cer < best_so_far:
            best_so_far = dev_cer
            speech.save(model, preproc, save_path, tag="best")

        save_state(
            state_path,
            model,
            optimizer,
            schedulers,
            precision,
            e + 1,
            *run_state,
            best_so_far,
        )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a speech model.")

    parser.add_argument("config", help="A json file with the training configuration.")
    parser.add_argument(
        "--deterministic",
        default=False,
        action="store_true",
        help="Run in deterministic mode (no cudnn). Only works on GPU.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the training saved in the config's save_path.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as fid:
        config = json.load(fid)

    # Exit cleanly on SIGTERM, such as from kill, so the data loader shuts
    # down its worker processes instead of leaving them running.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = speech.best_device()
    if device.type == "cuda":
        # Use TF32 tensor cores for the matrix products left in float32.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if device.type == "cuda" and args.deterministic:
        torch.backends.cudnn.enabled = False
    run(config, device, resume=args.resume)
