import argparse
import json
import math
import os
import random
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


def run_epoch(model, optimizer, train_ldr, writer, it, avg_loss):

    model_t = 0.0
    data_t = 0.0
    end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    for batch in tq:
        start_t = time.time()
        optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200).item()
        loss = loss.item()

        optimizer.step()

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


def eval_dev(model, ldr, preproc):
    losses = []
    all_preds = []
    all_labels = []

    model.set_eval()

    with torch.no_grad():
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


def save_state(path, model, optimizer, epoch, it, avg_loss, best_so_far):
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
        "rng": rng,
    }
    atomic_write(path, lambda tmp: torch.save(state, tmp))


def load_state(path, model, optimizer):
    """
    Restores a state saved with save_state. Returns the completed epochs,
    the iteration, the average loss and the best dev CER so far.
    """
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
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
    train_ldr = loader.make_loader(data_cfg["train_set"], preproc, batch_size, workers)
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

    start_epoch, it, avg_loss, best_so_far = 0, 0, 0.0, math.inf
    if resume and os.path.exists(state_path):
        start_epoch, it, avg_loss, best_so_far = load_state(
            state_path, model, optimizer
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

        run_state = run_epoch(model, optimizer, train_ldr, writer, *run_state)

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))

        dev_loss, dev_cer = eval_dev(model, dev_ldr, preproc)

        # Log for tensorboard
        writer.add_scalar("dev_loss", dev_loss, e)
        writer.add_scalar("dev_cer", dev_cer, e)
        writer.flush()

        speech.save(model, preproc, save_path)

        # Save the best model on the dev set
        if dev_cer < best_so_far:
            best_so_far = dev_cer
            speech.save(model, preproc, save_path, tag="best")

        save_state(state_path, model, optimizer, e + 1, *run_state, best_so_far)

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

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = speech.best_device()

    if device.type == "cuda" and args.deterministic:
        torch.backends.cudnn.enabled = False
    run(config, device, resume=args.resume)
