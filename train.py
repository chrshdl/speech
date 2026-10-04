import argparse
import json
import random
import time

import torch
import torch.optim
import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import speech
from speech import loader, models


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


def run(config, device):

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    # Loaders
    batch_size = opt_cfg["batch_size"]
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

    writer = SummaryWriter(config["save_path"])
    run_state = (0, 0)
    best_so_far = float("inf")
    for e in range(opt_cfg["epochs"]):
        start = time.time()

        run_state = run_epoch(model, optimizer, train_ldr, writer, *run_state)

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))

        dev_loss, dev_cer = eval_dev(model, dev_ldr, preproc)

        # Log for tensorboard
        writer.add_scalar("dev_loss", dev_loss, e)
        writer.add_scalar("dev_cer", dev_cer, e)
        writer.flush()

        speech.save(model, preproc, config["save_path"])

        # Save the best model on the dev set
        if dev_cer < best_so_far:
            best_so_far = dev_cer
            speech.save(model, preproc, config["save_path"], tag="best")

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
    args = parser.parse_args()

    with open(args.config, "r") as fid:
        config = json.load(fid)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = speech.best_device()

    if device.type == "cuda" and args.deterministic:
        torch.backends.cudnn.enabled = False
    run(config, device)
