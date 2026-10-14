import os
import pickle

import torch

from speech import models

MODEL = "model"
PREPROC = "preproc.pyc"


def get_names(path, tag="", epoch=None):
    """
    Returns the paths of the model and the preprocessor, such as
    best_model and best_preproc.pyc for the tag "best", or model_epoch_3
    and preproc_epoch_3.pyc for the model saved after epoch 3.
    """
    tag = tag + "_" if tag else ""
    epoch = f"_epoch_{epoch}" if epoch is not None else ""
    model = os.path.join(path, tag + MODEL + epoch)
    stem, ext = os.path.splitext(PREPROC)
    preproc = os.path.join(path, tag + stem + epoch + ext)
    return model, preproc


def atomic_write(path, write):
    """
    Calls write with a temporary path, then moves the file into place, so
    a crash while writing never leaves a partial file at path.
    """
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def save(model, preproc, path, tag="", epoch=None):
    """
    Saves the model weights and config, which with the preprocessor are
    enough to rebuild the model, under the names of get_names.
    """
    model_n, preproc_n = get_names(path, tag, epoch)
    checkpoint = {
        "class": type(model).__name__,
        "config": model.config,
        "state_dict": model.state_dict(),
    }
    atomic_write(model_n, lambda tmp: torch.save(checkpoint, tmp))

    def write_preproc(tmp):
        with open(tmp, "wb") as fid:
            pickle.dump(preproc, fid)

    atomic_write(preproc_n, write_preproc)


def load(path, tag="", epoch=None):
    """
    Loads a model and preprocessor saved with save. The model is on the
    CPU.
    """
    model_n, preproc_n = get_names(path, tag, epoch)
    with open(preproc_n, "rb") as fid:
        preproc = pickle.load(fid)
    checkpoint = torch.load(model_n, map_location="cpu")
    model_class = getattr(models, checkpoint["class"])
    model = model_class(preproc.input_dim, preproc.vocab_size, checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, preproc
