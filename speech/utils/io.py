import os
import pickle

import torch

from speech import models

MODEL = "model"
PREPROC = "preproc.pyc"


def get_names(path, tag):
    tag = tag + "_" if tag else ""
    model = os.path.join(path, tag + MODEL)
    preproc = os.path.join(path, tag + PREPROC)
    return model, preproc


def atomic_write(path, write):
    """
    Calls write with a temporary path, then moves the file into place, so
    a crash while writing never leaves a partial file at path.
    """
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def save(model, preproc, path, tag=""):
    """
    Saves the model weights and config, which with the preprocessor are
    enough to rebuild the model.
    """
    model_n, preproc_n = get_names(path, tag)
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


def load(path, tag=""):
    """
    Loads a model and preprocessor saved with save. The model is on the
    CPU.
    """
    model_n, preproc_n = get_names(path, tag)
    with open(preproc_n, "rb") as fid:
        preproc = pickle.load(fid)
    checkpoint = torch.load(model_n, map_location="cpu")
    model_class = getattr(models, checkpoint["class"])
    model = model_class(preproc.input_dim, preproc.vocab_size, checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, preproc
