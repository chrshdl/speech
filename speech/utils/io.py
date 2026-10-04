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
    torch.save(checkpoint, model_n)
    with open(preproc_n, "wb") as fid:
        pickle.dump(preproc, fid)


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
