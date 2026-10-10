import importlib

# Imported when first used, so the decoder and the language model, which
# need only NumPy, can be imported without PyTorch.
_LAZY = {
    "CTC": "speech.models.ctc_model",
    "Model": "speech.models.model",
    "Seq2Seq": "speech.models.seq2seq",
    "Transducer": "speech.models.transducer_model",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_LAZY[name]), name)
