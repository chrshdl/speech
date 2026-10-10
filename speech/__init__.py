import importlib

# Imported when first used, so the parts that run without PyTorch, such as
# speech.onnx_model, never import it.
_LAZY = {
    "best_device": "speech.utils.device",
    "compute_cer": "speech.utils.score",
    "compute_wer": "speech.utils.score",
    "load": "speech.utils.io",
    "save": "speech.utils.io",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_LAZY[name]), name)
