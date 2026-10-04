import torch


def best_device():
    """
    Returns the fastest available device: CUDA, then Apple's Metal
    (MPS), then the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
