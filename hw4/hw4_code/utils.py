import torch


def get_best_device() -> str:
    """
    Get the best available device in order of preference: cuda, mps, cpu.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
