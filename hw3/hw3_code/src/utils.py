import os
import sys
import json
import random
from dataclasses import asdict

import numpy as np
import torch
from typing import List
import logging
from datasets import load_dataset


def set_seed(seed):
    """Set random seed for reproducibility (basic version)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def setup_deterministic(seed: int = 3047, device: str = None) -> str:
    """
    Configure deterministic behavior for reproducible results.

    Args:
        seed: Random seed to use
        device: Device to set as default. If None, uses best available device.

    Returns:
        str: The device that was set
    """
    if device is None:
        device = get_best_device()

    # Set random seeds
    set_seed(seed)

    # Set environment variables for deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

    # Configure PyTorch for deterministic behavior
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    # Set default device
    torch.set_default_device(device)

    return device


def setup_logging():
    """monotonous bookkeeping"""
    work_dir = "logs"
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))


def format_review(row):
    return {
        "text": f"{row['translation']['eng']}[SEP]{row['translation']['engyay']}[END]"
    }


def load_and_prepare_data(split: str = "train") -> List[str]:
    """Load and prepare the data from a given split."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading piglatin dataset split: {split}...")

    try:
        dataset = load_dataset("cdleong/piglatin-mt")[split]
        data = [row["text"] for row in dataset.map(format_review).to_list()]
        if split == "train":
            random.shuffle(data)
        logger.info(f"Loaded {len(data)} samples from {split} split")
        return data
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
