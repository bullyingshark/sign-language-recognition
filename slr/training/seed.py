"""Reproducibility helper."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator used in training."""
    random.seed(seed)
    np.random.seed(seed)       # augmentation in SignDataset
    torch.manual_seed(seed)    # weight init, dropout, DataLoader shuffling