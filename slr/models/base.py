from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseSequenceModel(nn.Module, ABC):
    """Contract: input (batch, time, features), output (batch, num_classes).

    Every new architecture (Transformer, ST-GCN, ...) inherits from here, so
    Trainer and SignPredictor work with it unchanged. This is the foundation
    for the architecture-comparison experiment in the research part.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...