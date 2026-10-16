import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from slr.models.base import BaseSequenceModel


class Trainer:
    """Training loop. Works with any BaseSequenceModel, keeps the version
    with the best validation accuracy."""

    def __init__(self, model: BaseSequenceModel, model_name: str, lr: float = 1e-3) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.model_name = model_name
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self._best_state = None
        self.best_val_acc = 0.0

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self._criterion(logits, y)
                if train:
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                count += len(y)
        return total_loss / count, correct / count

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            if val_acc >= self.best_val_acc:
                self.best_val_acc = val_acc
                self._best_state = copy.deepcopy(self.model.state_dict())
            print(f"Epoch {epoch:3d}/{epochs} | train loss {train_loss:.3f} acc {train_acc:.3f}"
                  f" | val loss {val_loss:.3f} acc {val_acc:.3f}")