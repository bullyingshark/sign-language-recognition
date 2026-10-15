import numpy as np
import torch
from torch.utils.data import Dataset

from slr.data.repository import SampleRecord


class SignDataset(Dataset):
    """Loads .npy files and returns (sequence, class index) pairs."""

    def __init__(self, records: list[SampleRecord], sequence_length: int,
                 augment: bool = False) -> None:
        self._records = records
        self._length = sequence_length
        self._augment = augment

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, i: int):
        record = self._records[i]
        seq = self._fix_length(np.load(record.path))
        if self._augment:
            seq = self._augmented(seq)
        return torch.from_numpy(seq).float(), record.label

    def _fix_length(self, seq: np.ndarray) -> np.ndarray:
        """Crop or zero-pad to exactly sequence_length frames."""
        if len(seq) >= self._length:
            return seq[: self._length]
        pad = np.zeros((self._length - len(seq), seq.shape[1]), dtype=seq.dtype)
        return np.concatenate([seq, pad])

    @staticmethod
    def _augmented(seq: np.ndarray) -> np.ndarray:
        seq = seq * np.random.uniform(0.9, 1.1)
        noise = np.random.normal(0, 0.01, seq.shape).astype(seq.dtype)
        return seq + noise * (seq != 0)