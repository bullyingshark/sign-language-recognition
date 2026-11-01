"""Common skeleton for importing external datasets into the project's .npy format."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

from slr.config import Config
from slr.keypoints.features import FeatureBuilder


class RawSample(NamedTuple):
    """One gesture example as raw landmarks, before normalization."""
    label: str  # sign name, e.g. "hello"
    signer: str  # who performed it
    sample_id: str  # id of the example within the dataset
    left: np.ndarray  # (frames, 21, 3), zeros where the hand was not detected
    right: np.ndarray  # (frames, 21, 3)
    pose: np.ndarray  # (frames, 33, 3)


class BaseImporter(ABC):
    """Template method: run() is the shared algorithm, samples() differs per dataset."""

    def __init__(self, config: Config) -> None:
        self._cfg = config

    @abstractmethod
    def samples(self) -> Iterator[RawSample]:
        """Yield raw samples one by one."""

    def _target_path(self, raw: RawSample) -> Path:
        signer = raw.signer.replace("_", "-")
        sample_id = raw.sample_id.replace("_", "-")
        return self._cfg.data_dir / raw.label / f"{signer}_{sample_id}.npy"

    @staticmethod
    def _resample(seq: np.ndarray, target_len: int) -> np.ndarray:
        idx = np.linspace(0, len(seq) - 1, target_len)
        idx = np.round(idx).astype(int)
        return seq[idx]

    def run(self) -> int:
        """Convert every sample from samples() into data/raw/<label>/<signer>_<id>.npy.

        Returns the number of files saved.
        """
        saved = 0
        for raw in self.samples():
            target = self._target_path(raw)
            if target.exists():  # makes the import resumable
                continue
            n_frames = len(raw.left)
            if n_frames == 0:  # empty or unreadable sample
                continue
            vectors = np.stack([
                FeatureBuilder.build(raw.left[t], raw.right[t], raw.pose[t])
                for t in range(n_frames)
            ])
            sequence = self._resample(vectors, self._cfg.sequence_length)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, sequence)
            saved += 1
        return saved