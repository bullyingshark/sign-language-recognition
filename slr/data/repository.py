from typing import NamedTuple
from pathlib import Path
import random


class SampleRecord(NamedTuple):
    """One dataset entry: file location, class index, and who performed it."""
    path: Path
    label: int
    signer: str


class DatasetRepository:
    """Scans data/raw/ and knows which classes and examples exist."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self.label_names: list[str] = sorted(p.name for p in self._root.iterdir() if p.is_dir())

        if not self.label_names:
            raise FileNotFoundError(f"No sign folders found in {self._root}")

        self.records: list[SampleRecord] = [
            SampleRecord(f, idx, f.stem.rsplit("_", 1)[0])
            for idx, name in enumerate(self.label_names)
            for f in sorted((self._root / name).glob("*.npy"))
        ]

    @property
    def signers(self) -> list[str]:
        return sorted({r.signer for r in self.records})

    def split_random(self, val_ratio: float = 0.2, seed: int = 42):
        """Random split (the same person can appear in both train and val)."""
        records = self.records[:]
        random.Random(seed).shuffle(records)
        n_val = max(1, int(len(records) * val_ratio))
        return records[n_val:], records[:n_val]

    def split_by_signer(self, val_signers: list[str]):
        """Person-independent split: the given people appear ONLY in validation."""
        val = [r for r in self.records if r.signer in val_signers]
        train = [r for r in self.records if r.signer not in val_signers]
        if not val or not train:
            raise ValueError(f"Invalid split. Available signers: {self.signers}")
        return train, val