from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data/raw")
    checkpoint_dir: Path = Path("checkpoints")
    sequence_length: int = 30
    num_features: int = 225
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    confidence_threshold: float = 0.8
    smoothing_window: int = 5
    word_cooldown_sec: float = 2.0

    def checkpoint_for(self, model_name: str) -> Path:
        file_name = f"{model_name}.pt"
        path = self.checkpoint_dir / file_name
        return path

    def for_dataset(self, name: str) -> "Config":
        """Copy of this config whose data and checkpoints live in a per-dataset subfolder."""
        # data_dir.parent is "data" (from data/raw), so the result is data/<name>
        return replace(self, data_dir=self.data_dir.parent / name,
                       checkpoint_dir=self.checkpoint_dir / name)