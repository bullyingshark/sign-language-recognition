from dataclasses import dataclass
from pathlib import Path

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
