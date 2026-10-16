from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from slr.keypoints.features import FeatureBuilder
from slr.models.arch import build_model


class Prediction(NamedTuple):
    label: str
    confidence: float


class SignPredictor:
    """Sliding window over frames -> model -> smoothed prediction."""

    MIN_HANDS_RATIO = 0.3    # share of frames in the window that must contain a hand

    def __init__(self, checkpoint_path: Path, confidence_threshold: float = 0.8,
                 smoothing_window: int = 5) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.label_names: list[str] = ckpt["label_names"]
        self._seq_len: int = ckpt["sequence_length"]
        self._threshold = confidence_threshold

        self._model = build_model(ckpt["model_name"], ckpt["num_features"], len(self.label_names))
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()

        self._frames: deque[np.ndarray] = deque(maxlen=self._seq_len)
        self._probs: deque[np.ndarray] = deque(maxlen=smoothing_window)

    def push(self, vector: np.ndarray) -> None:
        self._frames.append(vector)

    def predict(self) -> Prediction | None:
        if len(self._frames) < self._seq_len or not self._hands_present_enough():
            self._probs.clear()
            return None
        x = torch.from_numpy(np.stack(self._frames)).float().unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0].numpy()
        self._probs.append(probs)
        mean_probs = np.mean(self._probs, axis=0)
        best = int(mean_probs.argmax())
        if mean_probs[best] < self._threshold:
            return None
        return Prediction(self.label_names[best], float(mean_probs[best]))

    def _hands_present_enough(self) -> bool:
        visible = sum(FeatureBuilder.hands_visible(v) for v in self._frames)
        return visible / len(self._frames) >= self.MIN_HANDS_RATIO

    def reset(self) -> None:
        """Clear buffers (e.g. after a word was added to the sentence)."""
        self._frames.clear()
        self._probs.clear()