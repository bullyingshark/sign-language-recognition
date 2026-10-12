import numpy as np
import mediapipe as mp

class KeypointExtractor:
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self._holistic_module = mp.solutions.holistic
        self._holistic = self._holistic_module.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray):
        """BGR-кадр -> сырой результат MediaPipe (ещё не вектор, а объект с landmarks)."""
        ...

    def to_vector(self, results) -> np.ndarray:
        """Сырой результат MediaPipe -> вектор признаков через FeatureBuilder."""
        ...

    def close(self) -> None:
        """Освобождает ресурсы MediaPipe."""
        ...