import numpy as np
import mediapipe as mp
import cv2

from slr.keypoints.features import FeatureBuilder

class KeypointExtractor:
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self._holistic_module = mp.solutions.holistic
        self._drawer = mp.solutions.drawing_utils
        self._holistic = self._holistic_module.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)
        return results

    def to_vector(self, results) -> np.ndarray:
        left = self._to_array(results.left_hand_landmarks, FeatureBuilder.HAND_POINTS)
        right = self._to_array(results.right_hand_landmarks, FeatureBuilder.HAND_POINTS)
        pose = self._to_array(results.pose_landmarks, FeatureBuilder.POSE_POINTS)

        return FeatureBuilder.build(left, right, pose)

    @staticmethod
    def _to_array(landmark_list, n_points: int) -> np.ndarray:
        if landmark_list is None:
            return np.zeros((n_points, 3), dtype=np.float32)
        return np.array([[p.x, p.y, p.z] for p in landmark_list.landmark], dtype=np.float32)

    def draw(self, frame_bgr: np.ndarray, results) -> None:
        """Draw detected landmarks onto the frame. Returns nothing — modifies frame_bgr in place."""
        parts = (
            (results.pose_landmarks, self._holistic_module.POSE_CONNECTIONS),
            (results.left_hand_landmarks, self._holistic_module.HAND_CONNECTIONS),
            (results.right_hand_landmarks, self._holistic_module.HAND_CONNECTIONS),
        )
        for landmarks, connections in parts:
            if landmarks is not None:
                self._drawer.draw_landmarks(frame_bgr, landmarks, connections)

    def close(self) -> None:
        self._holistic.close()