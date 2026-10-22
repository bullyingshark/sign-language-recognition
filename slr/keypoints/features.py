import numpy as np

class FeatureBuilder:
    HAND_POINTS = 21
    POSE_POINTS = 33
    _L_SHOULDER = 11
    _R_SHOULDER = 12

    @classmethod
    def vector_size(cls) -> int:
        return (2 * cls.HAND_POINTS + cls.POSE_POINTS) * 3

    @classmethod
    def build(cls, left: np.ndarray, right: np.ndarray, pose: np.ndarray) -> np.ndarray:
        l_shoulder = pose[cls._L_SHOULDER]
        r_shoulder = pose[cls._R_SHOULDER]

        origin = (l_shoulder + r_shoulder) / 2
        scale = np.linalg.norm(l_shoulder[:2] - r_shoulder[:2])

        if scale < 1e-6:  # shoulders not detected, normalization is impossible
            return np.zeros(cls.vector_size(), dtype=np.float32)

        for arr in (left, right, pose):
            present = np.any(arr != 0, axis=1)
            arr[present] = (arr[present] - origin) / scale

        return np.concatenate([left.ravel(), right.ravel(), pose.ravel()])

