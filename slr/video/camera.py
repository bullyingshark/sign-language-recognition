import numpy as np
import cv2



class VideoSource:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 mirror: bool = True) -> None:
        """Opens the camera. Raises RuntimeError if it could not be opened."""
        self._mirror = mirror
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {index}")

    def read(self) -> np.ndarray | None:
        """Returns one BGR frame, or None if a frame could not be read."""
        ok, frame = self._cap.read()
        if not ok:
            return None
        else:
            return cv2.flip(frame, 1) if self._mirror else frame

    def release(self) -> None:
        """Releases the camera."""
        ...

    def __enter__(self) -> "VideoSource":
        ...

    def __exit__(self, *exc) -> None:
        ...