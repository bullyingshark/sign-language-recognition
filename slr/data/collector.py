import time

import cv2
import numpy as np

from slr.config import Config
from slr.keypoints.extractor import KeypointExtractor
from slr.ui.text_renderer import TextRenderer
from slr.video.camera import VideoSource

class DatasetCollector:
    """Interactively records examples of a sign from the webcam.
        Each example is saved as data/raw/<sign>/<signer>_<index>.npy.
        Controls: SPACE - record an example, Q - quit."""

    def __init__(self, config: Config, extractor: KeypointExtractor) -> None:
        self._cfg = config
        self._extractor = extractor
        self._text = TextRenderer(28)

    def _record(self, cam: VideoSource) -> np.ndarray | None:
        """2-second countdown, then record sequence_length frames."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:  # countdown
            frame = cam.read()
            if frame is None:
                return None
            left = deadline - time.monotonic()
            self._text.draw(frame, f"Get ready: {left:.1f}", (10, 10), (0, 165, 255))
            cv2.imshow("Collector", frame)
            cv2.waitKey(1)

        vectors: list[np.ndarray] = []
        while len(vectors) < self._cfg.sequence_length:  # recording
            frame = cam.read()
            if frame is None:
                return None
            results = self._extractor.process(frame)
            vectors.append(self._extractor.to_vector(results))
            self._extractor.draw(frame, results)
            self._text.draw(frame, f"REC {len(vectors)}/{self._cfg.sequence_length}", (10, 10), (0, 0, 255))
            cv2.imshow("Collector", frame)
            cv2.waitKey(1)
        return np.stack(vectors)

    def _wait_for_space(self, cam: VideoSource, status: str) -> bool:
        """Show a live preview until SPACE is pressed. Returns False on Q."""
        while True:
            frame = cam.read()
            if frame is None:
                return False
            self._extractor.draw(frame, self._extractor.process(frame))
            self._text.draw(frame, status, (10, 10), (0, 255, 255))
            self._text.draw(frame, "SPACE - record, Q - quit", (10, 45))
            cv2.imshow("Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord(" "):
                return True

    def collect(self, label: str, signer: str, n_samples: int) -> None:
        out_dir = self._cfg.data_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(out_dir.glob(f"{signer}_*.npy")))

        with VideoSource(self._cfg.camera_index, self._cfg.frame_width,
                         self._cfg.frame_height) as cam:
            for done in range(n_samples):
                if not self._wait_for_space(cam, f"'{label}'  {done}/{n_samples}"):
                    break
                sequence = self._record(cam)
                if sequence is None:
                    break
                np.save(out_dir / f"{signer}_{index:04d}.npy", sequence)
                index += 1
        cv2.destroyAllWindows()