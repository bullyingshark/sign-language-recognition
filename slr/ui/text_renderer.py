import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TextRenderer:
    _FONT_CANDIDATES = (
        "arial.ttf",  # Windows
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    )

    def __init__(self, size: int = 30) -> None:
        """Loads a font of the given size, trying several known paths."""
        self._font = self._load_font(size)

    def _load_font(self, size: int):
        """Tries each candidate path; returns the first that loads, or None if none work."""
        for path in self._FONT_CANDIDATES:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return None


    def draw(self, frame: np.ndarray, text: str, org: tuple[int, int],
             color_bgr: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Draws text at position org (top-left corner). Modifies frame in place."""
        if self._font is None:
            cv2.putText(frame, text, (org[0], org[1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
            return frame
        else:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(image).text(org, text, font=self._font, fill=color_bgr[::-1])
            frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return frame
