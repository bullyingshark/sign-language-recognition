"""Fun quick test: detect a closed-fist -> open-hand motion and show "Hello".

This is a RULE-BASED heuristic demo, not a trained model - it does not use
the ML pipeline we'll build later (DatasetRepository / SignDataset / models/...).
It only proves that VideoSource + KeypointExtractor + TextRenderer already
work together end-to-end, with a fun visible result.

Not part of the main project structure - just a standalone script in the
project root, same idea as the earlier scratch_test.py.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from slr.keypoints.extractor import KeypointExtractor
from slr.ui.text_renderer import TextRenderer
from slr.video.camera import VideoSource

# MediaPipe hand landmark indices (fixed numbering, same for every hand)
WRIST = 0
MIDDLE_MCP = 9                      # base knuckle of the middle finger - used as a size reference
FINGER_TIPS = (4, 8, 12, 16, 20)    # thumb, index, middle, ring, pinky tips

# Hysteresis thresholds - tune these after watching the live "openness" value
# printed on screen; they will depend on your camera angle and hand size.
CLOSE_THRESHOLD = 1.05   # below this -> hand counts as "closed" (fist)
OPEN_THRESHOLD = 1.40    # above this -> hand counts as "open"

GREETING_SECONDS = 2.0   # how long "Hello" stays on screen after a detected gesture


def hand_openness(hand_landmarks) -> float | None:
    """Returns a rough 0..N "how open is the hand" score, or None if no hand is visible."""
    if hand_landmarks is None:
        return None
    points = np.array([[p.x, p.y] for p in hand_landmarks.landmark])
    scale = np.linalg.norm(points[WRIST] - points[MIDDLE_MCP])
    if scale < 1e-6:          # hand too small / degenerate on this frame
        return None
    tip_distances = [np.linalg.norm(points[WRIST] - points[tip]) for tip in FINGER_TIPS]
    return float(np.mean(tip_distances) / scale)


def main() -> None:
    extractor = KeypointExtractor()
    text = TextRenderer(36)

    state = "unknown"          # "open" | "closed" | "unknown"
    greeting_until = 0.0        # monotonic timestamp until which "Hello" is shown

    with VideoSource() as cam:
        while True:
            frame = cam.read()
            if frame is None:
                break

            results = extractor.process(frame)
            hand = results.right_hand_landmarks or results.left_hand_landmarks
            openness = hand_openness(hand)

            if openness is not None:
                if openness < CLOSE_THRESHOLD:
                    state = "closed"
                elif openness > OPEN_THRESHOLD:
                    if state == "closed":              # closed -> open transition detected
                        greeting_until = time.monotonic() + GREETING_SECONDS
                    state = "open"

            extractor.draw(frame, results)
            if time.monotonic() < greeting_until:
                text.draw(frame, "Hello", (10, 10), (0, 255, 0))
            elif openness is not None:
                text.draw(frame, f"openness: {openness:.2f} ({state})", (10, 10), (0, 255, 255))
            else:
                text.draw(frame, "show your hand", (10, 10), (0, 0, 255))

            cv2.imshow("Fist -> Hello test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()