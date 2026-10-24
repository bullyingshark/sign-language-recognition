import cv2

from slr.config import Config
from slr.inference.predictor import SignPredictor
from slr.inference.sentence import SentenceBuilder
from slr.keypoints.extractor import KeypointExtractor
from slr.ui.text_renderer import TextRenderer
from slr.video.camera import VideoSource


class RealtimeApp:
    """The "conductor": computes nothing itself, only wires the components together.

    Frame -> KeypointExtractor -> SignPredictor -> SentenceBuilder -> screen.
    Keys: Q - quit, C - clear text.
    """

    MAX_LINES = 3  # how many last lines of the sentence are shown
    LINE_HEIGHT = 36  # pixels per line (font size 30 + spacing)


    def __init__(self, config: Config, extractor: KeypointExtractor,
                 predictor: SignPredictor, sentence: SentenceBuilder) -> None:
        self._cfg = config
        self._extractor = extractor
        self._predictor = predictor
        self._sentence = sentence
        self._text = TextRenderer(30)

    def _render(self, frame, prediction) -> None:
        h, w = frame.shape[:2]
        lines = self._text.wrap(self._sentence.text, w - 20)[-self.MAX_LINES:]

        hint = f"{prediction.label} ({prediction.confidence:.0%})" if prediction else "..."
        self._text.draw(frame, hint, (10, 10), (0, 255, 255))  # current sign, top-left

        if lines:
            bar_h = len(lines) * self.LINE_HEIGHT + 20  # bar height from the line count
            cv2.rectangle(frame, (0, h - bar_h), (w, h), (0, 0, 0), -1)
            for i, line in enumerate(lines):
                self._text.draw(frame, line, (10, h - bar_h + 10 + i * self.LINE_HEIGHT), (255, 255, 255))

    def run(self) -> None:
        with VideoSource(self._cfg.camera_index, self._cfg.frame_width,
                         self._cfg.frame_height) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    break

                # 1. frame -> landmarks -> feature vector -> sliding window -> prediction
                results = self._extractor.process(frame)
                self._predictor.push(self._extractor.to_vector(results))
                current = self._predictor.predict()

                # 2. add the word to the sentence; if accepted, restart the window
                if current and self._sentence.update(current.label):
                    self._predictor.reset()

                # 3. draw landmarks first, text on top, then show the frame
                self._extractor.draw(frame, results)
                self._render(frame, current)
                cv2.imshow("Sign Language Recognition", frame)

                # 4. keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    self._sentence.clear()
        cv2.destroyAllWindows()