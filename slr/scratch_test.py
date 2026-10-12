import cv2

from slr.keypoints.extractor import KeypointExtractor
from slr.video.camera import VideoSource

extractor = KeypointExtractor()

with VideoSource() as cam:
    while True:
        frame = cam.read()
        if frame is None:
            break

        results = extractor.process(frame)
        extractor.draw(frame, results)

        cv2.imshow("Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

extractor.close()
cv2.destroyAllWindows()