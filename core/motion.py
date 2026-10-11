"""Frame-difference motion detection (CPU-only).

Optimisation: motion detection runs at a reduced resolution (detect_scale)
while the recording pipeline keeps the original frame.  A 0.5 scale cuts
pixel count to 1/4, 0.25 to 1/16 — the biggest CPU win with zero
practical impact on detection accuracy.
"""
import cv2
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25, min_area=500, blur_size=21,
                 detect_scale=0.5):
        self.threshold = threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self.detect_scale = detect_scale
        self._prev_gray = None

    def detect(self, frame):
        """Returns (has_motion: bool, motion_area: int).

        When detect_scale < 1.0 the frame is downscaled before processing.
        min_area is NOT auto-scaled — set it based on your detect resolution.
        """
        if self.detect_scale > 0 and self.detect_scale < 1.0:
            h, w = frame.shape[:2]
            dh = int(h * self.detect_scale)
            dw = int(w * self.detect_scale)
            small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_NEAREST)
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0

        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours)

        return motion_area >= self.min_area, int(motion_area)

    def reset(self):
        self._prev_gray = None
