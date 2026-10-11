"""Camera capture thread — reads frames into a deque for the pipeline."""
import cv2
import threading
import logging
from collections import deque

log = logging.getLogger(__name__)

class CameraCapture:
    def __init__(self, index=0, width=640, height=480, fps=30, queue_size=5):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._queue = deque(maxlen=queue_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._cap = None

    def start(self):
        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f'Cannot open camera {self.index}')
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info(f'Camera {self.index} started ({self.width}x{self.height}@{self.fps})')

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                log.warning('Camera read failed')
                continue
            with self._lock:
                self._queue.append(frame)

    def get_frame(self):
        with self._lock:
            return self._queue[-1].copy() if self._queue else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        log.info('Camera stopped')

    @property
    def is_running(self):
        return self._running and self._cap is not None and self._cap.isOpened()
