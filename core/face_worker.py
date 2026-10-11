"""Asynchronous face processing worker — runs detection + embedding in a
dedicated background thread so the main pipeline loop never blocks on NPU or
face post-processing.

Architecture
============
Without this module, the main pipeline thread does everything sequentially::

    [motion] → [record] → [face detect + embed + match] → [sleep]

When face processing takes 15-25 ms of a 66.7 ms budget the pipeline still
works, but if motion detection and recording also spike, frames get dropped.

With FaceWorker the expensive face work moves off the critical path::

    Main thread:         [motion] → [record] → submit frame → [sleep]
    FaceWorker thread:                               [detect] → [embed] → [match]

Because OpenCV and numpy release the GIL during C-level computation, and the
NPU blocks on hardware, these two threads genuinely run on different cores.

Thread safety
=============
- FaceDetector, FaceEmbedder: only accessed from the FaceWorker thread.
- FaceManager: accessed only from FaceWorker (process_face).  The SQLite
  database is in WAL mode and supports concurrent readers (web thread) and
  a single writer (this thread).
- The bounded queue (maxsize=2) prevents unbounded memory growth: when the
  worker falls behind, old frames are silently dropped.
"""
import threading
import queue
import logging

log = logging.getLogger(__name__)


class FaceWorker:
    def __init__(self, face_detector, face_embedder, face_manager):
        self.detector = face_detector
        self.embedder = face_embedder
        self.manager = face_manager

        # Bounded queue — only buffer 2 frames to avoid memory bloat when the
        # worker falls behind the main loop.
        self._queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread = None

        # Stats (read from any thread, but only updated by worker)
        self._lock = threading.Lock()
        self._processed = 0
        self._dropped = 0

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="FaceWorker")
        self._thread.start()
        log.info("FaceWorker started")

    def stop(self, timeout=5.0):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info(f"FaceWorker stopped (processed={self._processed}, "
                 f"dropped={self._dropped})")

    # ── public API ─────────────────────────────────────────────────────

    def submit(self, frame, recording_id):
        """Submit a frame for asynchronous face processing.

        Non-blocking.  If the internal queue is full the frame is silently
        dropped (the worker hasn't finished the previous batch yet).
        """
        try:
            # Copy the frame so the main thread can reuse/modify its buffer.
            self._queue.put_nowait((frame.copy(), recording_id))
        except queue.Full:
            with self._lock:
                self._dropped += 1

    @property
    def stats(self):
        with self._lock:
            return {
                'processed': self._processed,
                'dropped': self._dropped,
                'queue_size': self._queue.qsize(),
            }

    @property
    def available(self):
        """Whether face processing is enabled (modules loaded)."""
        return self.detector is not None

    # ── internals ──────────────────────────────────────────────────────

    def _run(self):
        """Main worker loop — blocks on queue, processes face frames."""
        while self._running:
            try:
                frame, recording_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_one_frame(frame, recording_id)
            except Exception:
                log.exception("Face processing error")
            finally:
                with self._lock:
                    self._processed += 1

    def _process_one_frame(self, frame, recording_id):
        """Full face pipeline for a single frame."""
        faces = self.detector.detect(frame)
        if not faces:
            return

        for face in faces:
            # Quality filter
            ok, quality = self.manager.is_quality_face(frame, face['bbox'])
            if not ok:
                continue

            embedding = self.embedder.get_embedding(frame, face['landmarks'])
            self.manager.process_face(
                frame, face['bbox'], face['confidence'],
                face['landmarks'], embedding,
                recording_id=recording_id, quality=quality,
            )
