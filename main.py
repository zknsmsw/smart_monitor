"""Main pipeline — orchestrates camera, motion detection, recording, face detection.

Multi-core optimisations (v2):
- FaceWorker runs detection + embedding in a dedicated thread so the main
  pipeline loop never blocks on NPU inference or face post-processing.
  Because OpenCV and numpy release the GIL during C-level work, and the NPU
  blocks on hardware, the two threads genuinely run on different cores.
- Frame copies are minimised: get_frame() already returns a copy from the
  camera deque; the main loop stores the reference directly instead of
  copying again.
- JPEG encoding for the MJPEG live stream is done once in the main loop
  (not on every web request), avoiding repeated cv2.imencode calls.
"""
import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime

import cv2
import config
from core.camera import CameraCapture
from core.motion import MotionDetector
from core.recorder import Recorder
from core.face_detector import FaceDetector
from core.face_embedder import FaceEmbedder
from core.face_manager import FaceManager
from core.face_worker import FaceWorker
from storage.database import Database
from storage.cleanup import DiskCleaner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('main')


class Pipeline:
    def __init__(self):
        os.makedirs(config.RECORDINGS_DIR, exist_ok=True)
        os.makedirs(config.FACE_CROPS_DIR, exist_ok=True)

        self.db = Database(config.DB_PATH)
        self.camera = CameraCapture(
            index=config.CAMERA_INDEX,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS,
        )
        self.motion = MotionDetector(
            threshold=config.MOTION_THRESHOLD,
            min_area=config.MOTION_MIN_AREA,
            blur_size=config.MOTION_BLUR_SIZE,
            detect_scale=getattr(config, 'MOTION_DETECT_SCALE', 0.5),
        )
        self.recorder = Recorder(
            output_dir=config.RECORDINGS_DIR,
            segment_seconds=config.RECORDING_SEGMENT_SECONDS,
            cooldown_seconds=config.RECORDING_COOLDOWN_SECONDS,
            codec=config.RECORDING_CODEC,
            fps=config.PIPELINE_FPS,
            frame_size=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
            db=self.db,
            use_ffmpeg=getattr(config, 'RECORDING_USE_FFMPEG', True),
            ffmpeg_encoders=getattr(config, 'RECORDING_FFMPEG_ENCODERS', None),
            ffmpeg_decoder=getattr(config, 'RECORDING_FFMPEG_DECODER', None),
            ffmpeg_hwaccel=getattr(config, 'RECORDING_FFMPEG_HWACCEL', None),
            ffmpeg_bitrate=getattr(config, 'RECORDING_FFMPEG_BITRATE', '2M'),
            # RGA hardware acceleration
            use_rga_scale=getattr(config, 'RECORDING_USE_RGA_SCALE', False),
            output_size=(getattr(config, 'RECORDING_OUTPUT_WIDTH', config.CAMERA_WIDTH),
                         getattr(config, 'RECORDING_OUTPUT_HEIGHT', config.CAMERA_HEIGHT)),
            overlay_timestamp=getattr(config, 'RECORDING_OVERLAY_TIMESTAMP', False),
            overlay_position=(getattr(config, 'RECORDING_OVERLAY_X', 10),
                              getattr(config, 'RECORDING_OVERLAY_Y', 10)),
            overlay_fontsize=getattr(config, 'RECORDING_OVERLAY_FONTSIZE', 24),
        )

        # Face modules — may fail if models not available
        self.face_worker = None
        self._init_face_modules()

        self.cleaner = DiskCleaner(
            config.RECORDINGS_DIR, config.FACE_CROPS_DIR, self.db,
            threshold=config.DISK_USAGE_THRESHOLD,
            interval=config.CLEANUP_INTERVAL_SECONDS,
        )

        self._running = False
        self._motion_event_id = None
        self._frame_count = 0

        # Live-stream state (updated by main loop, read by Flask thread)
        self._frame_lock = threading.Lock()
        self._latest_jpeg = None
        self._status_extra = {}

    # ── Face modules ────────────────────────────────────────────────────

    def _init_face_modules(self):
        """Create face modules and wrap them in an async FaceWorker."""
        try:
            detector = FaceDetector(
                rknn_path=config.RETINAFACE_RKNN,
                input_size=config.FACE_MODEL_INPUT_SIZE,
                conf_threshold=config.FACE_CONF_THRESHOLD,
                nms_threshold=config.FACE_NMS_THRESHOLD,
            )
            embedder = FaceEmbedder(
                rknn_path=config.FACE_EMBED_RKNN,
                input_size=config.EMBED_INPUT_SIZE,
            )
            manager = FaceManager(
                config.FACE_CROPS_DIR, self.db,
                match_threshold=config.FACE_MATCH_THRESHOLD,
                update_threshold=config.FACE_UPDATE_THRESHOLD,
                max_weight=config.FACE_MAX_WEIGHT,
            )
            self.face_worker = FaceWorker(detector, embedder, manager)
            log.info('Face modules initialized (async worker)')
        except Exception as e:
            log.warning(f'Face modules unavailable: {e}')
            self.face_worker = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self.camera.start()
        self.cleaner.start()
        if self.face_worker is not None:
            self.face_worker.start()

        # Web server in background thread
        web_thread = threading.Thread(target=self._start_web, daemon=True,
                                      name="WebServer")
        web_thread.start()

        log.info('Pipeline started. Press Ctrl+C to stop.')
        self._run_loop()

    def _start_web(self):
        from web.app import create_app
        app = create_app(self.db, self)
        app.run(host=config.WEB_HOST, port=config.WEB_PORT,
                threaded=True, use_reloader=False)

    def _run_loop(self):
        interval = 1.0 / config.PIPELINE_FPS
        jpeg_quality = getattr(config, 'MJPEG_QUALITY', 80)
        while self._running:
            t0 = time.time()

            # ── frame capture ──
            frame = self.camera.get_frame()   # already a .copy() from deque
            if frame is None:
                time.sleep(0.01)
                continue

            # ── motion detection ──
            has_motion, area = self.motion.detect(frame)

            # ── recording ──
            if has_motion:
                self.recorder.on_motion(frame)
                if self._motion_event_id is None:
                    self._motion_event_id = self.db.add_motion_event(
                        self.recorder.current_recording_id,
                        datetime.now().isoformat())

                # ── face detection (async — never blocks the main loop) ──
                self._frame_count += 1
                if (self._frame_count % config.FACE_DETECT_INTERVAL == 0
                        and self.face_worker is not None):
                    self.face_worker.submit(frame,
                                            self.recorder.current_recording_id)
            else:
                self.recorder.on_idle(frame)
                if self._motion_event_id is not None:
                    self.db.end_motion_event(self._motion_event_id)
                    self._motion_event_id = None

            # ── MJPEG snapshot (encode once, serve many times) ──
            ok, buf = cv2.imencode('.jpg', frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            with self._frame_lock:
                self._latest_jpeg = buf.tobytes() if ok else None

            # ── pacing ──
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        log.info('Stopping pipeline...')
        self._running = False
        self.recorder.stop()
        self.camera.stop()
        self.cleaner.stop()
        if self.face_worker is not None:
            self.face_worker.stop()

    # ── Web-facing properties ──────────────────────────────────────────

    @property
    def latest_frame_jpeg(self):
        """Return the most recent pre-encoded JPEG bytes (thread-safe)."""
        with self._frame_lock:
            return self._latest_jpeg

    @property
    def status(self):
        s = {
            'camera': self.camera.is_running,
            'recording': self.recorder.is_recording,
            'face_detection': self.face_worker is not None,
            'face_embedding': self.face_worker is not None,
        }
        if self.face_worker is not None:
            s['face_worker'] = self.face_worker.stats
        return s


def main():
    pipeline = Pipeline()

    def handle_signal(sig, frame):
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    pipeline.start()


if __name__ == '__main__':
    main()
