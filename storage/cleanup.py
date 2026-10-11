"""Disk space auto-cleanup — deletes oldest recordings when disk usage exceeds threshold."""
import os
import shutil
import threading
import time
import logging

log = logging.getLogger(__name__)

class DiskCleaner:
    def __init__(self, recordings_dir, crops_dir, db,
                 threshold=0.85, interval=600):
        self.recordings_dir = recordings_dir
        self.crops_dir = crops_dir
        self.db = db
        self.threshold = threshold
        self.interval = interval
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                self._check_and_clean()
            except Exception as e:
                log.error(f'Cleanup error: {e}')
            time.sleep(self.interval)

    def _check_and_clean(self):
        usage = shutil.disk_usage(self.recordings_dir)
        used_ratio = usage.used / usage.total
        if used_ratio < self.threshold:
            return

        log.info(f'Disk usage {used_ratio:.1%} exceeds {self.threshold:.0%}, cleaning up...')
        recordings = self.db.get_oldest_recordings(limit=10)
        for rec in recordings:
            path = rec['file_path']
            if os.path.exists(path):
                os.remove(path)
                log.info(f'Deleted recording: {path}')
                # Clean empty date dirs
                parent = os.path.dirname(path)
                if os.path.isdir(parent) and not os.listdir(parent):
                    os.rmdir(parent)
            # Clean associated face crops
            events = self.db.get_face_events_by_recording(rec['id'])
            for ev in events:
                crop = ev['crop_path']
                if crop and os.path.exists(crop):
                    os.remove(crop)
            self.db.delete_recording(rec['id'])

            # Re-check
            usage = shutil.disk_usage(self.recordings_dir)
            if usage.used / usage.total < self.threshold:
                break

    def stop(self):
        self._running = False
