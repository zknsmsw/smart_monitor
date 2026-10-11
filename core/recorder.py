"""Segmented video recorder — 30s segments, motion-triggered with cooldown.

Recording format notes (RK3566):
- Browsers/Safari prefer MP4(H.264).
- OpenCV VideoWriter often can3t reliably write H.264 directly.
- So we record a short mp4v "_raw.mp4" segment first, then call ffmpeg to reencode to H.264.
- If you have Rockchip MPP enabled ffmpeg, we try hardware decode/encode (rkmpp) first.
"""
import os
import cv2
import time
import logging
import subprocess
import threading
from datetime import datetime

log = logging.getLogger(__name__)

# Default RGBA colour for timestamp overlay (white text + semi-transparent black bg)
_TS_BG = (0, 0, 0, 160)      # semi-transparent black
_TS_FG = (255, 255, 255, 255)  # fully opaque white

class Recorder:
    def __init__(self, output_dir, segment_seconds=30, cooldown_seconds=3,
                 codec='mp4v', fps=15, frame_size=(640, 480), db=None,
                 use_ffmpeg=True,
                 ffmpeg_encoders=None,
                 ffmpeg_decoder=None,
                 ffmpeg_hwaccel=None,
                 ffmpeg_bitrate='2M',
                 # RGA hardware acceleration
                 use_rga_scale=False,
                 output_size=None,
                 overlay_timestamp=False,
                 overlay_position=(10, 10),
                 overlay_fontsize=24):
        self.output_dir = output_dir
        self.segment_seconds = segment_seconds
        self.cooldown_seconds = cooldown_seconds
        self.codec = codec
        self.fps = fps
        self.frame_size = frame_size
        self.db = db

        self.use_ffmpeg = use_ffmpeg

        # Hardware-only policy:
        # - You removed libx264, so never attempt to use it.
        # - Default to Rockchip MPP H.264 encoder.
        encoders = ffmpeg_encoders or ['h264_rkmpp']
        self.ffmpeg_encoders = [e for e in encoders if e != 'libx264']

        # Raw segments are recorded as mp4v (MPEG-4 Part 2). If caller didn't
        # specify a decoder, default to Rockchip MPP MPEG4 decoder.
        if ffmpeg_decoder is None and codec == 'mp4v':
            ffmpeg_decoder = 'mpeg4_rkmpp'
        self.ffmpeg_decoder = ffmpeg_decoder

        self.ffmpeg_hwaccel = ffmpeg_hwaccel
        self.ffmpeg_bitrate = ffmpeg_bitrate

        # --- RGA hardware acceleration settings ---
        self.use_rga_scale = use_rga_scale
        # output_size: target resolution for scale_rkrga. If None, defaults to frame_size.
        self.output_size = output_size or frame_size
        self.overlay_timestamp = overlay_timestamp
        self.overlay_pos = overlay_position
        self.overlay_fontsize = overlay_fontsize

        self._writer = None
        self._segment_start = None
        self._segment_path = None
        self._frame_count = 0
        self._recording = False
        self._last_motion_time = 0
        self._current_recording_id = None

    @property
    def is_recording(self):
        return self._recording

    @property
    def current_recording_id(self):
        return self._current_recording_id

    def on_motion(self, frame):
        self._last_motion_time = time.time()
        if not self._recording:
            self._start_segment(frame)
        else:
            self._write_frame(frame)

    def on_idle(self, frame):
        if not self._recording:
            return
        elapsed = time.time() - self._last_motion_time
        if elapsed >= self.cooldown_seconds:
            self._stop_segment()
        else:
            self._write_frame(frame)

    def _start_segment(self, frame):
        now = datetime.now()
        date_dir = os.path.join(self.output_dir, now.strftime('%Y-%m-%d'))
        os.makedirs(date_dir, exist_ok=True)
        # Record as _raw.mp4 first. If ffmpeg is enabled, we will re-encode to H.264.
        filename = now.strftime('%H-%M-%S') + ('_raw.mp4' if self.use_ffmpeg else '.mp4')
        self._segment_path = os.path.join(date_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(self._segment_path, fourcc, self.fps, self.frame_size)
        self._segment_start = time.time()
        self._frame_count = 0
        self._recording = True
        self._write_frame(frame)
        log.info(f'Recording started: {self._segment_path}')

    def _write_frame(self, frame):
        if self._writer is None:
            return
        if self.use_rga_scale:
            # Record at camera native resolution — scale_rkrga will downscale
            # during the ffmpeg H.264 conversion step.
            resized = frame
        else:
            resized = cv2.resize(frame, self.frame_size) if frame.shape[:2][::-1] != self.frame_size else frame
        self._writer.write(resized)
        self._frame_count += 1

        # Check segment duration
        if time.time() - self._segment_start >= self.segment_seconds:
            self._stop_segment()
            # Start new segment if still motion
            if time.time() - self._last_motion_time < self.cooldown_seconds:
                self._start_segment(frame)

    def _stop_segment(self):
        if self._writer:
            self._writer.release()
            self._writer = None
        if self._segment_path and os.path.exists(self._segment_path):
            duration = time.time() - self._segment_start
            start_time = datetime.fromtimestamp(self._segment_start).isoformat()
            end_time = datetime.now().isoformat()

            if self.use_ffmpeg and self._segment_path.endswith('_raw.mp4'):
                final_path = self._segment_path.replace('_raw.mp4', '.mp4')
                threading.Thread(
                    target=self._convert_to_h264,
                    args=(self._segment_path, final_path, start_time, end_time, duration),
                    daemon=True
                ).start()
                log.info(f'Recording stopped: {self._segment_path} ({duration:.1f}s), converting...')
            else:
                file_size = os.path.getsize(self._segment_path)
                if self.db:
                    self.db.add_recording(self._segment_path, start_time, end_time, duration, file_size)
                log.info(f'Recording stopped: {self._segment_path} ({duration:.1f}s, {file_size//1024}KB)')
        self._recording = False
        self._segment_path = None
        self._frame_count = 0
        self._current_recording_id = None

    def _convert_to_h264(self, raw_path, final_path, start_time, end_time, duration):
        """Convert raw mp4v segment to H.264 MP4 via ffmpeg + rkmpp.

        Hardware-only: use Rockchip MPP for decode/encode.
        If rkmpp fails, keep the raw mp4v file as-is.
        Optionally applies RGA hardware filters (scale_rkrga, overlay_rkrga).
        """
        encoder_used = None
        overlay_path = None
        try:
            # ---- RGA filter chain ----
            filter_parts = []
            need_filter_complex = False
            use_scale = (self.use_rga_scale
                         and self.output_size is not None
                         and self.output_size != self.frame_size)
            if use_scale:
                filter_parts.append(
                    f'scale_rkrga={self.output_size[0]}:{self.output_size[1]}'
                )

            if self.overlay_timestamp:
                overlay_path = self._create_timestamp_overlay(start_time)
                if overlay_path:
                    need_filter_complex = True

            for enc in self.ffmpeg_encoders:
                cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'info']

                # ---- HW decode ----
                if self.ffmpeg_hwaccel:
                    cmd += ['-hwaccel', self.ffmpeg_hwaccel]
                    cmd += ['-hwaccel_output_format', 'drm_prime']
                if self.ffmpeg_decoder:
                    cmd += ['-c:v', self.ffmpeg_decoder]

                cmd += ['-i', raw_path]

                # ---- Overlay input ----
                if overlay_path:
                    cmd += ['-i', overlay_path]

                # ---- Wire up filters ----
                if need_filter_complex:
                    if use_scale:
                        fc = f'[0:v]{filter_parts[0]}[bg]'
                        fc += f';[bg][1:v]overlay_rkrga=x={self.overlay_pos[0]}:y={self.overlay_pos[1]}[out]'
                    else:
                        fc = f'[0:v][1:v]overlay_rkrga=x={self.overlay_pos[0]}:y={self.overlay_pos[1]}[out]'
                    cmd += ['-filter_complex', fc, '-map', '[out]']
                elif filter_parts:
                    cmd += ['-vf', ','.join(filter_parts)]

                cmd += ['-c:v', enc, '-b:v', self.ffmpeg_bitrate,
                        '-movflags', '+faststart', final_path]

                ret = subprocess.run(cmd, capture_output=True, timeout=180)
                if ret.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    encoder_used = enc
                    break

                stderr_tail = ret.stderr.decode(errors='replace')[-400:]
                log.warning(f'rkmpp transcode failed (rc={ret.returncode}): {stderr_tail}')

            # ---- Cleanup ----
            if overlay_path and os.path.exists(overlay_path):
                try:
                    os.remove(overlay_path)
                except OSError:
                    pass

            if encoder_used:
                os.remove(raw_path)
                file_size = os.path.getsize(final_path)
                if self.db:
                    self.db.add_recording(final_path, start_time, end_time, duration, file_size)
                log.info(f'Converted to H.264 ({encoder_used}): {final_path} ({file_size//1024}KB)')
            else:
                file_size = os.path.getsize(raw_path)
                if self.db:
                    self.db.add_recording(raw_path, start_time, end_time, duration, file_size)
                log.warning('ffmpeg processing failed; keeping raw segment')

        except Exception as e:
            log.error(f'Convert error: {e}')
            if os.path.exists(raw_path):
                file_size = os.path.getsize(raw_path)
                if self.db:
                    self.db.add_recording(raw_path, start_time, end_time, duration, file_size)

    # ------------------------------------------------------------------ #
    #  RGA helpers
    # ------------------------------------------------------------------ #

    def _create_timestamp_overlay(self, start_time):
        """Create a semi-transparent PNG with the recording timestamp.

        The PNG is composited onto the video by ffmpeg's overlay_rkrga filter
        during the hardware-accelerated conversion.  This avoids the need for
        --enable-libfreetype (drawtext filter) in the ffmpeg build.

        Returns the path to the generated PNG, or None on failure.
        The caller is responsible for deleting the file after conversion.
        """
        try:
            import cv2
            import numpy as np
            from datetime import datetime

            # Parse the ISO-format start time
            if isinstance(start_time, str):
                try:
                    ts = datetime.fromisoformat(start_time)
                except Exception:
                    ts = datetime.now()
            else:
                ts = start_time

            text = ts.strftime('%Y-%m-%d %H:%M:%S')
            font = cv2.FONT_HERSHEY_DUPLEX
            # Scale font size proportionally to the configured size
            font_scale = self.overlay_fontsize / 28.0
            thickness = max(1, self.overlay_fontsize // 14)

            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            padding = max(6, self.overlay_fontsize // 4)
            img_w = tw + 2 * padding
            img_h = th + 2 * padding + baseline

            # RGBA image (OpenCV uses BGRA order)
            img = np.zeros((img_h, img_w, 4), dtype=np.uint8)

            # Semi-transparent black background
            cv2.rectangle(img, (0, 0), (img_w, img_h), _TS_BG, -1)
            # Fully opaque white text
            cv2.putText(img, text, (padding, th + padding),
                        font, font_scale, _TS_FG, thickness)

            overlay_dir = os.path.join(self.output_dir, '.overlay')
            os.makedirs(overlay_dir, exist_ok=True)
            ts_stamp = datetime.now().strftime('%H%M%S%f')[:-3]
            path = os.path.join(overlay_dir, f'ts_{ts_stamp}.png')
            cv2.imwrite(path, img)
            return path
        except Exception as e:
            log.warning(f'Failed to create timestamp overlay image: {e}')
            return None

    def stop(self):
        if self._recording:
            self._stop_segment()
