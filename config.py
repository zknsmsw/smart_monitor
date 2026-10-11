import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RECORDINGS_DIR = os.path.join(DATA_DIR, 'recordings')
FACE_CROPS_DIR = os.path.join(DATA_DIR, 'face_crops')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(DATA_DIR, 'monitor.db')

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Pipeline
PIPELINE_FPS = 15
FACE_DETECT_INTERVAL = 5  # every N pipeline frames

# Motion detection
MOTION_THRESHOLD = 25
MOTION_MIN_AREA = 500
MOTION_BLUR_SIZE = 21
# Scale factor for detection resolution (0.0-1.0). 0.5 = half, 0.25 = quarter.
# Smaller = lower CPU. Detection runs at scaled resolution, recording stays full-res.
MOTION_DETECT_SCALE = 0.5

# Recording
RECORDING_SEGMENT_SECONDS = 30
RECORDING_COOLDOWN_SECONDS = 3
# OpenCV capture -> mp4v raw segments, then ffmpeg transcode to H.264 MP4 for playback.
RECORDING_CODEC = 'mp4v'
RECORDING_USE_FFMPEG = True
# Prefer Rockchip MPP hardware pipeline.
# We record mp4v "_raw.mp4" segments, then transcode to H.264 MP4 for playback.
# Your environment does not have libx264, so do NOT include it as a fallback.
RECORDING_FFMPEG_ENCODERS = ['h264_rkmpp']
# Raw segments are MPEG-4 Part 2 (mp4v), so use the MPEG4 HW decoder.
RECORDING_FFMPEG_DECODER = 'mpeg4_rkmpp'
RECORDING_FFMPEG_HWACCEL = 'rkmpp'
RECORDING_FFMPEG_BITRATE = '2M'

# --- RGA hardware acceleration (Rockchip RGA 2D accelerator) ---
# scale_rkrga: replaces OpenCV's cv2.resize() with hardware-accelerated scaling.
# When enabled, raw segments are recorded at camera native resolution,
# and scale_rkrga is applied during ffmpeg H.264 conversion.
# Requires ffmpeg compiled with --enable-rkrga.
RECORDING_USE_RGA_SCALE = False  # Set True to enable hardware scaling
# Target output resolution (scale_rkrga will downscale/upscale to this)
RECORDING_OUTPUT_WIDTH = 640
RECORDING_OUTPUT_HEIGHT = 480

# overlay_rkrga: hardware-accelerated timestamp overlay on recordings.
# Generates a transparent PNG timestamp overlay via OpenCV, then composites
# it onto the video using overlay_rkrga during ffmpeg conversion.
# Avoids the need for --enable-libfreetype / drawtext filter.
RECORDING_OVERLAY_TIMESTAMP = False  # Set True to enable timestamp overlay
RECORDING_OVERLAY_X = 10             # X position (pixels from left)
RECORDING_OVERLAY_Y = 10             # Y position (pixels from top)
RECORDING_OVERLAY_FONTSIZE = 24      # Font size in pixels

# Tip: To make scale_rkrga meaningful, set CAMERA_WIDTH/HEIGHT higher than
# RECORDING_OUTPUT_WIDTH/HEIGHT (e.g. camera=1280x720, output=640x480).

# Face detection (RetinaFace)
FACE_MODEL_INPUT_SIZE = 320
FACE_CONF_THRESHOLD = 0.7    # 提高阈值，减少低质量检测框进入匹配流程
FACE_NMS_THRESHOLD = 0.4

# Face embedding (MobileFaceNet, 512-dim output)
EMBED_INPUT_SIZE = 112
EMBED_DIM = 512

# Face matching & Clustering
# 低光照下同一个人相似度会降到 0.4~0.5，放宽阈值减少”同人异组”
FACE_MATCH_THRESHOLD = 0.42  # 判定为同一人的门槛（原 0.55，降低以减少漏匹配）
FACE_UPDATE_THRESHOLD = 0.60 # 判定为”极度确信”并更新均值特征的门槛
FACE_MAX_WEIGHT = 100        # 聚类中心更新的最大样本权重

# Face crop quality filter (低光照场景下过滤模糊/过暗的人脸)
FACE_MIN_BLUR = 60           # Laplacian 方差最小值（低于此值视为模糊）
FACE_MIN_BRIGHTNESS = 35     # 平均灰度最小值（低于此值视为过暗）

# Face group buffering (多帧缓冲建新组 + 质量加权均值更新)
FACE_MIN_BUFFER_FRAMES = 3   # 新组建立前需积攒的帧数（避免坏首帧拖跨组中心）
FACE_QUALITY_WEIGHT_BLUR = 0.7   # 质量评分中模糊度的权重
FACE_QUALITY_WEIGHT_BRIGHT = 0.3 # 质量评分中亮度的权重

# Disk cleanup
CLEANUP_INTERVAL_SECONDS = 600
DISK_USAGE_THRESHOLD = 0.50

# Web
WEB_HOST = '0.0.0.0'
WEB_PORT = 5000
MJPEG_QUALITY = 80  # JPEG quality for live MJPEG stream (1-100, lower = smaller + faster)

# Model paths (RKNN only — NPU 推理，不支持 CPU 回退)
RETINAFACE_RKNN = os.path.join(MODELS_DIR, 'RetinaFace_mobile320.rknn')
FACE_EMBED_RKNN = os.path.join(MODELS_DIR, 'w600k_mbf_rk3566_int8.rknn')
