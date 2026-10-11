"""Face embedding extractor — w600k MobileFaceNet (NPU/RKNN only)."""
import numpy as np
import cv2
from core.npu_backend import InferenceBackend

# Standard face alignment destination points for 112x112
ALIGN_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

class FaceEmbedder:
    def __init__(self, rknn_path, input_size=112):
        self.input_size = input_size
        self.backend = InferenceBackend(rknn_path=rknn_path)

    @staticmethod
    def _preprocess_lighting(img_bgr):
        """CLAHE 自适应直方图均衡化 — 增强低光照下人脸局部对比度。"""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _align_face(self, frame, landmarks):
        """Align face using 5 landmarks via affine transform."""
        src = np.array(landmarks, dtype=np.float32)
        M = cv2.estimateAffinePartial2D(src, ALIGN_DST)[0]
        if M is None:
            # Fallback: just crop and resize
            x1 = int(min(p[0] for p in landmarks))
            y1 = int(min(p[1] for p in landmarks))
            x2 = int(max(p[0] for p in landmarks))
            y2 = int(max(p[1] for p in landmarks))
            pad = max(x2 - x1, y2 - y1) // 2
            h, w = frame.shape[:2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1, y1 = max(0, cx - pad), max(0, cy - pad)
            x2, y2 = min(w, cx + pad), min(h, cy + pad)
            crop = frame[y1:y2, x1:x2]
            return cv2.resize(crop, (self.input_size, self.input_size))
        aligned = cv2.warpAffine(frame, M, (self.input_size, self.input_size))
        return aligned

    def get_embedding(self, frame, landmarks):
        aligned = self._align_face(frame, landmarks)
        # CLAHE 光照增强（在 BGR 空间做，之后再转 RGB）
        aligned = self._preprocess_lighting(aligned)
        # BGR -> RGB，保留 uint8，RKNN 输入格式
        img = aligned[:, :, ::-1]

        # RKNN: NHWC uint8
        inp = img[np.newaxis, ...]  # shape (1, 112, 112, 3), dtype=uint8

        outputs = self.backend.run([inp])
        embedding = outputs[0].flatten().astype(np.float64)
        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def release(self):
        self.backend.release()
