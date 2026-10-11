"""RetinaFace face detector using NPU (RKNN).
Post-processing adapted from rknn_model_zoo/examples/RetinaFace/python/RetinaFace.py

_box_decode and _nms are delegated to a C shared library (_nms_fast.so)
for ~3-5x speedup.  Falls back to pure Python/numpy if the .so is absent.
"""
import ctypes
import os
import numpy as np
import cv2
from math import ceil
from itertools import product as product

from core.npu_backend import InferenceBackend

# ── Load C extensions (optional — fall back to pure Python) ──────────
_nms_lib = None
_landm_lib = None
_EXT_DIR = os.path.dirname(os.path.abspath(__file__))

_SO_NMS = os.path.join(_EXT_DIR, '_nms_fast.so')
if os.path.exists(_SO_NMS):
    try:
        _nms_lib = ctypes.cdll.LoadLibrary(_SO_NMS)
        _nms_lib.box_decode.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        _nms_lib.box_decode.restype = None

        _nms_lib.nms.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
        ]
        _nms_lib.nms.restype = ctypes.c_int
    except OSError:
        pass

_SO_LANDM = os.path.join(_EXT_DIR, '_decode_landm.so')
if os.path.exists(_SO_LANDM):
    try:
        _landm_lib = ctypes.cdll.LoadLibrary(_SO_LANDM)
        _landm_lib.decode_5_landmarks.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        _landm_lib.decode_5_landmarks.restype = None
    except OSError:
        pass

class FaceDetector:
    def __init__(self, rknn_path, input_size=320,
                 conf_threshold=0.5, nms_threshold=0.4):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.backend = InferenceBackend(rknn_path=rknn_path)
        # Precompute priors (only depends on input_size)
        self._priors = self._prior_box((input_size, input_size))

    def _prior_box(self, image_size):
        anchors = []
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
        for k, f in enumerate(feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes[k]:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    cx = (j + 0.5) * steps[k] / image_size[1]
                    cy = (i + 0.5) * steps[k] / image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]
        return np.array(anchors).reshape(-1, 4)

    def _letterbox(self, image):
        h, w = image.shape[:2]
        target = self.input_size
        ratio = min(target / w, target / h)
        nw, nh = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        ox, oy = (target - nw) // 2, (target - nh) // 2
        canvas[oy:oy+nh, ox:ox+nw] = resized
        return canvas, ratio, ox, oy

    def _box_decode(self, loc, priors):
        if _nms_lib is not None:
            N = loc.shape[0]
            loc_f = np.ascontiguousarray(loc, dtype=np.float32).flatten()
            priors_f = np.ascontiguousarray(priors, dtype=np.float32).flatten()
            boxes = np.empty(N * 4, dtype=np.float32)
            _nms_lib.box_decode(
                loc_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                priors_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                N,
            )
            return boxes.reshape(N, 4)

        # Pure-Python fallback
        variances = [0.1, 0.2]
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _decode_landm(self, pre, priors):
        if _landm_lib is not None:
            N = pre.shape[0]
            pre_f = np.ascontiguousarray(pre, dtype=np.float32).flatten()
            priors_f = np.ascontiguousarray(priors, dtype=np.float32).flatten()
            landm = np.empty(N * 10, dtype=np.float32)
            _landm_lib.decode_5_landmarks(
                pre_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                priors_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                landm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                N,
            )
            return landm.reshape(N, 10)

        # Pure-Python fallback
        variances = [0.1, 0.2]
        return np.concatenate([
            priors[:, :2] + pre[:, i*2:(i+1)*2] * variances[0] * priors[:, 2:]
            for i in range(5)
        ], axis=1)

    def _nms(self, dets, thresh):
        if _nms_lib is not None:
            N = dets.shape[0]
            dets_f = np.ascontiguousarray(dets, dtype=np.float32).flatten()
            keep = np.empty(N, dtype=np.int32)
            count = _nms_lib.nms(
                dets_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                N,
                ctypes.c_float(thresh),
                keep.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            )
            return keep[:count].tolist()

        # Pure-Python fallback
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(self, frame):
        """Detect faces in BGR frame.
        Returns list of dicts: {bbox: [x1,y1,x2,y2], confidence: float, landmarks: [[x,y]*5]}
        """
        img_h, img_w = frame.shape[:2]
        sz = self.input_size

        letterbox_img, ratio, ox, oy = self._letterbox(frame)
        infer_img = letterbox_img[:, :, ::-1]  # BGR -> RGB

        # RKNN 输入: NHWC uint8
        inp = np.expand_dims(infer_img, 0)

        outputs = self.backend.run([inp])
        loc, conf, landms = outputs[0], outputs[1], outputs[2]

        # Decode
        boxes = self._box_decode(loc.squeeze(0), self._priors)
        scale = np.array([sz, sz, sz, sz])
        boxes = boxes * scale
        boxes[:, 0::2] = np.clip((boxes[:, 0::2] - ox) / ratio, 0, img_w)
        boxes[:, 1::2] = np.clip((boxes[:, 1::2] - oy) / ratio, 0, img_h)

        scores = conf.squeeze(0)[:, 1]

        landmarks = self._decode_landm(landms.squeeze(0), self._priors)
        lm_scale = np.array([sz, sz] * 5)
        landmarks = landmarks * lm_scale
        landmarks[:, 0::2] = np.clip((landmarks[:, 0::2] - ox) / ratio, 0, img_w)
        landmarks[:, 1::2] = np.clip((landmarks[:, 1::2] - oy) / ratio, 0, img_h)

        # Filter by confidence
        inds = np.where(scores > self.conf_threshold)[0]
        boxes, scores, landmarks = boxes[inds], scores[inds], landmarks[inds]

        # Sort by score
        order = scores.argsort()[::-1]
        boxes, scores, landmarks = boxes[order], scores[order], landmarks[order]

        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = self._nms(dets, self.nms_threshold)

        results = []
        for i in keep:
            lm = landmarks[i].reshape(5, 2).tolist()
            results.append({
                'bbox': boxes[i].astype(int).tolist(),
                'confidence': float(scores[i]),
                'landmarks': lm,
            })
        return results

    def release(self):
        self.backend.release()
