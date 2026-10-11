"""Face group management — matching, grouping, storage with quality-aware updates.

B) 建新组缓冲 — 前 N 帧积攒后取质量加权均值作为初始组中心
D) 质量感知更新 — 清晰帧权重高、模糊帧权重低，防止坏帧持续污染
"""
import os
import cv2
import numpy as np
import logging

import config

log = logging.getLogger(__name__)


class FaceManager:
    def __init__(self, crops_dir, db, match_threshold=0.42, update_threshold=0.60, max_weight=100):
        self.crops_dir = crops_dir
        self.db = db
        self.match_threshold = match_threshold
        self.update_threshold = update_threshold
        self.max_weight = max_weight

        # 已确认群组缓存
        self._group_embeddings = {}
        self._load_groups()

        # 候选池 {candidate_id: {...}}（B 方案的核心）
        self._candidate_counter = -1
        self._candidates = {}

    def _load_groups(self):
        groups = self.db.get_all_face_groups()
        for g in groups:
            if g['mean_embedding'] is not None:
                self._group_embeddings[g['id']] = np.frombuffer(g['mean_embedding'], dtype=np.float64)

    def _cosine_similarity(self, a, b):
        return float(np.dot(a, b))

    # ──── 质量评估 ────

    @staticmethod
    def is_quality_face(frame, bbox):
        """检查人脸质量，返回 (pass: bool, score: float 0..1)"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return False, 0.0

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = gray.mean()

        min_blur = getattr(config, 'FACE_MIN_BLUR', 60)
        min_brightness = getattr(config, 'FACE_MIN_BRIGHTNESS', 35)

        if blur_val < min_blur:
            return False, 0.0
        if brightness < min_brightness:
            return False, 0.0

        # 质量评分：模糊度为主，亮度为辅，归一化到 0~1
        blur_weight = getattr(config, 'FACE_QUALITY_WEIGHT_BLUR', 0.7)
        bright_weight = getattr(config, 'FACE_QUALITY_WEIGHT_BRIGHT', 0.3)
        score = min(blur_val / 200.0, 1.0) * blur_weight + min(brightness / 128.0, 1.0) * bright_weight
        return True, round(max(0.0, min(1.0, score)), 4)

    # ──── 候选池（B 方案） ────

    def _add_candidate(self, frame, bbox, confidence, landmarks, embedding, quality, recording_id):
        """向候选池添加新人"""
        cid = self._candidate_counter
        self._candidate_counter -= 1
        self._candidates[cid] = {
            'embeddings': [embedding.copy()],
            'qualities': [quality],
            'best_frame': frame.copy(),
            'best_bbox': bbox[:],
            'best_quality': quality,
            'best_confidence': confidence,
            'best_landmarks': [lm[:] for lm in landmarks],
            'best_recording_id': recording_id,
        }
        buffer_size = getattr(config, 'FACE_MIN_BUFFER_FRAMES', 3)
        log.info(f"New candidate {cid}: 1/{buffer_size} buffered (quality={quality:.3f})")

    def _find_best_candidate(self, embedding):
        """在候选池中找相似度最高的匹配。返回 candidate_id 或 None。"""
        best_cid = None
        best_sim = -1
        for cid, cand in self._candidates.items():
            mean_emb = np.mean(cand['embeddings'], axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            sim = self._cosine_similarity(embedding, mean_emb)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        if best_sim >= self.match_threshold:
            return best_cid
        return None

    def _create_group_from_candidate(self, cid):
        """候选积攒满 → 质量加权平均 embedding → 创建正式群组"""
        cand = self._candidates[cid]
        buffer_size = getattr(config, 'FACE_MIN_BUFFER_FRAMES', 3)

        # B 方案核心：质量加权平均作为初始组中心
        embeddings = np.array(cand['embeddings'])
        qualities = np.array(cand['qualities'])
        weights = qualities / qualities.sum()
        mean_emb = np.sum(embeddings * weights[:, np.newaxis], axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        # 用最佳帧存 representative
        best_frame = cand['best_frame']
        best_bbox = cand['best_bbox']
        crop_img = self._crop_face(best_frame, best_bbox)
        group_dir = os.path.join(self.crops_dir, 'new_group_tmp')
        os.makedirs(group_dir, exist_ok=True)
        rep_path = os.path.join(group_dir, 'representative.jpg')
        cv2.imwrite(rep_path, crop_img)

        group_id = self.db.add_face_group(rep_path, mean_emb.tobytes())

        proper_dir = os.path.join(self.crops_dir, str(group_id))
        os.makedirs(proper_dir, exist_ok=True)
        new_rep_path = os.path.join(proper_dir, 'representative.jpg')
        os.rename(rep_path, new_rep_path)
        self.db.update_face_group_crop(group_id, new_rep_path)

        try:
            os.rmdir(group_dir)
        except OSError:
            pass

        self._group_embeddings[group_id] = mean_emb
        log.info(f"Group {group_id} created from candidate {cid} "
                 f"(quality-weighted avg of {len(embeddings)} frames)")

        # 为缓冲帧补存 event（用最佳帧的 bbox/confidence）
        for i, emb in enumerate(cand['embeddings']):
            event_dir = os.path.join(self.crops_dir, str(group_id))
            os.makedirs(event_dir, exist_ok=True)
            event_id = self.db.add_face_event(
                group_id, cand['best_recording_id'],
                cand['best_bbox'], cand['best_confidence'], '', emb.tobytes())
            crop_path = os.path.join(event_dir, f'{event_id}.jpg')
            cv2.imwrite(crop_path, crop_img)
            self.db.update_face_event_crop(event_id, crop_path)

        del self._candidates[cid]

    # ──── 群组更新（D 方案） ────

    def _update_existing_group(self, frame, bbox, confidence, landmarks, embedding,
                                group_id, best_sim, quality, recording_id):
        """质量感知均值更新 + event 入库"""
        if best_sim >= self.update_threshold:
            old_emb = self._group_embeddings[group_id]
            count = self.db.get_face_group_count(group_id)

            # D 方案核心：按质量加权替代等权更新
            weighted_count = min(count, self.max_weight - 1) + quality
            if count >= self.max_weight:
                new_mean = (old_emb * (self.max_weight - 1) + embedding * quality) / weighted_count
            else:
                new_mean = (old_emb * count + embedding * quality) / weighted_count

            new_mean = new_mean / np.linalg.norm(new_mean)
            self._group_embeddings[group_id] = new_mean
            self.db.update_face_group_embedding(group_id, new_mean.tobytes(), count + 1)

        # 统一保存 event
        event_dir = os.path.join(self.crops_dir, str(group_id))
        os.makedirs(event_dir, exist_ok=True)
        event_id = self.db.add_face_event(
            group_id, recording_id, bbox, confidence, '', embedding.tobytes())
        crop_path = os.path.join(event_dir, f'{event_id}.jpg')
        crop_img = self._crop_face(frame, bbox)
        cv2.imwrite(crop_path, crop_img)
        self.db.update_face_event_crop(event_id, crop_path)

        return group_id, event_id

    # ──── 主入口 ────

    def process_face(self, frame, bbox, confidence, landmarks, embedding,
                     recording_id=None, quality=1.0):
        embedding = embedding / np.linalg.norm(embedding)

        # 1) 匹配已有群组
        best_group_id = None
        best_sim = -1
        for gid, mean_emb in self._group_embeddings.items():
            sim = self._cosine_similarity(embedding, mean_emb)
            if sim > best_sim:
                best_sim = sim
                best_group_id = gid

        if best_sim >= self.match_threshold and best_group_id is not None:
            log.info(f"Match group {best_group_id} (sim={best_sim:.3f}, quality={quality:.3f})")
            return self._update_existing_group(
                frame, bbox, confidence, landmarks, embedding,
                best_group_id, best_sim, quality, recording_id)

        # 2) 匹配候选池
        best_cid = self._find_best_candidate(embedding)
        if best_cid is not None:
            cand = self._candidates[best_cid]
            cand['embeddings'].append(embedding.copy())
            cand['qualities'].append(quality)
            if quality > cand['best_quality']:
                cand['best_quality'] = quality
                cand['best_frame'] = frame.copy()
                cand['best_bbox'] = bbox[:]
                cand['best_confidence'] = confidence
                cand['best_landmarks'] = [lm[:] for lm in landmarks]
                cand['best_recording_id'] = recording_id

            buffer_size = getattr(config, 'FACE_MIN_BUFFER_FRAMES', 3)
            if len(cand['embeddings']) >= buffer_size:
                self._create_group_from_candidate(best_cid)
            else:
                log.info(f"Candidate {best_cid}: {len(cand['embeddings'])}/{buffer_size} buffered "
                         f"(sim={self._cosine_similarity(embedding, np.mean(cand['embeddings'], axis=0)):.3f})")
            return None, None

        # 3) 新候选
        self._add_candidate(frame, bbox, confidence, landmarks, embedding, quality, recording_id)
        return None, None

    # ──── 辅助方法 ────

    def _crop_face(self, frame, bbox, padding=0.35):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * padding), int(bh * padding)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        return frame[y1:y2, x1:x2]
