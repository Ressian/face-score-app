# -*- coding: utf-8 -*-
"""人脸检测、滚转对齐、质量诊断。与众包训练预处理保持一致。"""
import gc
import math
import os
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageOps

from . import config


def fix_mediapipe_nonascii_path():
    if os.name != "nt":
        return
    from mediapipe.python import solution_base

    module_path = os.path.abspath(solution_base.__file__)
    if module_path.isascii():
        return
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(module_path)))
    src_modules = os.path.join(pkg_root, "mediapipe", "modules")
    if not os.path.isdir(src_modules):
        return
    dst_root = os.path.join(tempfile.gettempdir(), "yanzhi_mediapipe_res")
    dst_modules = os.path.join(dst_root, "mediapipe", "modules")
    if not os.path.isdir(dst_modules):
        os.makedirs(dst_root, exist_ok=True)
        tmp = dst_modules + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src_modules, tmp)
        os.replace(tmp, dst_modules)
    solution_base.__file__ = os.path.join(dst_root, "mediapipe", "python", "solution_base.py")


def detect_largest(rgb, detector):
    res = detector.process(rgb)
    if not res.detections:
        return None
    best = max(res.detections,
               key=lambda d: d.location_data.relative_bounding_box.width *
               d.location_data.relative_bounding_box.height)
    h, w = rgb.shape[:2]
    r = best.location_data.relative_bounding_box
    bbox = np.asarray([r.xmin * w, r.ymin * h,
                       (r.xmin + r.width) * w, (r.ymin + r.height) * h],
                      dtype=np.float32)
    kps = np.asarray([[kp.x * w, kp.y * h]
                      for kp in best.location_data.relative_keypoints],
                     dtype=np.float32)
    return {"bbox": bbox, "kps": kps, "score": float(best.score[0])}


def head_pose(kps):
    eye_r, eye_l, nose, mouth = kps[0], kps[1], kps[2], kps[3]
    eye_mid = (eye_l + eye_r) / 2.0
    iod = float(np.linalg.norm(eye_l - eye_r)) + 1e-6
    yaw = math.degrees(math.atan2(float(nose[0] - eye_mid[0]), iod))
    v1 = float(nose[1] - eye_mid[1])
    v2 = float(mouth[1] - nose[1])
    pitch = (v1 / (v2 + 1e-6) - config.PITCH_RATIO_REF) * 25.0
    roll = math.degrees(math.atan2(float(eye_l[1] - eye_r[1]),
                                   float(eye_l[0] - eye_r[0])))
    return yaw, pitch, roll


def align_crop(rgb, det, size, margin):
    x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin), 16.0)
    er, el = det["kps"][0], det["kps"][1]
    dx, dy = float(el[0] - er[0]), float(el[1] - er[1])
    angle = 0.0 if abs(dx) + abs(dy) < 1e-6 else math.degrees(math.atan2(-dy, dx))
    im = Image.fromarray(rgb)
    if abs(angle) > 0.5:
        im = im.rotate(-angle, resample=Image.Resampling.BILINEAR,
                       center=(cx, cy), fillcolor=(0, 0, 0))
    box = (int(round(cx - side / 2.0)), int(round(cy - side / 2.0)),
           int(round(cx + side / 2.0)), int(round(cy + side / 2.0)))
    return im.crop(box).resize((size, size), Image.Resampling.BILINEAR), angle


def quality_metrics(rgb, det, crop_rgb):
    import cv2

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
    face_px = min(x2 - x1, y2 - y1)
    face_ratio = max(x2 - x1, y2 - y1) / float(max(h, w) + 1e-6)
    yaw, pitch, roll = head_pose(det["kps"])

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    hf = gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32)
    hf_energy = float(hf.std())
    gf = gray.astype(np.float32)
    k = np.ones((5, 5), np.float32) / 25.0
    m1 = cv2.filter2D(gf, -1, k)
    m2 = cv2.filter2D(gf * gf, -1, k)
    local_std = np.sqrt(np.clip(m2 - m1 * m1, 0.0, None))
    median_local_std = float(np.median(local_std))
    flat_ratio = float((local_std < 3.0).mean())

    issues = []
    if face_px < config.GATE_MIN_FACE_PX:
        issues.append("原图人脸仅 %.0f 像素（远景），放大后细节不足，分数会偏低" % face_px)
    if abs(yaw) > config.GATE_MAX_YAW_DEG:
        issues.append("偏头约 %.0f 度（超过 ±%.0f 度），几何特征开始不可靠"
                      % (abs(yaw), config.GATE_MAX_YAW_DEG))
    if abs(pitch) > config.GATE_MAX_PITCH_DEG:
        issues.append("%s约 %.0f 度（超过 ±%.0f 度），三庭比例会被拉偏"
                      % ("低头（俯拍）" if pitch > 0 else "抬头（仰拍）",
                         abs(pitch), config.GATE_MAX_PITCH_DEG))
    if sharpness < config.GATE_MIN_SHARPNESS:
        issues.append("画面偏糊（清晰度 %.0f），细节不足会让分数失真" % sharpness)
    if median_local_std < config.GATE_MIN_LOCAL_STD or flat_ratio > config.GATE_MAX_FLAT_RATIO:
        issues.append("皮肤纹理几乎被抹平（纹理 %.1f，平坦像素 %.0f%%），疑似磨皮/美颜，分数会偏高"
                      % (median_local_std, flat_ratio * 100))

    fatal = []
    if abs(yaw) > config.GATE_HARD_YAW_DEG:
        fatal.append("侧脸角度约 %.0f 度，超过 %.0f 度上限，本模型不支持侧脸打分"
                     % (abs(yaw), config.GATE_HARD_YAW_DEG))
    if abs(pitch) > config.GATE_HARD_PITCH_DEG:
        fatal.append("俯仰角约 %.0f 度，超过 %.0f 度上限，请正对镜头重拍"
                     % (abs(pitch), config.GATE_HARD_PITCH_DEG))

    c_det = min(1.0, float(det.get("score", 0.9)) / 0.85)
    c_px = float(np.clip((face_px - 60.0) / 140.0, 0.05, 1.0))
    c_pose = float(np.exp(-0.5 * ((yaw / 28.0) ** 2 + (pitch / 25.0) ** 2)))
    c_tex = float(np.clip((0.90 - flat_ratio) / 0.25, 0.10, 1.0))
    confidence = float(np.clip(c_det * c_px * c_pose * c_tex, 0.02, 1.0))

    return {
        "face_ratio": face_ratio, "face_px": face_px,
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "sharpness": sharpness, "hf_energy": hf_energy,
        "median_local_std": median_local_std, "flat_ratio": flat_ratio,
        "confidence": confidence, "fatal": fatal, "issues": issues,
    }


class FacePreprocessor:
    def __init__(self, size=config.INPUT_SIZE):
        self.size = size
        self._det = None
        self._mesh = None
        self._groups = None

    def _ensure(self):
        if self._det is not None:
            return
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        fix_mediapipe_nonascii_path()
        import mediapipe as mp
        self._mp = mp
        self._det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=config.DETECT_CONF)
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        from .features import groups_mediapipe
        self._groups = groups_mediapipe(mp.solutions.face_mesh)

    def close(self):
        for obj in (self._det, self._mesh):
            try:
                obj.close()
            except Exception:
                pass
        self._det = None
        self._mesh = None
        gc.collect()

    def process(self, image_path, force_full=False, force_margin=None):
        self._ensure()
        from .features import features_from_groups

        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        w, h = pil.size
        if max(w, h) > config.MAX_SIDE:
            scale = float(config.MAX_SIDE) / max(w, h)
            pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        rgb = np.asarray(pil)

        det = detect_largest(rgb, self._det)
        if det is None:
            return {
                "ok": False, "reason": "未检测到人脸，请换一张正脸清晰照片",
                "face_found": False, "image_size": (w, h),
            }

        bx1, by1, bx2, by2 = [float(v) for v in det["bbox"]]
        face_ratio0 = max(bx2 - bx1, by2 - by1) / float(max(rgb.shape[:2]) + 1e-6)
        aspect = float(pil.size[0]) / float(max(pil.size[1], 1))
        auto_full = (face_ratio0 > 0.45 and 0.80 <= aspect <= 1.25)
        if force_full or auto_full:
            # SCUT/MEBeauty 的训练图本身就是紧贴人脸的方图；对这类输入
            # 再做 bbox 裁切会切掉额头/下巴，必须保留整图构图。
            crop320 = pil.resize((320, 320), Image.Resampling.BILINEAR)
            model_source = pil
            roll_angle = 0.0
            margin = 0.0
        else:
            # 半身/远景照才按检测框外扩；近景按 0.10 轻微外扩。
            if force_margin is not None:
                margin = float(force_margin)
            else:
                margin = 0.10 if face_ratio0 > 0.45 else config.FACE_MARGIN
            crop320, roll_angle = align_crop(rgb, det, 320, margin)
            model_source = crop320
        crop_rgb = np.asarray(crop320)
        quality = quality_metrics(rgb, det, crop_rgb)
        if quality["fatal"]:
            return {
                "ok": False, "reason": "；".join(quality["fatal"]),
                "face_found": True, "quality": quality,
                "image_size": (w, h), "crop": crop320,
            }

        res = self._mesh.process(crop_rgb)
        if not res.multi_face_landmarks:
            return {
                "ok": False, "reason": "检测到人脸框，但关键点提取失败，请换清晰正脸照片",
                "face_found": True, "quality": quality,
                "image_size": (w, h), "crop": crop320,
            }
        pts = np.asarray([[p.x, p.y] for p in res.multi_face_landmarks[0].landmark],
                         dtype=np.float32)
        geo = features_from_groups(pts, self._groups)
        crop224 = model_source.resize((self.size, self.size), Image.Resampling.BILINEAR)

        return {
            "ok": True, "face_found": True, "reason": "",
            "image_size": (w, h), "crop": crop320, "crop_model": crop224,
            "landmarks": pts, "geo": geo, "quality": quality,
            "roll_angle": roll_angle, "bbox": det["bbox"], "det_score": det["score"],
            "margin": margin,
        }

