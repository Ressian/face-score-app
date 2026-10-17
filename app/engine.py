# -*- coding: utf-8 -*-
import os
import time

import numpy as np

from . import config
from .model import build_model
from .preprocess import FacePreprocessor


class ScoreEngine:
    def __init__(self, tracks=None, device="cpu", tta=False, threads=4,
                 gender="female"):
        self.tracks = list(tracks or config.available_tracks())
        if not self.tracks:
            raise RuntimeError("weights_v2 下没有可用权重")
        self.device = device
        self.tta = bool(tta)
        self.pre = FacePreprocessor()
        self._models = {}
        self._calib = {}
        self._gender_models = {}
        self._gender_calib = {}
        self._tf = None
        self._torch = None
        self._threads = int(threads)
        self._load()

    def _load_model(self, path):
        import torch

        model = build_model(14)
        state = torch.load(path, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state)
        return model.to(self.device).eval()

    def _load(self):
        import torch
        from torchvision.transforms import v2

        self._torch = torch
        torch.set_num_threads(max(1, self._threads))
        torch.set_grad_enabled(False)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self._tf = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(config.INPUT_SIZE, antialias=True),
            v2.Normalize(mean, std),
        ])
        for track in self.tracks:
            path = config.track_weight(track)
            if not os.path.exists(path):
                continue
            self._models[track] = self._load_model(path)
            self._calib[track] = config.load_calibration(track)
        if "scut" in self.tracks:
            for gender in config.GENDERS:
                path = config.track_weight("scut", gender)
                if os.path.basename(os.path.dirname(path)) != "scut_" + gender:
                    continue
                self._gender_models[gender] = self._load_model(path)
                self._gender_calib[gender] = config.load_calibration("scut", gender)
        self.tracks = [t for t in self.tracks if t in self._models]
        if not self.tracks:
            raise RuntimeError("权重加载失败")

    def close(self):
        self.pre.close()
        self._models.clear()
        self._gender_models.clear()

    def _tensor(self, pil):
        return self._tf(pil).unsqueeze(0).to(self.device)

    def _calibrate(self, track, value, gender="generic"):
        if track == "scut" and gender in self._gender_models:
            cal = self._gender_calib.get(gender)
        else:
            cal = self._calib.get(track)
        if not cal:
            return float(value)
        xp, fp = cal
        return float(np.interp(float(value), np.asarray(xp, dtype=np.float64),
                               np.asarray(fp, dtype=np.float64)))

    def _infer_one(self, track, x, geo, gender="generic"):
        torch = self._torch
        if track == "scut" and gender in self._gender_models:
            model = self._gender_models[gender]
        else:
            model = self._models[track]
        g = torch.from_numpy(np.asarray(geo, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score, _ = model(x, g)
            raw = float(score.reshape(-1)[0].cpu())
            if self.tta:
                score2, _ = model(torch.flip(x, dims=[3]), g)
                raw = 0.5 * (raw + float(score2.reshape(-1)[0].cpu()))
        return raw

    def predict(self, image_path, track="scut", gender="female"):
        t0 = time.time()
        gender = gender if gender in config.GENDERS else "generic"
        if track != "ensemble" and track not in self._models:
            track = self.tracks[0]
        gender_used = gender if gender in self._gender_models else "generic"
        model_label = config.TRACK_LABELS.get(track, track)
        if track in ("scut", "ensemble") and gender_used in config.GENDERS:
            model_label += " / " + config.GENDER_LABELS[gender_used]
        pack = self.pre.process(image_path)
        if not pack.get("ok"):
            return {
                "ok": False, "reason": pack.get("reason", "无法处理该图片"),
                "face_found": pack.get("face_found", False),
                "quality": pack.get("quality", {}), "crop": pack.get("crop"),
                "track": track, "gender": gender_used,
                "model_label": model_label, "elapsed": time.time() - t0,
            }

        x = self._tensor(pack["crop_model"])
        geo = pack["geo"]
        per_track = {}
        for name in self.tracks:
            use_gender = gender_used if name == "scut" else "generic"
            raw = self._infer_one(name, x, geo, use_gender)
            per_track[name] = {
                "raw": raw,
                "calibrated": self._calibrate(name, raw, use_gender),
            }

        if track == "ensemble":
            chosen = self.tracks
            raw = float(np.mean([per_track[t]["raw"] for t in chosen]))
            calibrated = float(np.mean([per_track[t]["calibrated"] for t in chosen]))
        else:
            chosen = [track]
            raw = per_track[track]["raw"]
            calibrated = per_track[track]["calibrated"]

        quality = pack["quality"]
        confidence = float(quality.get("confidence", 0.5))
        adjusted, corrections = self._quality_adjust(raw, quality, confidence)

        return {
            "ok": True, "reason": "", "track": track, "tracks_used": chosen,
            "gender": gender_used, "model_label": model_label,
            "score_raw": float(np.clip(raw, config.SCORE_MIN, config.SCORE_MAX)),
            "score_calibrated": float(np.clip(calibrated, config.SCORE_MIN, config.SCORE_MAX)),
            "score_adjusted": float(np.clip(adjusted, config.SCORE_MIN, config.SCORE_MAX)),
            "corrections": corrections,
            "confidence": confidence,
            "reliability": self._reliability(confidence),
            "per_track": per_track,
            "quality": quality,
            "issues": quality.get("issues", []),
            "crop": pack["crop"], "crop_model": pack["crop_model"],
            "landmarks": pack["landmarks"], "geo": geo,
            "elapsed": time.time() - t0,
        }

    def _quality_adjust(self, raw, quality, confidence):
        score = float(raw)
        notes = []

        flat = float(quality.get("flat_ratio", 0.0))
        if flat > config.GATE_MAX_FLAT_RATIO:
            penalty = min(0.8, (flat - config.GATE_MAX_FLAT_RATIO) / 0.18 * 0.8)
            score -= penalty
            notes.append("磨皮补偿 -%.2f" % penalty)

        pitch = float(quality.get("pitch", 0.0))
        if pitch < -10.0:
            bonus = min(0.4, (-pitch - 10.0) / 15.0 * 0.4)
            score += bonus
            notes.append("仰视补偿 +%.2f" % bonus)

        face_px = float(quality.get("face_px", 9999.0))
        if face_px < config.GATE_MIN_FACE_PX:
            k = max(config.SHRINK_MIN, face_px / config.GATE_MIN_FACE_PX)
            score = 6.0 + (score - 6.0) * k
            notes.append("远景收缩至 %.2f" % score)
        elif confidence < 0.35:
            k = config.SHRINK_MIN + (1.0 - config.SHRINK_MIN) * (confidence / 0.35)
            score = 6.0 + (score - 6.0) * k
            notes.append("低置信度收缩至 %.2f" % score)

        return score, notes

    @staticmethod
    def _reliability(confidence):
        if confidence >= 0.75:
            return "高"
        if confidence >= 0.45:
            return "中"
        return "低"


def format_result(result, show_tracks=True):
    if not result.get("ok"):
        return "无法打分：%s" % result.get("reason", "未知错误")
    lines = []
    lines.append("评分模型：%s" % result.get("model_label", ""))
    lines.append("综合分：%.2f / 10" % result["score_raw"])
    lines.append("分布校准分：%.2f / 10" % result["score_calibrated"])
    if result.get("corrections"):
        lines.append("质量修正参考分：%.2f / 10" % result["score_adjusted"])
    lines.append("置信度：%.0f%%（%s）" % (result["confidence"] * 100, result["reliability"]))
    if show_tracks and len(result.get("per_track", {})) > 1:
        lines.append("")
        lines.append("各路线原始分：")
        for name, item in result["per_track"].items():
            label = config.TRACK_LABELS.get(name, name)
            if name == "scut" and result.get("gender") in config.GENDERS:
                label += " / " + config.GENDER_LABELS[result["gender"]]
            lines.append("  %s：%.2f（校准 %.2f）" % (
                label, item["raw"], item["calibrated"]))
    q = result.get("quality", {})
    lines.append("")
    lines.append("人脸框：%.0f px，偏头 %+.1f°，俯仰 %+.1f°" % (
        q.get("face_px", 0), q.get("yaw", 0), q.get("pitch", 0)))
    lines.append("清晰度：%.0f，纹理：%.2f，平坦像素：%.0f%%" % (
        q.get("sharpness", 0), q.get("median_local_std", 0), q.get("flat_ratio", 0) * 100))
    if result.get("corrections"):
        lines.append("")
        lines.append("质量修正：" + "；".join(result["corrections"]))
    if result.get("issues"):
        lines.append("")
        lines.append("可靠性提示：")
        for item in result["issues"]:
            lines.append("  · " + item)
    return "\n".join(lines)

