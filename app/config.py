# -*- coding: utf-8 -*-
import json
import os
import sys

APP_NAME = "颜值打分 v2"
APP_VERSION = "2.1.0"

INPUT_SIZE = 224
MAX_SIDE = 1600
FACE_MARGIN = 0.35
DETECT_CONF = 0.5

GATE_MIN_FACE_PX = 96.0
GATE_MAX_YAW_DEG = 22.0
GATE_MAX_PITCH_DEG = 20.0
GATE_MIN_SHARPNESS = 40.0
GATE_MIN_LOCAL_STD = 1.10
GATE_MAX_FLAT_RATIO = 0.72
PITCH_RATIO_REF = 1.42

GATE_HARD_YAW_DEG = 30.0
GATE_HARD_PITCH_DEG = 32.0

SCORE_MIN = 1.0
SCORE_MAX = 10.0
SHRINK_MIN = 0.45

TRACK_ORDER = ("scut", "meb", "crowd")
TRACK_LABELS = {
    "scut": "SCUT 主模型",
    "meb": "MEBeauty 模型",
    "crowd": "众包模型",
    "ensemble": "三模型平均",
}

GENDERS = ("female", "male")
GENDER_LABELS = {
    "female": "女性专用",
    "male": "男性专用",
    "generic": "通用",
}


def resource_root():
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def weight_root():
    return os.path.join(resource_root(), "weights_v2")


def track_dir(track):
    return os.path.join(weight_root(), track)


def track_weight(track, gender=None):
    if track == "scut" and gender in GENDERS:
        path = os.path.join(track_dir("scut_" + gender), "best.pth")
        if os.path.exists(path):
            return path
    return os.path.join(track_dir(track), "best.pth")


def track_calib(track, gender=None):
    if track == "scut" and gender in GENDERS:
        path = os.path.join(track_dir("scut_" + gender), "calib.json")
        if os.path.exists(path):
            return path
    return os.path.join(track_dir(track), "calib.json")


def load_calibration(track, gender=None):
    if track == "scut" and gender in GENDERS:
        path = os.path.join(track_dir("scut_" + gender), "calib.json")
    else:
        path = track_calib(track, gender)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pred = data.get("pred") or []
    gt = data.get("gt") or []
    if len(pred) < 2 or len(pred) != len(gt):
        return None
    return pred, gt


def available_tracks():
    return [t for t in TRACK_ORDER if os.path.exists(track_weight(t))]


def available_genders():
    found = [g for g in GENDERS if os.path.exists(track_weight("scut", g))]
    if os.path.exists(track_weight("scut")):
        found.append("generic")
    return found

