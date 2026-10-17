# -*- coding: utf-8 -*-
"""与训练完全一致的 14 维几何特征（mediapipe 468 点）。"""
import numpy as np

DIM_NAMES = ["mid_low_ratio", "lip_chin_ratio", "five_1", "five_2", "sym_feature",
             "nose_ratio", "mouth_nose_ratio", "jaw_ratio", "face_w_ratio",
             "mouth_w_ratio", "eye_aspect_left", "eye_aspect_right",
             "brow_eye_gap", "face_aspect"]

EPS = 1e-6
NOSE_LOW_FRAC = 0.35


def groups_mediapipe(mp_face_mesh):
    def idx(pairs):
        s = set()
        for a, b in pairs:
            s.add(a)
            s.add(b)
        return sorted(s)

    return {
        "contour": idx(mp_face_mesh.FACEMESH_FACE_OVAL),
        "brow_l": idx(mp_face_mesh.FACEMESH_LEFT_EYEBROW),
        "brow_r": idx(mp_face_mesh.FACEMESH_RIGHT_EYEBROW),
        "eye_l": idx(mp_face_mesh.FACEMESH_LEFT_EYE),
        "eye_r": idx(mp_face_mesh.FACEMESH_RIGHT_EYE),
        "nose": idx(mp_face_mesh.FACEMESH_NOSE),
        "mouth": idx(mp_face_mesh.FACEMESH_LIPS),
    }


def _side_groups(pts, groups):
    if groups.get("eye_l") is not None and groups.get("eye_r") is not None:
        return groups["eye_l"], groups["eye_r"], groups["brow_l"], groups["brow_r"]
    ex = list(groups["eye"])
    bx = list(groups["brow"])
    exs = np.median(pts[ex, 0])
    bxs = np.median(pts[bx, 0])
    el = [i for i in ex if pts[i, 0] <= exs]
    er = [i for i in ex if pts[i, 0] > exs]
    bl = [i for i in bx if pts[i, 0] <= bxs]
    br = [i for i in bx if pts[i, 0] > bxs]
    return el, er, bl, br


def features_from_groups(pts, groups):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("关键点形状应为 (N,2)，实际 %s" % (pts.shape,))
    ey_l, ey_r, bx_l, bx_r = _side_groups(pts, groups)

    contour = pts[groups["contour"]]
    chin = contour[np.argmax(contour[:, 1])]
    face_w = float(contour[:, 0].max() - contour[:, 0].min())
    mid_x = 0.5 * (float(contour[:, 0].min()) + float(contour[:, 0].max()))

    brow_center = pts[np.concatenate([bx_l, bx_r])].mean(axis=0)
    el = pts[ey_l]
    er = pts[ey_r]
    eye_center = 0.5 * (el.mean(axis=0) + er.mean(axis=0))
    e_w_l = float(el[:, 0].max() - el[:, 0].min())
    e_w_r = float(er[:, 0].max() - er[:, 0].min())
    e_h_l = float(el[:, 1].max() - el[:, 1].min())
    e_h_r = float(er[:, 1].max() - er[:, 1].min())
    outer_l = el[np.argmin(el[:, 0])]
    outer_r = er[np.argmax(er[:, 0])]
    d_eyes = float(np.linalg.norm(outer_l - outer_r))
    inner_l = el[np.argmax(el[:, 0])]
    inner_r = er[np.argmin(er[:, 0])]
    inter_eye = float(np.linalg.norm(inner_l - inner_r))

    nose = pts[groups["nose"]]
    nose_base = nose[np.argmax(nose[:, 1])]
    ny = nose[:, 1]
    cut = ny.max() - NOSE_LOW_FRAC * (ny.max() - ny.min())
    low = nose[ny >= cut]
    nose_w = float(low[:, 0].max() - low[:, 0].min())
    mouth = pts[groups["mouth"]]
    lip_top = mouth[np.argmin(mouth[:, 1])]
    mouth_w = float(mouth[:, 0].max() - mouth[:, 0].min())

    up_mid = float(np.linalg.norm(brow_center - nose_base))
    lip_chin = float(np.linalg.norm(lip_top - chin))
    low_face = float(np.linalg.norm(nose_base - chin))
    philtrum = float(np.linalg.norm(nose_base - lip_top))
    jaw_h = float(np.linalg.norm(brow_center - chin))

    nbin = 8
    ys = contour[:, 1]
    lo, hi = ys.min(), ys.max()
    bid = np.clip(((ys - lo) / (hi - lo + EPS) * nbin).astype(int), 0, nbin - 1)
    gaps = []
    for b in range(nbin):
        sel = contour[bid == b]
        if len(sel) >= 2:
            gaps.append(abs((mid_x - sel[:, 0].min()) - (sel[:, 0].max() - mid_x)))
    sym = float(np.mean(gaps)) / (face_w + EPS) if gaps else 0.0

    return np.array([
        up_mid / (low_face + EPS),
        philtrum / (lip_chin + EPS),
        0.5 * (e_w_l + e_w_r) / (d_eyes + EPS),
        inter_eye / (d_eyes + EPS),
        sym,
        nose_w / (d_eyes + EPS),
        mouth_w / (nose_w + EPS),
        jaw_h / (d_eyes + EPS),
        face_w / (d_eyes + EPS),
        mouth_w / (d_eyes + EPS),
        e_h_l / (e_w_l + EPS),
        e_h_r / (e_w_r + EPS),
        float(np.linalg.norm(eye_center - brow_center)) / (d_eyes + EPS),
        face_w / (jaw_h + EPS),
    ], dtype=np.float32)

