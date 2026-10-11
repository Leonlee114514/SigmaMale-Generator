#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_lock.py —— 锁头效果的独立验收脚本。

对「原片 / 锁头后」各跑一遍人脸检测与光流，给出可核对的数字：
  * face_center_std_px       人脸中心在画面里的抖动标准差（越小＝锁得越稳）
  * face_center_range_px     人脸中心最大位移范围
  * lock_ratio               抖动下降比例（相对原片）
  * bg_flow_px               背景平均光流幅度（应该仍然很大：证明确实是背景在动，不是把画面冻住）
  * border_dark_delta        边框暗像素比例相对原片的变化（露黑边的代理指标）

用法：python measure_lock.py before.mp4 after.mp4 [--yunet-model 模型.onnx]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np


def make_detector(model, w, h, threshold=0.6):
    model = resolve_model(model)
    if model:
        det = cv2.FaceDetectorYN.create(model, "", (w, h), threshold, 0.3, 500)
        return ("yunet", det)
    if not hasattr(cv2, "CascadeClassifier"):
        raise SystemExit(
            "缺少人脸检测模型：请用 --yunet-model 指定 YuNet onnx，或设置环境变量 LOCKHEAD_YUNET。\n"
            "下载：https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    return ("haar", cv2.CascadeClassifier(path))


def resolve_model(path):
    """参数 → 脚本同目录 → 环境变量 LOCKHEAD_YUNET。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (path, os.path.join(here, "face_detection_yunet_2023mar.onnx"), os.environ.get("LOCKHEAD_YUNET")):
        if cand and os.path.isfile(cand):
            return cand
    return None


def detect(det, kind, frame):
    h, w = frame.shape[:2]
    if kind == "yunet":
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
        return (
            float(f[0] + f[2] / 2),
            float(f[1] + f[3] / 2),
            float(f[2]),
            float(f[3]),
            float((f[4] + f[6]) / 2),
            float((f[5] + f[7]) / 2),
        )
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    boxes = det.detectMultiScale(gray, 1.1, 5, minSize=(max(24, int(min(w, h) * 0.06)),) * 2)
    if len(boxes) == 0:
        return None
    x, y, bw, bh = max(boxes, key=lambda r: r[2] * r[3])
    return float(x + bw / 2), float(y + bh / 2), float(bw), float(bh), None, None


def analyse(path, model, flow_stride=3, flow_scale=0.5):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kind, det = make_detector(model, w, h)
    centers, sizes, eye_mids, shifts, dark_ratio = [], [], [], [], []
    prev_gray = None
    win = None
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hit = detect(det, kind, frame)
        if hit is not None:
            centers.append((hit[0], hit[1]))
            sizes.append((hit[2], hit[3]))
            if hit[4] is not None:
                eye_mids.append((hit[4], hit[5]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # 相位相关测全局位移：比稠密光流稳，且直接给出「背景整体动了多少」
            small_prev = cv2.resize(prev_gray, None, fx=flow_scale, fy=flow_scale, interpolation=cv2.INTER_AREA)
            small_now = cv2.resize(gray, None, fx=flow_scale, fy=flow_scale, interpolation=cv2.INTER_AREA)
            if win is None:
                win = cv2.createHanningWindow((small_now.shape[1], small_now.shape[0]), cv2.CV_32F)
            (sx, sy), _ = cv2.phaseCorrelate(np.float32(small_prev), np.float32(small_now), win)
            shifts.append((sx / max(flow_scale, 1e-6), sy / max(flow_scale, 1e-6)))
        b = max(2, int(min(w, h) * 0.02))
        ring = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
        dark_ratio.append(float(np.mean(ring < 8)))
        prev_gray = gray
        frame_id += 1
    cap.release()

    m = np.array(shifts, dtype=np.float64) if shifts else np.zeros((1, 2))
    e = np.array(eye_mids, dtype=np.float64) if eye_mids else None
    c = np.array(centers, dtype=np.float64)
    s = np.array(sizes, dtype=np.float64)
    if len(c) == 0:
        return {"path": os.path.abspath(path), "frames": frame_id, "detected": 0}
    return {
        "path": os.path.abspath(path),
        "frames": frame_id,
        "detected": int(len(c)),
        "hit_rate": round(len(c) / max(frame_id, 1), 4),
        "face_center_std_px": round(float(np.mean(np.std(c, axis=0))), 3),
        "face_center_range_px": round(float(max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))), 3),
        "face_width_std_px": round(float(np.std(s[:, 0])), 3),
        "eye_mid_std_px": None if e is None else round(float(np.mean(np.std(e, axis=0))), 3),
        "eye_mid_range_px": None if e is None else round(float(max(np.ptp(e[:, 0]), np.ptp(e[:, 1]))), 3),
        "bg_motion_median_px": round(float(np.mean(np.hypot(m[:, 0], m[:, 1]))), 3),
        "bg_jitter_px": round(float(np.mean(np.hypot(*(np.diff(m, axis=0).T)))), 3) if len(m) > 1 else 0.0,
        "border_dark_ratio": round(float(np.mean(dark_ratio)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="锁头效果验收：比较原片与锁头后的画面")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--yunet-model", default=None)
    ap.add_argument("--json", default=None, help="把结果写入 JSON")
    ap.add_argument("--flow-stride", type=int, default=3, help="每隔几帧算一次背景光流（越大越快）")
    args = ap.parse_args()

    b = analyse(args.before, args.yunet_model)
    a = analyse(args.after, args.yunet_model)
    ratio = None
    if b.get("face_center_std_px") and a.get("face_center_std_px") is not None:
        ratio = a["face_center_std_px"] / max(b["face_center_std_px"], 1e-9)
    out = {
        "before": b,
        "after": a,
        "lock_ratio": None if ratio is None else round(float(ratio), 4),
        "border_dark_delta": None
        if (b.get("border_dark_ratio") is None or a.get("border_dark_ratio") is None)
        else round(float(a["border_dark_ratio"] - b["border_dark_ratio"]), 4),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
