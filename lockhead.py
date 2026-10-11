#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lockhead.py —— 锁头运镜（head-lock）Python 实现。

把画面中的头部固定在指定位置，背景自然绕着头转。四步流水线：
  1) 逐帧估计头部状态：中心 p_t、尺度 s_t、倾角 theta_t；
  2) 轨迹后处理：野值剔除 + 缺失帧插值 + 高斯平滑（离线对称核，无滞后）；
  3) 合成相似变换（平移 + 旋转 + 缩放），把头部映到锚点，并整体乘以放大量 k；
     k 由「覆盖条件」闭式解出，保证旋转/位移后画面仍填满画布（不露黑边）；
  4) cv2.warpAffine 逐帧重采样写出。

与普通防抖的区别：防抖稳定「场景」，背景不动；锁头稳定「头」，所以背景会动。
依赖：numpy + opencv-python(>=4.5)。人脸检测可选 YuNet onnx 模型，缺失时退回 Haar。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import cv2
import numpy as np


# ------------------------------------------------------------------ 数据结构


@dataclass
class Track:
    """逐帧头部状态；缺失帧为 NaN。"""

    cx: np.ndarray
    cy: np.ndarray
    scale: np.ndarray
    theta: np.ndarray
    box: np.ndarray
    valid: np.ndarray

    def __len__(self) -> int:
        return len(self.cx)


def empty_track(n: int) -> Track:
    n = max(int(n), 1)
    return Track(
        cx=np.full(n, np.nan),
        cy=np.full(n, np.nan),
        scale=np.full(n, np.nan),
        theta=np.full(n, np.nan),
        box=np.full((n, 4), np.nan),
        valid=np.zeros(n, dtype=bool),
    )


def slice_track(t: Track, n: int) -> Track:
    return Track(t.cx[:n], t.cy[:n], t.scale[:n], t.theta[:n], t.box[:n], t.valid[:n])


# ------------------------------------------------------------------ 头部检测


class FaceDetector:
    """逐帧人脸检测：优先 YuNet（带 5 点关键点，可给倾角），否则 Haar。"""

    def __init__(self, width, height, model=None, threshold=0.6, detect_scale=1.0, lock_point="auto"):
        self.lock_point = lock_point
        self.detect_scale = float(detect_scale)
        self.size = (max(2, int(width * self.detect_scale)), max(2, int(height * self.detect_scale)))
        self.kind = "none"
        self._yunet = None
        self._haar = None
        if model and os.path.isfile(model):
            try:
                self._yunet = cv2.FaceDetectorYN.create(model, "", self.size, threshold, 0.3, 500)
                self.kind = "yunet"
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] YuNet 加载失败({exc})，退回 Haar", file=sys.stderr)
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is None:
            raise RuntimeError(
                "当前 OpenCV 没有 Haar 级联检测器（OpenCV 5 已移除），请提供 YuNet 模型：\n"
                "  --yunet-model face_detection_yunet_2023mar.onnx\n"
                "下载：https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self._haar = cascade_cls(path)
        if self._haar.empty():
            raise RuntimeError(f"无法加载 Haar 级联：{path}")
        self.kind = "haar"
        self.min_face = max(24, int(min(self.size) * 0.06))

    def detect(self, frame):
        """返回 (cx, cy, scale, theta, score)，坐标还原到原图尺度；失败返回 None。"""
        inv = 1.0 / self.detect_scale
        small = frame if self.detect_scale == 1.0 else cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self._yunet is not None:
            self._yunet.setInputSize(self.size)
            _, faces = self._yunet.detect(small)
            if faces is None or len(faces) == 0:
                return None
            f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
            x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
            reye = (float(f[4]), float(f[5]))
            leye = (float(f[6]), float(f[7]))
            dx, dy = leye[0] - reye[0], leye[1] - reye[1]
            theta = math.atan2(dy, dx) if (dx or dy) else 0.0
            if self.lock_point == "eyes":
                cx, cy = (reye[0] + leye[0]) / 2, (reye[1] + leye[1]) / 2
            else:  # face（默认）：人脸框中心；实测比关键点更稳（见 README 实测表）
                cx, cy = x + w / 2, y + h / 2
            return cx * inv, cy * inv, w * inv, theta, float(f[14])
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        boxes = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face, self.min_face)
        )
        if len(boxes) == 0:
            return None
        x, y, w, h = max(boxes, key=lambda r: r[2] * r[3])
        return (x + w / 2) * inv, (y + h / 2) * inv, w * inv, 0.0, 1.0


class LKTracker:
    """光流 + 相似变换：跟踪「框里的东西」，不依赖人脸。适合背影、遮挡、非人脸主体。"""

    def __init__(self, box, max_points=150):
        self.box = tuple(float(v) for v in box)
        self.max_points = int(max_points)
        self.prev = None
        self.pts = None
        self.state = (self.box[0] + self.box[2] / 2.0, self.box[1] + self.box[3] / 2.0, self.box[2], 0.0)

    def _seed(self, gray):
        x, y, w, h = self.box
        mask = np.zeros(gray.shape, dtype=np.uint8)
        pad = 0.1
        x0, y0 = int(max(0, x - w * pad)), int(max(0, y - h * pad))
        x1, y1 = int(min(gray.shape[1], x + w * (1 + pad))), int(min(gray.shape[0], y + h * (1 + pad)))
        mask[y0:y1, x0:x1] = 255
        pts = cv2.goodFeaturesToTrack(gray, self.max_points, 0.01, 7, mask=mask)
        self.pts = pts
        return pts is not None and len(pts) >= 8

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev is None or self.pts is None or len(self.pts) < 8:
            ok = self._seed(gray)
            self.prev = gray
            return self.state if ok else None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, winSize=(21, 21), maxLevel=3)
        if nxt is None or status is None:
            self.prev = gray
            return None
        sel = status.ravel() == 1
        p0, p1 = self.pts[sel], nxt[sel]
        if len(p0) < 8:
            self.prev = gray
            self._seed(gray)
            return None
        mat, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        self.prev = gray
        if mat is None:
            self._seed(gray)
            return None
        a, b = float(mat[0, 0]), float(mat[1, 0])
        cx, cy, scale, prev_theta = self.state
        nx = a * cx + float(mat[0, 1]) * cy + float(mat[0, 2])
        ny = b * cx + float(mat[1, 1]) * cy + float(mat[1, 2])
        self.state = (nx, ny, max(scale * math.hypot(a, b), 1e-6), prev_theta + math.atan2(b, a))
        if inliers is not None:
            p1 = p1[inliers.ravel() == 1]
        if len(p1) < 8:
            self._seed(gray)
            return None
        self.pts = p1.reshape(-1, 1, 2).astype(np.float32)
        return self.state


def resolve_model(path):
    """按 参数 → 脚本同目录 → 环境变量 LOCKHEAD_YUNET 的顺序找 YuNet 模型。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path, os.path.join(here, "face_detection_yunet_2023mar.onnx"), os.environ.get("LOCKHEAD_YUNET")]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


# ------------------------------------------------------------------ 轨迹后处理


def median_filter(x, k=5):
    half = k // 2
    pad = np.pad(x, half, mode="edge")
    return np.array([np.median(pad[i : i + k]) for i in range(len(x))], dtype=np.float64)


def fill_clean(x, valid):
    """缺失帧线性插值，再做中值去野值。"""
    x = np.asarray(x, dtype=np.float64).copy()
    idx = np.arange(len(x))
    good = valid & np.isfinite(x)
    if good.sum() == 0:
        return np.zeros(len(x))
    if good.sum() < len(x):
        x[~good] = np.interp(idx[~good], idx[good], x[good])
    if len(x) >= 5:
        med = median_filter(x, 5)
        mad = float(np.median(np.abs(x - med))) + 1e-9
        x[np.abs(x - med) > 6.0 * 1.4826 * mad] = med[np.abs(x - med) > 6.0 * 1.4826 * mad]
    return x


def gaussian_smooth(x, sigma):
    """离线高斯平滑，对称核 → 无相位滞后。sigma 以帧为单位。"""
    if sigma <= 1e-6 or len(x) < 3:
        return np.asarray(x, dtype=np.float64)
    half = max(1, int(math.ceil(3.0 * sigma)))
    t = np.arange(-half, half + 1, dtype=np.float64)
    kern = np.exp(-0.5 * (t / sigma) ** 2)
    kern /= kern.sum()
    pad = np.pad(np.asarray(x, dtype=np.float64), half, mode="reflect")
    return np.convolve(pad, kern, mode="valid")


# ------------------------------------------------------------------ 变换求解


def cover_k(px, py, gain, theta, width, height, anchor):
    """最小放大量 k，使变换后画面仍完整覆盖画布（不露黑边）。

    前向映射：u → c + k*gain*R(theta)*(u - p)，c 为锚点。反解画布四角回源坐标，
    要求仍在 [0,W]×[0,H] 内，逐角点逐轴取 k 的下界。
    """
    if gain <= 1e-9:
        return 1.0
    c = np.array([width * anchor[0], height * anchor[1]], dtype=np.float64)
    ct, st = math.cos(-theta), math.sin(-theta)
    rot = np.array([[ct, -st], [st, ct]], dtype=np.float64)
    k = 1.0
    for corner in ((0, 0), (width, 0), (width, height), (0, height)):
        d = rot @ (np.array(corner, dtype=np.float64) - c)
        for axis in (0, 1):
            room = ((width - px) if d[axis] >= 0 else px) if axis == 0 else ((height - py) if d[axis] >= 0 else py)
            k = max(k, abs(d[axis]) / (max(room, 1e-6) * gain))
    return k


def build_matrix(px, py, gain, theta, k, width, height, anchor):
    """2x3 仿射矩阵（源 → 输出），把 (px,py) 精确映射到锚点。"""
    c = np.array([width * anchor[0], height * anchor[1]], dtype=np.float64)
    ct, st = math.cos(theta), math.sin(theta)
    lin = k * gain * np.array([[ct, -st], [st, ct]], dtype=np.float64)
    t = c - lin @ np.array([px, py], dtype=np.float64)
    return np.array([[lin[0, 0], lin[0, 1], t[0]], [lin[1, 0], lin[1, 1], t[1]]], dtype=np.float64)


# ------------------------------------------------------------------ 参数与主流程


def parse_anchor(text):
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("锚点格式：x,y（归一化，如 0.5,0.42）")
    return float(parts[0]), float(parts[1])


def parse_box(text):
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("框格式：x,y,w,h（像素）")
    return tuple(float(p) for p in parts)


def build_parser():
    p = argparse.ArgumentParser(
        description="锁头运镜：把画面中的头部固定在指定位置（Python 实现）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="输入视频")
    p.add_argument("output", help="输出视频")
    p.add_argument("--mode", choices=["auto", "face", "lk"], default="auto",
                   help="face=逐帧人脸检测；lk=光流跟踪框内目标；auto=优先人脸，命中率低时退回光流")
    p.add_argument("--box", type=parse_box, default=None, help="lk 模式起始框 x,y,w,h")
    p.add_argument("--yunet-model", default=None, help="YuNet onnx 路径（默认自动查找）")
    p.add_argument("--face-threshold", type=float, default=0.6, help="人脸检测置信度阈值")
    p.add_argument("--lock-point", choices=["face", "eyes"], default="face",
                   help="锁定点：face=人脸框中心（默认，实测更稳）；eyes=两眼中点（头大幅转动时更贴解剖位置）")
    p.add_argument("--detect-scale", type=float, default=1.0, help="检测缩放比例（<1 更快）")
    p.add_argument("--auto-min-hit", type=float, default=0.5, help="auto 模式人脸命中率下限")
    p.add_argument("--smooth", type=float, default=0.06,
                   help="轨迹平滑强度（秒），本工具的核心取舍旋钮：0=像素级锁死（背景会带检测抖动）；"
                        "0.05~0.1=只滤噪声（推荐）；0.3+=背景更顺但头会明显漂移")
    p.add_argument("--rot-strength", type=float, default=1.0, help="头部旋转补偿强度 0~1（对应剪映“旋转强度”）")
    p.add_argument("--size-strength", type=float, default=1.0, help="保持头部大小不变的强度 0~1（对应“保持主体大小不变”）")
    p.add_argument("--anchor", type=parse_anchor, default=(0.5, 0.5), help="头部落点（归一化 x,y）")
    p.add_argument("--margin", type=float, default=0.02, help="额外放大余量，防边界露边")
    p.add_argument("--max-zoom", type=float, default=3.0, help="放大量上限")
    p.add_argument("--out-scale", type=float, default=1.0, help="输出分辨率倍率")
    p.add_argument("--codec", default="mp4v", help="输出编码 fourcc")
    p.add_argument("--ffmpeg", default=None, help="ffmpeg 路径：存在则把原音轨合回输出")
    p.add_argument("--debug-dir", default=None, help="导出轨迹 CSV / 统计 JSON / 预览视频")
    p.add_argument("--quiet", action="store_true", help="少打印")
    return p


def scan_track(args, cap, width, height):
    """第一遍：扫描视频得到头部轨迹。返回 (Track, 方式说明)。"""
    n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    mode, used = args.mode, args.mode
    track = empty_track(n_hint)
    if mode in ("face", "auto"):
        det = FaceDetector(
            width, height, resolve_model(args.yunet_model), args.face_threshold, args.detect_scale, args.lock_point
        )
        used = f"face/{det.kind}"
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hit = det.detect(frame)
            if hit is not None:
                cx, cy, scale, theta, _ = hit
                track.cx[i], track.cy[i], track.scale[i], track.theta[i] = cx, cy, scale, theta
                track.box[i] = (cx - scale / 2, cy - scale / 2, scale, scale)
                track.valid[i] = True
            i += 1
            if i % 60 == 0:
                print(f"\r  检测中 {i} 帧 …", end="", file=sys.stderr, flush=True)
        track = slice_track(track, i)
        print(f"\r  人脸检测：{i} 帧，命中 {int(track.valid.sum())} 帧", file=sys.stderr)
        if mode == "face" or (i and track.valid.mean() >= args.auto_min_hit):
            return track, used
        print(f"  [auto] 命中率 {track.valid.mean():.0%} 偏低，改用光流跟踪", file=sys.stderr)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    seed = None
    if mode == "auto":
        idx = np.flatnonzero(track.valid)
        if len(idx):
            seed = tuple(track.box[idx[0]])
        else:
            # 整段都没检测到人脸：用画面中心的保守区域起跟，而不是直接报错
            seed = (width * 0.30, height * 0.25, width * 0.40, height * 0.50)
            print(
                "  [auto] 全程未检测到人脸，改用画面中心区域做光流跟踪；"
                "主体不在中心请用 --box 指定起始框",
                file=sys.stderr,
            )
    if seed is None:
        if args.box is None:
            raise SystemExit("lk 模式需要 --box x,y,w,h 指定起始框（或改用 --mode face）")
        seed = args.box
    tracker = LKTracker(seed)
    track = empty_track(n_hint)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        state = tracker.update(frame)
        if state is not None:
            cx, cy, scale, theta = state
            track.cx[i], track.cy[i], track.scale[i], track.theta[i] = cx, cy, scale, theta
            track.box[i] = (cx - scale / 2, cy - scale / 2, scale, scale)
            track.valid[i] = True
        i += 1
    track = slice_track(track, i)
    return track, (f"{used}→lk" if mode == "auto" else "lk")


def main(argv=None):
    args = build_parser().parse_args(argv)
    log = (lambda *a: None) if args.quiet else print

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"打不开输入视频：{args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if width == 0 or height == 0:
        raise SystemExit("读不到视频尺寸")
    log(f"输入 {width}x{height} @ {fps:.2f} fps")

    log("[1/3] 扫描头部轨迹 …")
    track, used = scan_track(args, cap, width, height)
    cap.release()
    if len(track) == 0:
        raise SystemExit("没有读到任何帧")
    if track.valid.mean() < 0.3:
        print(f"[warn] 跟踪命中率仅 {track.valid.mean():.0%}，结果可能不稳；可试 --box", file=sys.stderr)

    log("[2/3] 轨迹平滑与变换求解 …")
    valid = track.valid
    cx, cy = fill_clean(track.cx, valid), fill_clean(track.cy, valid)
    sc, th = fill_clean(track.scale, valid), fill_clean(track.theta, valid)
    sigma = max(0.0, args.smooth * fps)
    cx_s, cy_s = gaussian_smooth(cx, sigma), gaussian_smooth(cy, sigma)
    sc_s, th_s = gaussian_smooth(sc, sigma), gaussian_smooth(th, sigma)
    # 被平滑掉的部分 = 头在成片里残留的漂移；和背景抖动一起构成核心取舍
    drift = np.hypot(cx - cx_s, cy - cy_s)

    ref_scale = float(np.median(sc_s))
    ref_theta = float(np.median(th_s))
    if not np.isfinite(ref_scale) or ref_scale <= 0:
        raise SystemExit("头部尺度估计异常，跟踪可能完全失败（试试 --box 指定起始框）")
    gains = (ref_scale / sc_s) ** args.size_strength
    thetas = -args.rot_strength * (th_s - ref_theta)

    need_k = max(
        cover_k(cx_s[i], cy_s[i], gains[i], thetas[i], width, height, args.anchor) for i in range(len(track))
    )
    k = min(need_k * (1.0 + args.margin), args.max_zoom)
    if need_k * (1.0 + args.margin) > args.max_zoom:
        print(f"[warn] 需要约 {need_k:.2f}x 放大，已钳制到 {args.max_zoom}x；可能出现黑边", file=sys.stderr)
    log(f"      方式 {used}｜命中率 {valid.mean():.0%}｜内推放大 {k:.3f}x")

    matrices = [
        build_matrix(cx_s[i], cy_s[i], gains[i], thetas[i], k, width, height, args.anchor)
        for i in range(len(track))
    ]
    out_w, out_h = max(2, int(round(width * args.out_scale))), max(2, int(round(height * args.out_scale)))
    if (out_w, out_h) != (width, height):
        s = np.diag([out_w / width, out_h / height, 1.0])
        matrices = [s @ m for m in matrices]

    log("[3/3] 逐帧重采样写出 …")
    cap = cv2.VideoCapture(args.input)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*args.codec), fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"无法创建输出视频（编码 {args.codec} 不可用？）：{args.output}")
    preview = None
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        preview = cv2.VideoWriter(
            os.path.join(args.debug_dir, "preview.mp4"), cv2.VideoWriter_fourcc(*args.codec), fps, (width, height)
        )

    rows, residuals, border_bad, i = [], [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(matrices):
            break
        m = matrices[i]
        writer.write(cv2.warpAffine(frame, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        p = m @ np.array([cx_s[i], cy_s[i], 1.0])
        residual = float(np.hypot(p[0] - out_w * args.anchor[0], p[1] - out_h * args.anchor[1]))
        residuals.append(residual)
        inv = cv2.invertAffineTransform(m)
        corners = np.array([[0, 0, 1], [out_w, 0, 1], [out_w, out_h, 1], [0, out_h, 1]], dtype=np.float64).T
        src = (inv @ corners).T
        inside = bool(
            np.all(src[:, 0] >= -0.5) and np.all(src[:, 0] <= width + 0.5)
            and np.all(src[:, 1] >= -0.5) and np.all(src[:, 1] <= height + 0.5)
        )
        border_bad += 0 if inside else 1
        rows.append([i, cx[i], cy[i], sc[i], th[i], cx_s[i], cy_s[i], sc_s[i], th_s[i], gains[i], thetas[i], residual, 0 if inside else 1, m[0, 2], m[1, 2]])
        if preview is not None:
            vis = frame.copy()
            if valid[i]:
                bx, by, bw, bh = track.box[i]
                cv2.rectangle(vis, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 255, 0), 2)
            cv2.drawMarker(vis, (int(width * args.anchor[0]), int(height * args.anchor[1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(vis, f"frame {i}  hit={'Y' if valid[i] else 'N'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            preview.write(vis)
        i += 1
    cap.release()
    writer.release()
    if preview is not None:
        preview.release()

    stats = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "frames": i,
        "fps": fps,
        "out_size": [out_w, out_h],
        "mode": used,
        "hit_rate": float(valid.mean()),
        "smooth_seconds": args.smooth,
        "smooth_seconds": args.smooth,
        "anchor": list(args.anchor),
        "punch_in_k": k,
        "needed_k": float(need_k),
        "max_lock_residual_px": float(np.max(residuals)) if residuals else 0.0,
        "frames_outside_source": int(border_bad),
        "head_travel_px": float(np.hypot(cx - np.median(cx), cy - np.median(cy)).max()) if i else 0.0,
        "head_drift_std_px": float(np.std(drift)),
        "head_drift_max_px": float(np.max(drift)),
        "bg_jitter_px": float(np.mean(np.abs(np.diff([r[13] for r in rows])))) if len(rows) > 2 else 0.0,
    }
    if args.debug_dir:
        with open(os.path.join(args.debug_dir, "track.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "cx_raw", "cy_raw", "scale_raw", "theta_raw", "cx_s", "cy_s", "scale_s", "theta_s", "gain", "theta_applied", "lock_residual_px", "outside_source", "tx", "ty"])
            w.writerows(rows)
        with open(os.path.join(args.debug_dir, "stats.json"), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)
    if args.ffmpeg and shutil.which(args.ffmpeg):
        tmp = args.output + ".mux.mp4"
        subprocess.run([args.ffmpeg, "-y", "-i", args.output, "-i", args.input, "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-shortest", tmp],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.isfile(tmp):
            os.replace(tmp, args.output)
    log(f"完成：{i} 帧 → {args.output}\n      锁定残差 max {stats['max_lock_residual_px']:.4f} px｜越界帧 {border_bad}｜头部原始位移 {stats['head_travel_px']:.1f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
