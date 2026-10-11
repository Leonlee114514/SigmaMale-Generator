"""暖色矩形接缝检测（probe_warm.py）
====================================================================
量化 `render_mode` 里 sigma 模式那块**暖色矩形**的可见程度。

背景
----
`render_mode` 的 sigma 分支做的是：

    warm = warmify(face_crop)          # R x1.27 / B x0.78 的暖色 LUT
    dark[y:y+bh, x:x+bw] = warm        # ← 硬边矩形直接贴上去

人脸 bbox 的边界上是一个**突变**（暖色偏移量级 15~30/255），
所以画面里能看出一条矩形接缝 —— 脸小时不明显，放大成人脸特写就很突兀。

这条接缝不必靠眼睛看。它有一个硬指标：**跨 bbox 边界的相邻像素色差**。

    自然画面跨这条线也有梯度，所以要和**同一帧、同一位置**的输入图对比：
    输入图的边界台阶 = 自然梯度；渲染图的边界台阶 = 自然梯度 + 人为接缝。
    两者之差，就是人为接缝的强度。

输入刻意选**参考片**（`2022年的剪辑 belike.mp4`）—— 它从没经过我们的管线，
是真正干净的底子。用自己录的素材会自欺：那些素材本身就带这块矩形。

用法：
    python probe_warm.py                     # 默认跑参考片
    python probe_warm.py --tag before        # 存图加后缀，便于前后对照
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import mediapipe as mp                                         # noqa: E402
import sigma_realtime as sr                                    # noqa: E402
from mediapipe.tasks import python as mp_python                # noqa: E402
from mediapipe.tasks.python import vision                      # noqa: E402

REFOUT = os.path.join(BASE, "output", "_probe_warm")


def build_landmarkers():
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=sr.MODEL_FACE),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False))
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=sr.MODEL_POSE),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return face, pose


def grab(path, t):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def detect_bbox(face_lm, state, frame):
    """在**已经镜像过**的帧上检测，拿到的 bbox 才和 render_mode 用的一致
    （process_frame 内部第一步就是 cv2.flip）。"""
    state["ts_ms"] = max(state["ts_ms"] + 1, int(time.monotonic() * 1000))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = face_lm.detect_for_video(img, state["ts_ms"])
    if not (res and res.face_landmarks):
        return None
    return sr.face_bbox(res.face_landmarks[0], frame.shape[1], frame.shape[0])


def border_step(img, bbox):
    """跨 bbox 四条边的相邻像素色差（三通道取最大）。

    这就是"接缝强度"的直接度量：硬边贴片在边界上会留下一个大台阶。
    每条边只取中间 80%，避开四个角（角上是两条边交叠，读不出单一台阶）。
    """
    x, y, bw, bh = bbox
    H, W = img.shape[:2]
    f = img.astype(np.int16)
    vals = []
    y0, y1 = max(0, y + int(bh * 0.1)), min(H, y + int(bh * 0.9))
    x0, x1 = max(0, x + int(bw * 0.1)), min(W, x + int(bw * 0.9))
    if y1 > y0:                       # 左右两条边
        for xc, xo in ((x, x - 1), (x + bw - 1, x + bw)):
            if 0 <= xc < W and 0 <= xo < W:
                vals.append(np.abs(f[y0:y1, xc] - f[y0:y1, xo]).max(axis=1))
    if x1 > x0:                       # 上下两条边
        for yc, yo in ((y, y - 1), (y + bh - 1, y + bh)):
            if 0 <= yc < H and 0 <= yo < H:
                vals.append(np.abs(f[yc, x0:x1] - f[yo, x0:x1]).max(axis=1))
    if not vals:
        return 0.0
    return float(np.concatenate([v.astype(np.float32) for v in vals]).mean())


def global_step(img):
    """全图平均相邻像素色差 —— "这份画面本身的自然梯度有多大"的参照。"""
    f = img.astype(np.int16)
    dh = np.abs(f[:, 1:] - f[:, :-1]).max(axis=2).mean()
    dv = np.abs(f[1:, :] - f[:-1, :]).max(axis=2).mean()
    return float((dh + dv) / 2)


ROWS = [("输入 参考片原帧", "绝对干净，作为基准"),
        ("debug + bare", "= record_auto.py --no-grade 录到的"),
        ("sigma + bare", "= 默认模式录到的")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="取帧来源视频（默认参考片）")
    ap.add_argument("--times", default="14.78,21.0,5.0,24.0,11.0")
    ap.add_argument("--tag", default="now", help="输出图后缀，如 before/after")
    a = ap.parse_args()

    src = a.src or os.path.join(os.path.expanduser("~"), "Desktop",
                                "2022年的剪辑 belike.mp4")
    if not os.path.exists(src):
        print(f"取帧来源不存在：{src}")
        return 1
    os.makedirs(REFOUT, exist_ok=True)

    face_lm, pose_lm = build_landmarkers()
    state = sr.new_state()

    print("挑一帧能检出人脸、且脸不至于太小的：")
    picked = None
    for t in [float(x) for x in a.times.split(",")]:
        fr = grab(src, t)
        if fr is None:
            print(f"  t={t:6.2f}  读不到帧")
            continue
        flipped = cv2.flip(fr, 1)          # process_frame 内部第一步就是镜像
        bbox = detect_bbox(face_lm, state, flipped)
        if not bbox:
            print(f"  t={t:6.2f}  未检出人脸")
            continue
        print(f"  t={t:6.2f}  bbox={bbox}  高 {bbox[3]}px")
        if bbox[3] >= 20:
            picked = (t, fr, flipped, bbox)
            break

    if picked is None:
        print("没找到可用的人脸帧（要求 bbox 高 >= 20px）")
        return 1

    t, fr, flipped, bbox = picked
    print(f"\n取样帧：{os.path.basename(src)} @ {t:.2f}s   "
          f"画幅 {flipped.shape[1]}x{flipped.shape[0]}   bbox={bbox}")

    imgs = {"输入 参考片原帧": flipped}
    for mode, label in (("debug", "debug + bare"), ("sigma", "sigma + bare")):
        _, hf, hp, clean = sr.process_frame(
            fr, state, face_lm, pose_lm, mode, "probe", False, True)
        imgs[label] = clean

    print()
    print("=" * 76)
    print(f"{'对象':18}{'说明':34}{'边界台阶':>10}{'全图梯度':>10}")
    print("=" * 76)
    base = None
    for name, note in ROWS:
        img = imgs[name]
        s, g = border_step(img, bbox), global_step(img)
        if base is None:
            base, extra = s, "   ← 基准"
        else:
            extra = (f"   ← 人为接缝 +{s - base:.1f}" if s - base > 2
                     else "   ← 与输入同级(无接缝)")
        print(f"{name:18}{note:34}{s:>10.2f}{g:>10.2f}{extra}")
    print("=" * 76)

    d = int(np.abs(imgs["debug + bare"].astype(np.int16)
                   - flipped.astype(np.int16)).max())
    print(f"\n[恒等性] debug+bare 与输入的最大单像素差 = {d}  ->  "
          + ("✓ 逐像素完全一致（--no-grade 真的不录任何东西）" if d == 0
             else "✗ 仍有改动"))

    x, y, bw, bh = bbox
    inside = (slice(y, y + bh), slice(x, x + bw))
    dif = np.abs(imgs["sigma + bare"].astype(np.int16) - flipped.astype(np.int16))
    mask = np.ones(flipped.shape[:2], bool)
    mask[inside] = False
    print(f"[暖色量级] sigma 相对输入：bbox 内 {dif[inside].mean():5.1f}/255"
          f"   bbox 外 {dif[mask].mean():5.1f}/255"
          f"   （比值越大越像「贴了张暖色纸」）")

    pad = max(30, int(bh * 0.6))
    X0, Y0 = max(0, x - pad), max(0, y - pad)
    X1, Y1 = min(flipped.shape[1], x + bw + pad), min(flipped.shape[0], y + bh + pad)
    crop = (slice(Y0, Y1), slice(X0, X1))
    print()
    for name, _ in ROWS:
        key = name.split(" ")[0].replace("输入", "input")
        fn = os.path.join(REFOUT, f"{key}_{a.tag}.png")
        cv2.imwrite(fn, cv2.resize(imgs[name][crop], None, fx=3, fy=3,
                                   interpolation=cv2.INTER_NEAREST))
        print("  存图", os.path.relpath(fn, BASE))
    print(f"  （已放大 3 倍、裁到人脸周边，便于肉眼比对）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
