"""录制保真度探针（probe_raw.py）
====================================================================
回答一个问题：**哪条录制路径出来的画面，逐像素等于摄像头原画？**

为什么要单独有这么一个探针
--------------------------
2026-09-24 起 `bare` 不再抑制任何绘制（按用户要求回滚）——
各模式自己的框 / 标签 / 线框都被视为"效果本体"。于是本项目只剩
**两条路**，选错就白录：

    raw   —— **零处理**，逐像素等于摄像头采集帧（仅镜像）
    其余   —— 都带各自的框与标签，烧进像素、后期擦不掉

所以本探针的判据是：**只有 raw 应当是 0.00 / 0**。
其余模式非 0 属于**预期**（它们在画效果），不是 bug。

这个脚本把每条路径与"摄像头原画"的差异量出来，结论不靠肉眼。

度量方式
--------
`process_frame` 第一步就是 `cv2.flip`（镜像，符合照镜子的直觉），
所以"原画"在这里定义为**镜像后的采集帧**。逐像素比较：

    平均差 = 0 且 最大差 = 0   ->  逐像素恒等，就是原画
    否则                       ->  该模式对画面做了改动（调色）

输入用参考片（`2022年的剪辑 belike.mp4`）—— 它从没经过本项目管线，
是干净的底子；用自己录的素材会把"已经烧进去的东西"当成原画。

用法：
    python probe_raw.py                      # 默认跑参考片
    python probe_raw.py --src 某视频.mp4
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

REFOUT = os.path.join(BASE, "output", "_probe_raw")

# 要检查的路径：(mode, 备注)
PATHS = [
    ("raw",   "= --raw / --no-grade（唯一零处理，应恒等）"),
    ("debug", "= 检测可视化：青色线框 + 绿色骨架"),
    ("sigma", "= 默认：暖色方框 + 金角框 + 标语"),
    ("chad",  "= 方形黑白高对比 + 两行标语"),
    ("scan",  "= 绿调 + 网格 + 锁定框 + 读数"),
]


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


def has_face_on(face_lm, state, frame):
    """在**已镜像**的帧上检测 —— 和 process_frame 内部看到的是同一张图。"""
    state["ts_ms"] = max(state["ts_ms"] + 1, int(time.monotonic() * 1000))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = face_lm.detect_for_video(img, state["ts_ms"])
    return bool(res and res.face_landmarks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="取帧来源视频（默认参考片）")
    ap.add_argument("--times", default="14.78,21.0,5.0,24.0")
    ap.add_argument("--tag", default="now")
    a = ap.parse_args()

    src = a.src or os.path.join(os.path.expanduser("~"), "Desktop",
                                "2022年的剪辑 belike.mp4")
    if not os.path.exists(src):
        print(f"取帧来源不存在：{src}")
        return 1
    os.makedirs(REFOUT, exist_ok=True)

    face_lm, pose_lm = build_landmarkers()

    picked = None
    print("挑一帧能检出人脸的（有脸才走得到 sigma 的暖色分支）：")
    for t in [float(x) for x in a.times.split(",")]:
        fr = grab(src, t)
        if fr is None:
            print(f"  t={t:6.2f}  读不到帧")
            continue
        flipped = cv2.flip(fr, 1)
        ok_face = has_face_on(face_lm, sr.new_state(), flipped)
        print(f"  t={t:6.2f}  {'检出人脸' if ok_face else '未检出'}")
        if ok_face:
            picked = (t, fr, flipped)
            break

    if picked is None:
        print("没找到含人脸的帧，无法验证 sigma 分支")
        return 1

    t, fr, flipped = picked
    h, w = flipped.shape[:2]
    print(f"\n取样帧：{os.path.basename(src)} @ {t:.2f}s   {w}x{h}")

    # 每条路径都用**全新的 state**，避免上一路径的帧序影响本路径
    # （ts_ms 用 monotonic 兜底，所以重置 state 也不会让 MediaPipe 报时间戳回退）
    results = {}
    for mode, _ in PATHS:
        _, hf, hp, clean = sr.process_frame(
            fr, sr.new_state(), face_lm, pose_lm, mode, "probe", False, True)
        d = np.abs(clean.astype(np.int16) - flipped.astype(np.int16))
        results[mode] = (clean, float(d.mean()), int(d.max()))

    print()
    print("=" * 84)
    print("录制取到的画面  vs  摄像头原画（镜像后）")
    print("=" * 84)
    print(f"{'模式':8}{'平均差':>9}{'最大差':>9}   {'结论':14}说明")
    print("-" * 84)
    for mode, note in PATHS:
        _, mean_d, max_d = results[mode]
        if max_d == 0:
            verdict = "✓ 逐像素恒等"
        elif mean_d < 2 and max_d < 12:
            verdict = "≈ 几乎无差"
        else:
            verdict = "✗ 有调色"
        print(f"{mode:8}{mean_d:>9.2f}{max_d:>9d}   {verdict:14}{note}")
    print("=" * 84)

    raw_ok = results["raw"][2] == 0
    print()
    if raw_ok:
        print("结论：raw 是**唯一**零处理的模式，录到的画面 = 摄像头原画。")
        print("      其余模式（bare 也一样）都带上各自的框与标签 ——")
        print("      那是刻意的效果，不是 bug。要干净素材就用 raw。")
    else:
        print("⚠ raw 模式竟然不是恒等的，请检查 render_mode 的 raw 分支。")
    print()
    print("对照：sigma 平均差 %.1f（暖色方框 + 金角框 + 标语）、"
          % results["sigma"][1])
    print("      debug 平均差 %.1f（检测线框）。" % results["debug"][1])
    print("      → 回滚前 debug 在 bare 下是 0.00（那时它被当作原图路径），")
    print("        现在线框回来了所以非 0 —— 这正是回滚生效的证据。")

    # 存一张横排对照图：原画 / raw / sigma（取画面中部一条带，人一般在中部）
    mid = w // 2
    x0, x1 = max(0, mid - w // 6), min(w, mid + w // 6)
    y0, y1 = max(0, h // 6), min(h, h - h // 6)
    strip = [flipped, results["raw"][0], results["sigma"][0]]
    labels = ["CAMERA ORIGINAL", "MODE: RAW", "MODE: SIGMA"]
    tiles = []
    for im, lab in zip(strip, labels):
        tile = im[y0:y1, x0:x1].copy()
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 34), (0, 0, 0), -1)
        cv2.putText(tile, lab, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.hstack(tiles)
    fn = os.path.join(REFOUT, f"compare_{a.tag}.png")
    cv2.imwrite(fn, sheet)
    print("存图", os.path.relpath(fn, BASE), "(左→右：原画 / raw / sigma)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
