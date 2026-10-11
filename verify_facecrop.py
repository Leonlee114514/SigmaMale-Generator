"""人脸特写验收（verify_facecrop.py）
====================================================================
在**输出视频**上重新跑人脸检测，客观验收"人脸特写"是否真的成立了。

为什么不看观感：人脸特写出错时的表现是"画面里根本没有脸"，
但肉眼要一帧帧看才能确认。而重跑检测能量化：

  1. **人脸命中率** —— 有多少帧真的检测到脸。裁偏时这个数会暴跌。
  2. **人脸位置** —— 人脸框中心在画面里的归一化位置 (0~1)。
     期望落在中央区域；裁偏时会被推到边缘甚至出画。
  3. **人脸占画面高度比** —— 特写的意义所在。原始素材里只有约 16%，
     特写后应达到 30% 以上。
  4. **逐帧位移** —— 人脸在画面里的稳定性（锁头效果）。
     越小越稳；大说明锁头没生效或轨迹在抖。

这是"裁切链路"的端到端验收 —— 它不检查中间变量，只看最终画面里
脸到底在不在、稳不稳。中间量全对但最终画面错了的情况真实发生过
（见 face_crop.py 里"坐标放大 2 倍"与"反射填充不足"两个 bug）。

用法：
    python verify_facecrop.py output/新素材_人脸特写.mp4
    python verify_facecrop.py 输出.mp4 --expect-ratio 0.30
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FACE = os.path.join(BASE, "models", "face_landmarker.task")
DETECT_WIDTH = 640

import mediapipe as mp                                     # noqa: E402
from mediapipe.tasks import python as mp_python            # noqa: E402
from mediapipe.tasks.python import vision                  # noqa: E402


def face_bbox(lm, w, h):
    """478 点的包围盒（像素，相对传入尺寸）。"""
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    if not xs:
        return None
    x0 = int(max(0, min(xs) * w))
    y0 = int(max(0, min(ys) * h))
    x1 = int(min(w, max(xs) * w))
    y1 = int(min(h, max(ys) * h))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def main():
    ap = argparse.ArgumentParser(description="人脸特写客观验收")
    ap.add_argument("video")
    ap.add_argument("--expect-ratio", type=float, default=0.25,
                    help="期望的人脸占画面高度比下限（默认 0.25）")
    ap.add_argument("--min-hit", type=float, default=0.85,
                    help="期望的人脸命中率下限（默认 0.85）")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        print("文件不存在:", a.video)
        return 1
    if not os.path.exists(MODEL_FACE):
        print("模型缺失:", MODEL_FACE)
        return 1

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("打不开视频")
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 70)
    print(f"人脸特写验收: {os.path.basename(a.video)}")
    print("=" * 70)
    print(f"  {W}x{H}  {fps:.2f}fps  {n_total} 帧 ({n_total/max(fps,1e-6):.2f}s)")
    print()

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FACE),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    lmk = vision.FaceLandmarker.create_from_options(opts)

    cx_list, cy_list, bh_list = [], [], []
    hit = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > DETECT_WIDTH:
            sc = DETECT_WIDTH / w
            det = cv2.resize(frame, (DETECT_WIDTH, int(h * sc)))
        else:
            sc, det = 1.0, frame
        rgb = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lmk.detect_for_video(mp_img, int(idx * 1000 / fps))
        if res and res.face_landmarks:
            bb = face_bbox(res.face_landmarks[0], det.shape[1], det.shape[0])
            if bb:
                x, y, bw, bh = (v / sc for v in bb)
                hit += 1
                cx_list.append((x + bw / 2) / W)
                cy_list.append((y + bh / 2) / H)
                bh_list.append(bh / H)
        idx += 1
    cap.release()
    lmk.close()

    if idx == 0:
        print("  读到 0 帧")
        return 1

    n = len(cx_list)
    print(f"[1] 人脸命中率")
    print(f"    {hit}/{idx} = {hit/idx*100:.1f}%   "
          f"{'✓' if hit/idx >= a.min_hit else '✗ 低于下限 %.0f%%' % (a.min_hit*100)}")
    if n == 0:
        print("\n  完全没有检测到人脸 —— 裁切位置错了")
        return 2

    cx = np.array(cx_list)
    cy = np.array(cy_list)
    bh = np.array(bh_list)

    print(f"\n[2] 人脸位置（归一化 0~1，画面中心 0.5）")
    print(f"    水平 中位 {np.median(cx):.3f}  范围 {cx.min():.3f}-{cx.max():.3f}")
    print(f"    垂直 中位 {np.median(cy):.3f}  范围 {cy.min():.3f}-{cy.max():.3f}")
    in_frame = ((cx > 0.05) & (cx < 0.95) & (cy > 0.05) & (cy < 0.95)).mean()
    print(f"    落在画面内(0.05~0.95)的比例 {in_frame*100:.0f}%")
    center_ok = 0.25 < np.median(cx) < 0.75 and 0.20 < np.median(cy) < 0.80
    print(f"    位置合理: {'✓' if center_ok else '✗ 人脸偏离画面中央'}")

    print(f"\n[3] 人脸占画面高度比")
    print(f"    中位 {np.median(bh)*100:.1f}%   范围 {bh.min()*100:.1f}-{bh.max()*100:.1f}%")
    ratio_ok = np.median(bh) >= a.expect_ratio
    print(f"    {'✓' if ratio_ok else '✗'} 期望 ≥ {a.expect_ratio*100:.0f}%"
          f"（原始素材里约 16%）")

    print(f"\n[4] 人脸在画面里的稳定性（锁头效果）")
    d = np.hypot(np.diff(cx) * W, np.diff(cy) * H) if n > 1 else np.array([0.0])
    print(f"    逐帧位移 中位 {np.median(d):.2f}px  最大 {d.max():.2f}px")
    print(f"    {'✓ 稳定' if np.median(d) < 3 else '⚠ 偏抖（>3px）'}")

    ok = (hit / idx >= a.min_hit) and center_ok and ratio_ok
    print("\n" + "=" * 70)
    print("验收结果:", "通过" if ok else "未通过")
    print("=" * 70)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
