"""量"锁头"效果（probe_lock.py）
====================================================================
在**输出视频**上重新跑人脸检测，量头部位置漂移。

为什么要在输出上量：
    "锁头运镜"的定义是「头部在画面里固定不动」。
    这个量只能在成片上测 —— 裁切参数（alpha/dead-zone）只是手段，
    不是结果。参数看着合理不代表头真的锁住了。

指标：
    水平/垂直漂移 = 人脸中心位置的标准差（像素，按输出宽度归一化到 1080 宽）
    越小越"锁得住"。理想 <1% 画面宽。
"""
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import face_crop as fc       # noqa: E402
import mediapipe as mp       # noqa: E402
from mediapipe.tasks import python as mp_python      # noqa: E402
from mediapipe.tasks.python import vision            # noqa: E402


def measure(video):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=fc.MODEL_FACE),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    lm = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    xs, ys, hs = [], [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fh, fw = f.shape[:2]
        sc = fc.DETECT_WIDTH / fw
        det = cv2.resize(f, (fc.DETECT_WIDTH, int(fh * sc)))
        rgb = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
        r = lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(i * 1000 / fps))
        if r and r.face_landmarks:
            bb = fc.face_bbox(r.face_landmarks[0], det.shape[1], det.shape[0])
            if bb:
                x, y, bw, bh = (v / sc for v in bb)
                xs.append((x + bw / 2) / fw)      # 归一化到画面宽
                ys.append((y + bh / 2) / fh)      # 归一化到画面高
                hs.append(bh / fh)
        i += 1
    cap.release()
    lm.close()
    if len(xs) < 10:
        return None
    xs, ys, hs = np.asarray(xs), np.asarray(ys), np.asarray(hs)
    return {
        "n": len(xs), "total": i,
        "cx": float(xs.mean()), "cy": float(ys.mean()),
        "sx": float(xs.std()) * 100,          # % 画面宽
        "sy": float(ys.std()) * 100,          # % 画面高
        "range_x": float(xs.max() - xs.min()) * 100,
        "range_y": float(ys.max() - ys.min()) * 100,
        "h": float(hs.mean()) * 100,          # 头占画面高的比例
        "sx_px": float(xs.std()) * 1080,      # 换算到 1080 宽的像素
        "sy_px": float(ys.std()) * 1920,
    }


CASES = [
    ("output/_lock_完全.mp4", "完全锁头 alpha=1.0 dz=0"),
    ("output/_lock_适中.mp4", "适中锁头 alpha=0.35 dz=0"),
    ("output/_fc_证件照宽.mp4", "当前默认 alpha=0.12 dz=6"),
    ("output/新素材_人脸特写.mp4", "旧取景 zoom=1.8"),
]

print(f"{'配置':<30}{'头占画面':>9}{'水平漂移':>11}{'垂直漂移':>11}"
      f"{'水平范围':>10}{'垂直范围':>10}")
print("-" * 84)
for v, lab in CASES:
    p = os.path.join(BASE, v)
    if not os.path.exists(p):
        print(f"{lab:<30}  (缺文件)")
        continue
    m = measure(p)
    if m is None:
        print(f"{lab:<30}  检测失败")
        continue
    print(f"{lab:<30}{m['h']:>8.1f}%{m['sx_px']:>9.1f}px{m['sy_px']:>9.1f}px"
          f"{m['range_x']*10.8:>8.0f}px{m['range_y']*19.2:>8.0f}px")
print()
print("水平/垂直漂移 = 人脸中心的标准差（换算到 1080x1920 画布）")
print("'锁头'理想值：标准差 < 10px（<1% 画面宽）")
