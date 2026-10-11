"""找出"实际画面 vs 模糊填充"的可靠判据。

试过逐行标准差（std），失败：模糊填充是**同一场景的模糊版**，
仍保留低频结构（渐变、大色块），行 std 可高达 38，
和实际画面的 20-40 完全重叠，区分不开。

模糊的物理本质是**高频被低通滤掉**。所以判据应该用高频能量：
    · 逐行 Laplacian 方差
    · 或 逐行 |原图 - 高斯模糊| 的均值
两者都只对"细节"敏感，对低频渐变不敏感。
"""
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
VID = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE, "output", "sigma_20260922_204319.mp4")

cap = cv2.VideoCapture(VID)
ok, f = cap.read()
cap.release()
if not ok:
    print("读帧失败")
    sys.exit(1)

g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
h = g.shape[0]

# 判据 A：逐行标准差（已知失败，作对照）
std_a = g.std(axis=1)

# 判据 B：逐行 Laplacian 方差
lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
lap_b = lap.var(axis=1)

# 判据 C：逐行高频能量 = |原图 - 高斯模糊|
blur = cv2.GaussianBlur(g, (0, 0), 3.0)
hf_c = np.abs(g - blur).mean(axis=1)

print(f"{VID.split(os.sep)[-1]}  高 {h}")
print(f"\n{'y':>6} {'std':>8} {'lapVar':>10} {'HF':>8}   判定")
print("-" * 48)
for y in range(0, h, 40):
    a, b, c = std_a[y], lap_b[y], hf_c[y]
    verdict = "画面" if b > 20 else "模糊"
    print(f"{y:6d} {a:8.1f} {b:10.1f} {c:8.2f}   {verdict}")

# 用 Laplacian 方差定位边界
thr = 20.0
rows = np.nonzero(lap_b > thr)[0]
print(f"\n[Laplacian 判据 thr={thr}] 实际画面 y ∈ "
      f"[{rows.min() if len(rows) else '-'}, {rows.max() if len(rows) else '-'}]")

# 用 HF 能量定位
thr2 = hf_c.max() * 0.25
rows2 = np.nonzero(hf_c > thr2)[0]
print(f"[HF 判据 thr={thr2:.2f}] 实际画面 y ∈ "
      f"[{rows2.min() if len(rows2) else '-'}, {rows2.max() if len(rows2) else '-'}]")

print(f"\n参照：之前用 std>10 得到 [542, 1766]（过宽，已知失败）")
