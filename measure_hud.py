"""量出录像里 HUD 元素的精确位置，判断人脸特写能否避开它们。

为什么必须量：
    HUD 是**烧进像素**的，无法擦除。唯一出路是"裁掉"。
    但裁多少才够？靠目测不准 —— 差几像素就会带进一条黄线或半个字。
    这里逐行统计"HUD 特征色"的像素数，定位精确边界。
"""
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
VID = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE, "output", "sigma_20260922_204319.mp4")

cap = cv2.VideoCapture(VID)
ok, frame = cap.read()
cap.release()
if not ok:
    print("读帧失败")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"{os.path.basename(VID)}  {w}x{h}")

b, g, r = frame[:, :, 0].astype(int), frame[:, :, 1].astype(int), frame[:, :, 2].astype(int)

# HUD 特征色（BGR）：
#   金色 SIGMA_GOLD=(60,190,250)  绿色 SCAN_GREEN=(150,255,120)  红色 REC
gold = (b > 180) & (g > 120) & (g < 220) & (r < 120)
green = (g > 200) & (b > 100) & (b < 200) & (r < 170)
red = (r > 150) & (g < 100) & (b < 100)
hud = gold | green | red

rows = hud.sum(axis=1)
print("\n有 HUD 像素的行（按行统计，只列 >2 像素的）：")
runs, start = [], None
for y in range(h):
    if rows[y] > 2 and start is None:
        start = y
    elif rows[y] <= 2 and start is not None:
        runs.append((start, y - 1, int(rows[start:y].max())))
        start = None
if start is not None:
    runs.append((start, h - 1, int(rows[start:].max())))
for a, c, mx in runs:
    print(f"  y {a:5d} - {c:5d}   最高 {mx:4d} 像素   (高 {c-a+1})")

print("\n各色分别统计：")
for name, m in (("金色", gold), ("绿色", green), ("红色", red)):
    rr = m.sum(axis=1)
    ys = np.nonzero(rr > 1)[0]
    if len(ys):
        print(f"  {name}: y {ys.min()} - {ys.max()}")
    else:
        print(f"  {name}: 无")

# 找"摄像头实际画面"的纵向范围（上下是模糊填充，用行方差判断）
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
var = gray.std(axis=1)
print("\n行方差（判断模糊填充 vs 实际画面，抽样）：")
for y in range(0, h, 60):
    print(f"  y={y:5d}  std={var[y]:6.1f}  {'实际画面' if var[y] > 12 else '模糊填充'}")
