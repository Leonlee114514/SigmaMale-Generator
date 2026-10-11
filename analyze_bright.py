"""逐帧亮度分析：定位成片里"全白/全暗"的时段，判断是否超出预期。

⚠ 为什么要用"测量"而不是"看缩略图"：
    缩略图是 2fps 采样，连续 3 格全白可能是 3 格采样点恰好都落在闪白上，
    也可能是真的有 1 秒白屏 —— 两者结论完全不同。
    必须逐帧量才知道。
"""
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))
import edit_slop as es       # noqa: E402

VID = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    BASE, "output", "成片_剪辑感.mp4")
W, H, FPS = 120, 213, 30.0

r = subprocess.run([es.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", VID,
                    "-vf", f"scale={W}:{H},format=gray",
                    "-f", "rawvideo", "-"],
                   capture_output=True, timeout=600)
n = len(r.stdout) // (W * H)
b = np.frombuffer(r.stdout[:n * W * H], dtype=np.uint8).reshape(n, W * H)
b = b.mean(axis=1)

print(f"{os.path.basename(VID)}  {n} 帧  {n/FPS:.2f}s")
print(f"亮度: 均值 {b.mean():.1f}  最小 {b.min():.1f}  最大 {b.max():.1f}")

# 全白段（>235）
white = b > 235
print(f"\n全白帧 {int(white.sum())} 个（{white.sum()/FPS:.2f}s）")
runs, start = [], None
for i, w in enumerate(white):
    if w and start is None:
        start = i
    elif not w and start is not None:
        runs.append((start, i - 1))
        start = None
if start is not None:
    runs.append((start, len(white) - 1))
print("  连续白段（时长 >0.1s 的才算异常）：")
bad = 0
for a, c in runs:
    d = (c - a + 1) / FPS
    flag = "  <-- 过长!" if d > 0.1 else ""
    if d > 0.1:
        bad += 1
    print(f"    {a/FPS:7.3f}s - {c/FPS:7.3f}s   {d*1000:6.0f}ms{flag}")
print(f"  过长白段数: {bad}（闪白设计为 0.05s，即 1-2 帧）")

# 全黑段
dark = b < 12
print(f"\n全黑帧 {int(dark.sum())} 个（{dark.sum()/FPS:.2f}s）")

# 按拍统计平均亮度（看每拍是否一致）
print("\n按拍的平均亮度（Drop 之后）：")
t = es.DROP_T_VIDEO
for k in range(8):
    a, c = int(t * FPS), int((t + es.PERIOD) * FPS)
    seg = b[a:c]
    if len(seg):
        print(f"  第 {int(t/es.PERIOD):2d} 拍  {seg.mean():6.1f}  "
              f"max {seg.max():5.0f}  (t={t:.3f}s)")
    t += es.PERIOD
