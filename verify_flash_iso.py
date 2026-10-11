"""隔离验证闪白时序：暗源 + 仅闪白滤镜，看哪些帧真的亮了。

为什么要隔离：在成片里检测闪白会被硬切的亮度跳变干扰。
用一个**恒定暗画面**做源，任何亮帧就只可能来自闪白，判据就干净了。
"""
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import edit_slop as es       # noqa: E402

OUTD = os.path.join(BASE, "output", "_fx")
os.makedirs(OUTD, exist_ok=True)
W, H = 160, 284
FPS = 30.0
PERIOD = es.PERIOD
DROP = es.DROP_T_VIDEO

# 恒定暗画面（亮度 ~40），这样只有闪白会让它变亮
src = os.path.join(OUTD, "dark.mp4")
subprocess.run([es.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", f"color=c=0x282828:s=320x240:r=30:d=12",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p", src], timeout=300)

for dur in (0.04, 0.05, 0.06):
    expr = (f"{0.85:g}*gte(t,{DROP:.4f})"
            f"*lt(mod(t,{PERIOD:.6f}),{dur})")
    dst = os.path.join(OUTD, "flashonly.mp4")
    if os.path.exists(dst):
        os.remove(dst)
    r = subprocess.run([es.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                        "-i", src, "-vf",
                        f"eq=brightness='{expr}':eval=frame",
                        "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "18", "-pix_fmt", "yuv420p", dst],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"dur={dur}: 渲染失败 {(r.stderr or '')[:100]}")
        continue

    # 解出逐帧亮度
    rr = subprocess.run([es.FFMPEG, "-hide_banner", "-loglevel", "error",
                         "-i", dst, "-vf", f"scale={W}:{H},format=gray",
                         "-f", "rawvideo", "-"], capture_output=True, timeout=300)
    n = len(rr.stdout) // (W * H)
    b = np.frombuffer(rr.stdout[:n * W * H], dtype=np.uint8)
    b = b.reshape(n, W * H).mean(axis=1)
    bright = [i for i, v in enumerate(b) if v > 150]

    # 聚成事件
    ev, cur = [], []
    for i in bright:
        if cur and i - cur[-1] <= 1:
            cur.append(i)
        else:
            if cur:
                ev.append(cur[0])
            cur = [i]
    if cur:
        ev.append(cur[0])

    print(f"\n=== 闪白窗口 {dur}s ===")
    print(f"  亮帧数 {len(bright)}  事件数 {len(ev)}")
    if len(ev) > 1:
        g = np.diff(np.asarray(ev)) / FPS
        print(f"  事件间距 均值 {g.mean()*1000:.1f}ms "
              f"(拍周期 {PERIOD*1000:.1f}ms)  最小 {g.min()*1000:.1f}ms "
              f"最大 {g.max()*1000:.1f}ms")
        bad = [(i, round(x * 1000, 1)) for i, x in enumerate(g)
               if abs(x - PERIOD) > 0.05]
        print(f"  间距异常(偏离>50ms)的个数: {len(bad)}  {bad[:6]}")
