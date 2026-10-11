"""测量切点偏差与动态模糊帧数的关系。

背景：发现开动态模糊后切点整体晚 ~98ms，关掉降到 ~31ms。
这里把模糊帧数做成梯度，看延迟是否随帧数线性增长 ——
如果线性，说明是 tmix 的缓冲区深度直接造成的。
"""
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import edit_slop as es       # noqa: E402

W, H, FPS = 160, 284, 30.0


def cuts_of(v):
    r = subprocess.run(
        [es.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", v,
         "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, timeout=900)
    n = len(r.stdout) // (W * H)
    a = np.frombuffer(r.stdout[:n * W * H], dtype=np.uint8)
    a = a.reshape(n, W * H).astype(np.float32)
    d = np.abs(np.diff(a, axis=0)).mean(axis=1)
    cs = [i + 1 for i in range(len(d)) if d[i] > 18.0]
    m, cur = [], []
    for i in cs:
        if cur and i - cur[-1] <= 2:
            cur.append(i)
        else:
            if cur:
                m.append(int(np.mean(cur)))
            cur = [i]
    if cur:
        m.append(int(np.mean(cur)))
    return np.asarray(m) / FPS


def stats(v):
    ct = cuts_of(v)
    if len(ct) == 0:
        return None, 0
    devs = []
    hit = 0
    for k in range(20):
        t = es.DROP_T_VIDEO + k * es.PERIOD
        j = int(np.argmin(np.abs(ct - t)))
        devs.append((ct[j] - t) * 1000)
        if abs(ct[j] - t) < 0.04:
            hit += 1
    return float(np.median(devs)), hit


print(f"{'模糊帧数':>8}{'中位偏差':>13}{'40ms内命中':>13}")
print("-" * 36)
for n in (0, 2, 3, 4, 5):
    p = os.path.join(BASE, "output", f"_cb{n}.mp4")
    if not os.path.exists(p):
        continue
    md, hit = stats(p)
    if md is None:
        print(f"{n:>8}   无法测出切点")
    else:
        print(f"{n:>8}{md:>+12.1f}ms{hit:>10}/20")
