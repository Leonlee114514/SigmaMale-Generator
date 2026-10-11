"""对比修复前后的切点精度（三步递进）。

① 原始：tmix 时间混合 + 入点不对齐帧网格
② 只换空间模糊
③ 再叠加入点对齐
"""
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import edit_slop as es       # noqa: E402

W, H = 160, 284
N_BEATS = 23


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
    return np.asarray(m) / 30.0


def devs(v):
    ct = cuts_of(v)
    out = []
    for k in range(N_BEATS):
        t = es.DROP_T_VIDEO + k * es.PERIOD
        j = int(np.argmin(np.abs(ct - t)))
        out.append((ct[j] - t) * 1000)
    return np.asarray(out)


CASES = [
    ("output/成片_剪辑感.mp4", "① 原始（tmix + 入点不对齐）"),
    ("output/_sp.mp4",         "② 换空间模糊"),
    ("output/_fix2.mp4",       "③ ②+入点对齐帧网格"),
]

print(f"{'版本':<32}{'中位':>9}{'均值':>9}{'≤20ms':>9}{'≤35ms':>9}")
print("-" * 70)
for v, lab in CASES:
    p = os.path.join(BASE, v)
    if not os.path.exists(p):
        print(f"{lab:<32}  (缺文件)")
        continue
    d = devs(p)
    print(f"{lab:<32}{np.median(d):>+8.1f}ms{d.mean():>+8.1f}ms"
          f"{int((np.abs(d) <= 20).sum()):>6}/{N_BEATS}"
          f"{int((np.abs(d) <= 35).sum()):>6}/{N_BEATS}")
print()
print("参考：切点只能落在 33ms 的帧网格上，所以 |偏差| ≤ 17ms 是理论最优")
