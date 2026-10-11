"""打印逐拍偏差表，看修复前后形态是否变化。"""
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import edit_slop as es       # noqa: E402

W, H = 160, 284


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


def table(v, label):
    ct = cuts_of(v)
    print(f"\n=== {label} ===")
    print(f"{'拍':>3}{'理论':>9}{'实测':>9}{'偏差ms':>9}{'帧':>5}")
    vals = []
    for k in range(23):
        t = es.DROP_T_VIDEO + k * es.PERIOD
        j = int(np.argmin(np.abs(ct - t)))
        dv = (ct[j] - t) * 1000
        vals.append(dv)
        print(f"{k:>3}{t:>9.3f}{ct[j]:>9.3f}{dv:>+9.1f}{round((ct[j]-t)*30):>5}")
    vals = np.asarray(vals)
    print(f"  中位 {np.median(vals):+.1f}ms  均值 {vals.mean():+.1f}ms")
    return vals


def summary(v, label):
    ct = cuts_of(v)
    vals = []
    for k in range(23):
        t = es.DROP_T_VIDEO + k * es.PERIOD
        j = int(np.argmin(np.abs(ct - t)))
        vals.append((ct[j] - t) * 1000)
    vals = np.asarray(vals)
    print(f"{label:<22} 切点 {len(ct):>3} 个   "
          f"中位 {np.median(vals):>+7.1f}ms   "
          f"均值 {vals.mean():>+7.1f}ms   "
          f"≤20ms {int((np.abs(vals)<=20).sum()):>2}/23")
    print(f"{'':22} 前 8 拍: " + ", ".join(f"{d:+.0f}" for d in vals[:8]))


print("=" * 92)
print("不同配置下的切点偏差")
print("=" * 92)
for v, lab in (("output/_bare.mp4", "① 完全无效果基线"),
               ("output/_sp.mp4", "② 空间模糊"),
               ("output/_fix2.mp4", "③ 空间模糊+入点对齐"),
               ("output/成片_剪辑感.mp4", "④ 旧版(tmix 4帧)")):
    p = os.path.join(BASE, v)
    if os.path.exists(p):
        summary(p, lab)
    else:
        print(f"{lab:<22} (缺文件 {v})")

