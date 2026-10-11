"""参考片快剪段逐镜特征提取（analyze_ref_shots_detail.py）
====================================================================
要"照搬参考片的黑白位置"，就必须精确知道**每一镜**是黑白还是彩色。
analyze_ref_shots.py 用的是整镜平均饱和度，会把"半黑白"的镜糊掉。

这一步逐镜做更细的判断：
  1. 每镜逐帧算饱和度，看它的**时间分布**（是否有中途切黑白）
  2. 每镜的亮度分布（判断有无闪白/负片闪）
  3. 用更大的采样尺寸（256）让饱和度判断更准

输出：output/_ref_analysis/shots_detail.json
"""
import json
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
REF = r"C:\Users\Lyonl\Desktop\2022年的剪辑 belike.mp4"
WORK = os.path.join(BASE, "output", "_ref_analysis")
W = H = 256
FPS = 30.0


def decode(path, W, H, fps):
    raw = os.path.join(WORK, "_detail.raw")
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", path, "-vf", f"scale={W}:{H},fps={fps}",
                    "-pix_fmt", "rgb24", "-f", "rawvideo", raw],
                   capture_output=True)
    fs = W * H * 3
    n = os.path.getsize(raw) // fs
    with open(raw, "rb") as f:
        d = f.read(n * fs)
    os.remove(raw)
    return np.frombuffer(d, dtype=np.uint8).reshape(n, H, W, 3), fs


def main():
    arr, _ = decode(REF, W, H, FPS)
    n = len(arr)
    print(f"解码 {n} 帧 {W}x{H} @ {FPS}fps  ({n/FPS:.3f}s)")

    jp = os.path.join(WORK, "shots_ref.json")
    with open(jp, encoding="utf-8") as f:
        shots = json.load(f)["shots"]

    a = arr.astype(np.int16)
    print(f"\n共 {len(shots)} 镜，逐镜分析：\n")
    print(f"{'#':>3} {'起':>7} {'止':>7} {'ms':>6} {'饱和度中位':>9} "
          f"{'彩色帧%':>8} {'解码':>6}  特征")
    print("-" * 78)

    detailed = []
    for s in shots:
        f0 = int(round(s["t0"] * FPS))
        f1 = min(int(round(s["t1"] * FPS)), n)
        if f1 <= f0:
            continue
        seg = a[f0:f1]
        sat = (seg.max(axis=3) - seg.min(axis=3)).mean(axis=(1, 2))
        lum = seg.mean(axis=(1, 2, 3))
        med_sat = float(np.median(sat))
        # 彩色帧：饱和度 > 12（黑白素材压缩后仍有 3-8 的噪声底）
        color_ratio = float((sat > 12).mean())
        if med_sat > 14 and color_ratio > 0.7:
            cls = "彩色"
        elif med_sat < 9 and color_ratio < 0.25:
            cls = "黑白"
        else:
            cls = "混合"
        feats = []
        # 逐帧饱和度变化 -> 是否中途切黑白
        if 0.25 < color_ratio < 0.75:
            feats.append("中途变色")
        # 近白帧（闪白）
        nw = int((lum > 215).sum())
        if nw:
            feats.append(f"近白{nw}帧")
        # 近黑帧
        nb = int((lum < 20).sum())
        if nb > 1:
            feats.append(f"近黑{nb}帧")
        # 亮度跳变（负片闪会产生）
        if len(lum) > 2:
            jump = np.abs(np.diff(lum))
            if jump.max() > 45:
                feats.append(f"亮度跳变{jump.max():.0f}")

        print(f"{s['i']:>3} {s['t0']:>7.3f} {s['t1']:>7.3f} {s['dur']*1000:>6.0f} "
              f"{med_sat:>9.1f} {color_ratio*100:>7.0f}% {cls:>6}  "
              f"{' '.join(feats)}")

        detailed.append({**s, "med_sat": med_sat,
                         "color_ratio": color_ratio, "cls": cls,
                         "feats": feats})

    # ---- 快剪段（900ms 镜）的黑白序列 ----
    fast = [d for d in detailed if 0.85 < d["dur"] < 0.95]
    print(f"\n快剪段 {len(fast)} 镜的黑白序列（照搬用）:")
    seq = []
    for i, d in enumerate(fast):
        seq.append(d["cls"])
        print(f"  镜{i:>2}  {d['t0']:>7.3f}s  {d['cls']}  "
              f"(饱和度 {d['med_sat']:.1f})")
    print(f"\n  序列: {seq}")

    # 统计
    from collections import Counter
    print(f"  统计: {dict(Counter(seq))}")

    out = os.path.join(WORK, "shots_detail.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"shots": detailed, "fast_seq": seq, "fast": fast},
                  f, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
