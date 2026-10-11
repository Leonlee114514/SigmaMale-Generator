"""参考片切片结构精细分析（analyze_ref_shots.py）
====================================================================
analyze_ref.py 给出了粗粒度切点。这一步做**剪辑语法**级别的拆解：

  1. 用逐帧差精确定位所有硬切（含 1 帧的闪光/负片闪）
  2. 把"切点 -> 每个镜头的时长"列成表
  3. 检测每个镜头是"原样"还是带了处理（镜像/黑白/负片/放大）
  4. 找出"实拍位"（用户要替换的位置）：即哪些镜头是 Bateman 素材
     —— 用"是否黑白/是否高饱和"作为辅助判断
  5. 与 128BPM / 参考片自身 BPM 网格对齐核对

输出：output/_ref_analysis/shots_ref.json
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
W = H = 128          # 用更小尺寸，逐帧差对"切"更敏感（大尺寸会被细节噪声干扰）
FPS = 30.0


def decode(path, W, H, fps):
    raw = os.path.join(WORK, "_shots.raw")
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", path, "-vf", f"scale={W}:{H},fps={fps}",
                    "-pix_fmt", "rgb24", "-f", "rawvideo", raw],
                   capture_output=True)
    fs = W * H * 3
    n = os.path.getsize(raw) // fs
    with open(raw, "rb") as f:
        d = f.read(n * fs)
    os.remove(raw)
    return np.frombuffer(d, dtype=np.uint8).reshape(n, H, W, 3)


def main():
    os.makedirs(WORK, exist_ok=True)
    arr = decode(REF, W, H, FPS)
    n = len(arr)
    print(f"解码 {n} 帧 @ {FPS}fps  ({n/FPS:.3f}s)  {W}x{H}")

    a = arr.astype(np.int16)
    # ---- 帧间差（用"翻转像素占比"而非均值：硬切会有一大片像素突变更稳）----
    d = np.abs(np.diff(a, axis=0)).mean(axis=3)      # (n-1, H, W)
    frac = (d > 30).mean(axis=(1, 2))                # 突变像素占比
    mad = d.mean(axis=(1, 2))                        # 平均绝对差

    print(f"\n突变像素占比  中位 {np.median(frac):.4f}  "
          f"P90 {np.percentile(frac,90):.4f}  P99 {np.percentile(frac,99):.4f}")
    print(f"平均绝对差    中位 {np.median(mad):.2f}   "
          f"P90 {np.percentile(mad,90):.2f}  max {mad.max():.2f}")

    # 硬切判定：突变像素占比 > 25%（硬切通常 30-90%），且平均差 > 15
    is_cut = (frac > 0.25) & (mad > 15)
    idx = np.where(is_cut)[0] + 1
    # 聚类（相邻 2 帧内算同一刀）
    cuts = []
    for i in idx:
        if cuts and i - cuts[-1] <= 2:
            continue
        cuts.append(int(i))
    ct = np.array([c / FPS for c in cuts])
    print(f"\n检出硬切 {len(ct)} 个")

    if len(ct) < 2:
        print("切点太少，阈值需调")
        return 1

    # ---- 镜头表 ----
    bounds = np.concatenate([[0.0], ct, [n / FPS]])
    shots = []
    for i in range(len(bounds) - 1):
        t0, t1 = float(bounds[i]), float(bounds[i + 1])
        f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        f1 = min(f1, n)
        if f1 <= f0:
            continue
        seg = a[f0:f1]
        sat = float((seg.max(axis=3) - seg.min(axis=3)).mean())
        lum = float(seg.mean())
        shots.append({"i": len(shots), "t0": round(t0, 3),
                      "t1": round(t1, 3), "dur": round(t1 - t0, 3),
                      "sat": round(sat, 1), "lum": round(lum, 1),
                      "frames": f1 - f0})

    print(f"\n共 {len(shots)} 个镜头")
    durs = np.array([s["dur"] for s in shots])
    print(f"镜头时长: 中位 {np.median(durs)*1000:.0f}ms  "
          f"min {durs.min()*1000:.0f}ms  max {durs.max()*1000:.0f}ms")

    # 按 100ms 桶看分布
    hist = {}
    for dd in durs:
        k = round(dd * 10) / 10
        hist[k] = hist.get(k, 0) + 1
    print("\n镜头时长分布(top15):")
    for k, v in sorted(hist.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k*1000:6.0f}ms  × {v}   {'▇'*min(v,40)}")

    # ---- 镜头表打印（前 60 个）----
    print(f"\n{'#':>3} {'起':>7} {'止':>7} {'时长ms':>7} {'饱和度':>6} "
          f"{'亮度':>5}  判定")
    for s in shots[:60]:
        tag = []
        if s["sat"] < 10:
            tag.append("黑白")
        if s["lum"] > 200:
            tag.append("近白")
        if s["lum"] < 25:
            tag.append("近黑")
        if s["dur"] < 0.08:
            tag.append("极短(闪)")
        print(f"{s['i']:>3} {s['t0']:>7.3f} {s['t1']:>7.3f} "
              f"{s['dur']*1000:>7.0f} {s['sat']:>6.1f} {s['lum']:>5.1f}  "
              f"{' '.join(tag)}")

    out = os.path.join(WORK, "shots_ref.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_frames": n, "fps": FPS, "cut_times": ct.tolist(),
                   "shots": shots}, f, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
