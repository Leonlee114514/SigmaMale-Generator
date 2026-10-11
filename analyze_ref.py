"""参考片拆解器（analyze_ref.py）
====================================================================
把用户给的参考片彻底拆开，用于"完全复刻"：

  1. 元数据（尺寸/时长/帧率/音轨）
  2. 音频分析（BPM / Drop / 段落）—— 关键：这次参考片的 BGM 和我们用的一致吗？
  3. 画面结构分析：
     - 逐帧帧间差 -> 切点时刻表（硬切在哪、间隔多少）
     - 每段的长度分布（是不是也有"每拍一刀"的规律）
     - 黑白/彩色占比（是不是有黑白段）
     - 亮度（有没有闪白）
  4. 抽帧存图，便于人工核对画面内容

用法：
    python analyze_ref.py
    python analyze_ref.py --ref "C:/path/to/ref.mp4"
"""
import argparse
import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
DEFAULT_REF = r"C:\Users\Lyonl\Desktop\2022年的剪辑 belike.mp4"
WORK = os.path.join(BASE, "output", "_ref_analysis")


def probe(path):
    """读元数据。本机没有 ffprobe，只能解析 ffmpeg -i 的 stderr。
    ⚠ ffmpeg -i 的退出码总是 1，判定只能看内容。"""
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    txt = r.stderr or ""
    info = {"raw": [l.strip() for l in txt.splitlines()
                    if "Stream #" in l or "Duration" in l]}
    for line in txt.splitlines():
        if "Duration" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            info["duration"] = int(h) * 3600 + int(m) * 60 + float(s)
        if "Stream #" in line and "Video:" in line:
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                info["size"] = (int(m.group(1)), int(m.group(2)))
            fm = re.search(r"([\d.]+)\s*fps", line)
            if fm:
                info["fps"] = float(fm.group(1))
            info["vcodec"] = "h264" if "h264" in line else line.split(":")[-1][:40]
        info["has_audio"] = info.get("has_audio", False) or "Audio:" in line
    return info


def decode_raw(path, out_raw, W=180, H=180, fps=60):
    """解码成小尺寸原始 rgb24 裸流 —— 逐帧分析用这个，比抽 png 快得多。"""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", path,
           "-vf", f"scale={W}:{H},fps={fps}",
           "-pix_fmt", "rgb24", "-f", "rawvideo", out_raw]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_raw):
        print("解码失败:", r.stderr[-400:])
        return None
    return W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--fps", type=float, default=60.0,
                    help="分析用采样帧率（越高切点定位越准）")
    ap.add_argument("--frames-dir", action="store_true",
                    help="额外抽 PNG 帧供人眼核对")
    a = ap.parse_args()

    if not os.path.exists(a.ref):
        print("参考片不存在:", a.ref)
        return 1
    os.makedirs(WORK, exist_ok=True)

    print("=" * 66)
    print("参考片:", a.ref)
    print("=" * 66)
    info = probe(a.ref)
    for l in info["raw"]:
        print("  ", l)
    print(f"\n  时长 {info.get('duration', 0):.3f}s   分辨率 {info.get('size')}   "
          f"帧率 {info.get('fps')}   音轨 {info.get('has_audio')}")

    # ---- 解码成裸流 ----
    raw = os.path.join(WORK, "ref.raw")
    W = H = 180
    if not decode_raw(a.ref, raw, W, H, a.fps):
        return 1
    fsize = W * H * 3
    n = os.path.getsize(raw) // fsize
    print(f"\n  解码 {n} 帧 @ {a.fps}fps 用于逐帧分析")

    with open(raw, "rb") as f:
        data = f.read(n * fsize)

    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, H, W, 3).astype(np.int16)

    # ---- 逐帧帧间差 -> 切点 ----
    diff = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2, 3))
    print(f"\n  帧间差  中位 {np.median(diff):.2f}  均值 {diff.mean():.2f}  "
          f"P95 {np.percentile(diff, 95):.2f}  max {diff.max():.2f}")

    # 阈值取"中位数的若干倍"太松，用 P90 和绝对下限的组合
    thr = max(np.percentile(diff, 90), np.median(diff) * 3, 6.0)
    cuts = [i + 1 for i, d in enumerate(diff) if d > thr]
    # 合并相邻（同一刀可能跨 2 帧）
    merged = []
    for c in cuts:
        if merged and c - merged[-1] <= 1:
            merged[-1] = c
        else:
            merged.append(c)
    ct = [c / a.fps for c in merged]
    print(f"  检出切点 {len(ct)} 个（阈值 {thr:.2f}）")

    if len(ct) >= 2:
        gaps = np.diff(ct)
        print(f"\n  切点间隔: 中位 {np.median(gaps)*1000:.0f}ms  "
              f"min {gaps.min()*1000:.0f}ms  max {gaps.max()*1000:.0f}ms")
        # 间隔直方图（粗分桶 100ms）
        hist = {}
        for g in gaps:
            k = round(g * 10) / 10
            hist[k] = hist.get(k, 0) + 1
        print("  间隔分布（前 12 个最高频）:")
        for k, v in sorted(hist.items(), key=lambda x: -x[1])[:12]:
            print(f"    {k*1000:6.0f}ms  × {v}")

    # ---- 色彩/黑白分析 ----
    # 饱和度：max-min 通道均值。接近 0 = 黑白
    sat = (arr.max(axis=3) - arr.min(axis=3)).mean(axis=(1, 2))
    lum = arr.mean(axis=(1, 2, 3))
    bw_frames = int((sat < 10).sum())
    print(f"\n  饱和度  中位 {np.median(sat):.1f}  黑白帧(<10) {bw_frames}/{n} "
          f"({bw_frames/n*100:.1f}%)")
    bright_frames = int((lum > 235).sum())
    print(f"  亮度    中位 {np.median(lum):.1f}  近白帧(>235) {bright_frames}/{n} "
          f"({bright_frames/n*100:.1f}%)")

    # ---- 完整切点表 ----
    print(f"\n  前 40 个切点时刻 (s): {[round(x,3) for x in ct[:40]]}")

    # ---- 存 json 供后续使用 ----
    out_json = os.path.join(WORK, "ref_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"info": {k: v for k, v in info.items() if k != "raw"},
                   "n_frames": n, "analysis_fps": a.fps,
                   "cut_times": ct,
                   "gaps_ms": (np.diff(ct) * 1000).tolist() if len(ct) >= 2 else [],
                   "sat_median": float(np.median(sat)),
                   "lum_median": float(np.median(lum)),
                   "bw_ratio": bw_frames / n,
                   "bright_ratio": bright_frames / n},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  分析结果 -> {out_json}")

    # ---- 抽 PNG 供人眼核对 ----
    if a.frames_dir:
        fd = os.path.join(WORK, "frames")
        os.makedirs(fd, exist_ok=True)
        dur = info.get("duration", 30)
        stamps = [round(dur * i / 24, 2) for i in range(24)]
        for i, t in enumerate(stamps):
            subprocess.run(
                [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", f"{t}", "-i", a.ref, "-frames:v", "1",
                 "-vf", "scale=360:-2", os.path.join(fd, f"f{i:02d}_{t}.jpg")],
                capture_output=True, text=True)
        print(f"  抽取 24 帧 -> {fd}")

    # 清掉裸流（大文件）
    try:
        os.remove(raw)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
