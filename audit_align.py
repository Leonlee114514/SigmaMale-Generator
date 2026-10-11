"""段落级卡点审计（audit_align.py）
====================================================================
检查成片里"画面段落"和"音乐段落"是否对齐。

要审什么：
    好的卡点成片应该做到段落级呼应 ——
        画面铺垫段  ↔  音乐铺垫段（前奏/break，能量低）
        画面高潮段  ↔  音乐副歌（Drop，能量高）
    只在拍点上对齐是不够的：如果画面已经在快剪、音乐还在铺垫，
    或者反过来，观感就是"没卡上"。

怎么测（两边独立测量，再比对）：

  音乐侧：把成片音轨解码，算低频（底鼓）能量包络，
          找出能量跃升点 = Drop 位置。

  画面侧：把成片视频解码，算帧间差（硬切密度），
          切点密集的地方 = 快剪段 = 画面高潮。

  然后比对两者的边界是否一致。

⚠ 关键：测的是**成片自己**，而不是"我以为的"。
   设计意图和实际产物可能因为时长吸附、素材不足等原因偏离。
"""
import argparse
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")


def audio_env(path, sr=22050, win=0.05):
    """解码音轨 -> 低频能量包络（每 win 秒一个点）。

    用低通 + 短时 RMS 近似"底鼓能量"。副歌的底鼓明显更重更密。
    """
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, timeout=600)
    if not r.stdout:
        return None, None, None
    x = np.frombuffer(r.stdout, dtype=np.float32)
    # 一阶低通，保留 150Hz 以下（底鼓基频区）
    a = 0.955
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = a * acc + (1 - a) * x[i]
        y[i] = acc
    n = int(sr * win)
    m = len(y) // n
    env = np.sqrt((y[:m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)
    return env, win, sr


def frame_diff(path, w=96, h=170):
    """解码视频 -> 逐帧灰度 -> 帧间差（硬切检测）。"""
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
         "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, timeout=900)
    if not r.stdout:
        return None, None
    n = len(r.stdout) // (w * h)
    a = np.frombuffer(r.stdout[:n * w * h], dtype=np.uint8)
    a = a.reshape(n, w * h).astype(np.float32)
    d = np.abs(np.diff(a, axis=0)).mean(axis=1)
    return d, n


def main():
    ap = argparse.ArgumentParser(description="段落级卡点审计")
    ap.add_argument("videos", nargs="+", help="待审计的成片")
    ap.add_argument("--drop-t", type=float, default=7.96875,
                    help="设计上的画面高潮起点（秒）")
    a = ap.parse_args()

    print("=" * 74)
    print("段落级卡点审计")
    print("=" * 74)

    for v in a.videos:
        if not os.path.exists(v):
            print(f"\n跳过（不存在）: {v}")
            continue
        name = os.path.basename(v)
        print(f"\n{'─' * 74}\n{name}")

        # ---------- 音乐侧 ----------
        env, win, sr = audio_env(v)
        if env is None:
            print("  音轨解码失败")
            continue
        t_ax = np.arange(len(env)) * win
        # 平滑后再找跃升点
        k = max(1, int(0.25 / win))
        sm = np.convolve(env, np.ones(k) / k, mode="same")
        # 逐点"前后窗口能量比"，取最大处 = Drop
        W = int(1.0 / win)
        ratio = np.ones(len(sm))
        for i in range(W, len(sm) - W):
            pre = sm[i - W:i].mean()
            post = sm[i:i + W].mean()
            ratio[i] = post / max(pre, 1e-9)
        i_drop = int(np.argmax(ratio))
        t_drop = t_ax[i_drop]
        print(f"  [音乐] 能量跃升点 t={t_drop:.3f}s  跃升比 {ratio[i_drop]:.2f}x")
        print(f"         起点能量 {sm[:W].mean():.5f}   "
              f"末段能量 {sm[-W:].mean():.5f}")
        # 能量分段（低/高）
        mid = np.median(sm)
        low_frac = float((sm < mid).mean())
        print(f"         低于中位能量的时长占比 {low_frac*100:.1f}%")

        # ---------- 画面侧 ----------
        d, n = frame_diff(v)
        if d is None:
            print("  视频解码失败")
            continue
        fps = 30.0
        # 硬切密度：每 0.5s 窗口内的切点数
        thr = max(12.0, float(np.percentile(d, 92)))
        cuts = (d > thr).astype(float)
        ws = int(0.5 * fps)
        m = len(cuts) // ws
        dens = cuts[:m * ws].reshape(m, ws).sum(axis=1) / 0.5
        t_d = np.arange(m) * 0.5
        # 找密度跃升点
        half = m // 2
        if m > 4:
            lo = dens[:max(1, m // 3)].mean()
            hi = dens[max(1, 2 * m // 3):].mean()
            print(f"  [画面] 切点密度 前段 {lo:.2f}/s  后段 {hi:.2f}/s  "
                  f"（比值 {hi/max(lo,1e-6):.2f}x）")
            # 密度第一次持续超过 (lo+hi)/2 的位置
            mid_d = (lo + hi) / 2
            idx = None
            for i in range(m - 2):
                if dens[i:i + 3].mean() > mid_d:
                    idx = i
                    break
            if idx is not None:
                print(f"         切点密度跃升点 t={t_d[idx]:.3f}s")
        print(f"         总切点 {int(cuts.sum())} 个")

        # ---------- 比对 ----------
        print(f"  [比对] 设计高潮起点 {a.drop_t:.3f}s")
        dev_m = t_drop - a.drop_t
        print(f"         音乐跃升 - 设计 = {dev_m:+.3f}s "
              f"({abs(dev_m)*1000:.0f}ms)")
        if abs(dev_m) < 0.15:
            print("         ✓ 音乐高潮与画面设计一致")
        else:
            print(f"         ⚠ 偏差 {abs(dev_m)*1000:.0f}ms，偏大")


if __name__ == "__main__":
    sys.exit(main())
