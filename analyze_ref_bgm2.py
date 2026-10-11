"""参考片 BGM 节拍分析 v2（analyze_ref_bgm2.py）
====================================================================
v1 失败原因：正差分包络对这个 lo-fi 音源不敏感，lag 取到边界、投票得 0。

改用**谱通量（spectral flux）**：对每个 hop 做短时 FFT，累加"只有上升的
频段能量"。这是 onset detection 的标准做法，对弱鼓点远比时域差分敏感。

思路：
  1. STFT -> 谱通量 onset 强度
  2. 自相关估 BPM（含倍频/半频歧义检查）
  3. 扫相位找最佳网格
  4. 用参考片 17 个硬切反推：它们是不是落在某套网格上？
     —— 这一步最可靠，因为硬切是"人工确认过的事实"
"""
import json
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
BGM = os.path.join(BASE, "bgm", "ref_bgm.m4a")
SR = 22050
HOP = 256          # ~11.6ms
NFFT = 1024


def decode(path, sr=SR):
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, dtype="<f4").astype(np.float64)


def spectral_flux(y, sr=SR, nfft=NFFT, hop=HOP):
    """谱通量 onset 强度（只累加上升的频段）。"""
    win = np.hanning(nfft)
    n = (len(y) - nfft) // hop
    spec = np.empty((n, nfft // 2 + 1))
    for i in range(n):
        seg = y[i*hop:i*hop+nfft] * win
        spec[i] = np.abs(np.fft.rfft(seg))
    # 对数压缩（避免大能量主导），再取正差分
    s = np.log1p(spec * 100)
    diff = np.diff(s, axis=0)
    diff[diff < 0] = 0
    flux = diff.sum(axis=1)
    return flux, hop / sr


def ac_bpm(flux, dt, lo=50, hi=200):
    """自相关估 BPM，返回候选（含倍频）。"""
    x = flux - flux.mean()
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    a = max(1, int(60/hi/dt))
    b = min(len(ac), int(60/lo/dt))
    if b <= a + 1:
        return []
    seg = ac[a:b].copy()
    # 找局部峰
    cands = []
    for i in range(1, len(seg)-1):
        if seg[i] > seg[i-1] and seg[i] >= seg[i+1]:
            cands.append((seg[i], 60.0 / ((a + i) * dt)))
    cands.sort(reverse=True)
    return cands[:8]


def scan_phase(times, period, tol=0.03):
    """给定一组时刻和周期，找最佳相位，返回 (相位, 中位偏差, 命中率)。"""
    if not times:
        return None
    best = None
    for ph in np.arange(0, period, 0.001):
        dev = [abs(((t - ph + period/2) % period) - period/2) for t in times]
        med = float(np.median(dev))
        hit = sum(1 for x in dev if x < tol) / len(dev)
        score = med - 0.015 * hit
        if best is None or score < best[0]:
            best = (score, ph, med, hit)
    return best[1], best[2], best[3]


def main():
    print("=" * 72)
    print("参考片 BGM 节拍分析 v2（谱通量）")
    print("=" * 72)
    y = decode(BGM)
    print(f"  {os.path.basename(BGM)}  {len(y)/SR:.3f}s")

    flux, dt = spectral_flux(y)
    print(f"  谱通量 {len(flux)} 帧  dt={dt*1000:.2f}ms")

    # ---- 自相关候选 ----
    print(f"\n  [自相关 BPM 候选]")
    for sc, bpm in ac_bpm(flux, dt):
        print(f"    {bpm:7.2f} BPM  周期 {60/bpm*1000:7.2f}ms   相关 {sc:.1f}")

    # ---- 用一个"最可能"的 BPM 做相位扫描，并检查其整数倍是否更优 ----
    print(f"\n  [相位扫描] 对候选 BPM 逐一评估")
    cands = ac_bpm(flux, dt)[:6]
    seen = set()
    for sc, bpm in cands:
        if bpm < 50 or bpm > 210:
            continue
        key = round(bpm)
        if key in seen:
            continue
        seen.add(key)
        P = 60 / bpm
        r = scan_phase([t * dt for t in np.where(flux > flux.mean() + flux.std())[0]],
                       P)
        if r:
            print(f"    {bpm:7.2f} BPM ({P*1000:6.2f}ms)  相位 {r[0]:.4f}s  "
                  f"中位偏差 {r[1]*1000:6.1f}ms  命中 {r[2]*100:.0f}%")

    # ---- 关键：用参考片自己的硬切反推网格 ----
    jp = os.path.join(BASE, "output", "_ref_analysis", "shots_ref.json")
    with open(jp, encoding="utf-8") as f:
        J = json.load(f)
    cuts = J["cut_times"]
    fc = [c for c in cuts if c >= 14.0]
    print(f"\n  ★ [参考片 {len(fc)} 个快剪硬切 反推网格]  这是最可靠的一步")
    print(f"     相邻硬切间隔(ms): "
          f"{[round(a*1000) for a in np.diff(fc)]}")
    print(f"     中位间隔 {np.median(np.diff(fc))*1000:.1f}ms  "
          f"-> 隐含 BPM {60/np.median(np.diff(fc)):.2f}")

    # 用中位间隔作为周期，扫相位
    for P in (0.900, float(np.median(np.diff(fc)))):
        r = scan_phase(fc, P)
        if r:
            print(f"     周期 {P*1000:6.2f}ms -> 相位 {r[0]:.4f}s  "
                  f"中位偏差 {r[1]*1000:6.1f}ms  命中 {r[2]*100:.0f}%")

    # ---- 两倍周期检验（如果 900ms 是 2 拍，那 450ms 才是真拍）----
    print(f"\n  [倍频检验] 450ms 网格能否也拟合上？")
    r = scan_phase(fc, 0.450)
    if r:
        print(f"     450ms 相位 {r[0]:.4f}s  中位偏差 {r[1]*1000:.1f}ms  "
              f"命中 {r[2]*100:.0f}%")

    # ---- 音频侧：用谱通量峰值直接看拍点 ----
    print(f"\n  [谱通量峰值时刻] 前 40 个（用于人工核对）")
    thr = flux.mean() + flux.std()
    pks = [i for i in range(1, len(flux)-1)
           if flux[i] > thr and flux[i] >= flux[i-1] and flux[i] > flux[i+1]]
    ts = [round(i * dt, 3) for i in pks]
    print(f"     {ts[:40]}")
    print(f"     共 {len(ts)} 个")

    out = os.path.join(BASE, "bgm", "ref_beats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"file": os.path.basename(BGM), "duration": len(y)/SR,
                   "flux_peaks": ts,
                   "ref_cuts": cuts,
                   "ref_fast_cuts": fc}, f, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
