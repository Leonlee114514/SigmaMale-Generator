"""参考片 BGM 节拍分析（analyze_ref_bgm.py）
====================================================================
用户确定：新成片就用参考片这首 BGM，卡点按它的节奏来。
所以必须先把它分析干净：

  1. 估 BPM（多种方法交叉验证）
  2. 定相位（拍网格落在哪）
  3. 定位 Drop / 结构边界
  4. **核对 900ms 是不是整拍** —— 这是复刻的核心前提
  5. 核对参考片的 17 个硬切是否落在拍网格上（如果是，那参考片本身就是
     按"每拍一刀"做的，我们的复刻就照这个网格排）

输出：bgm/ref_beats.json
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


def decode(path, sr=SR):
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, dtype="<f4").astype(np.float64)


def analyze(y, sr=SR, win=0.02, hop=0.005):
    w, h = int(win * sr), int(hop * sr)
    n = (len(y) - w) // h
    e = np.sqrt(np.array([np.mean(y[i*h:i*h+w] ** 2) for i in range(n)]))
    d = np.diff(e)
    d[d < 0] = 0
    return e, d, h


def bpm_ac(d, hop, lo=50, hi=200):
    """自相关估 BPM。
    ⚠ lag 的搜索范围必须夹到自相关数组的实际长度内 ——
      60/50=1.2s 对应 lag=1.2/0.005=240，而包络只有几千帧虽然够，
      但 lo/hi 给反或数据太短时 a>=b 会得到空切片直接崩。"""
    x = d - d.mean()
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    a, b = int(60/hi/hop), int(60/lo/hop)
    a = max(1, min(a, len(ac) - 2))
    b = max(a + 1, min(b, len(ac)))
    lag = a + int(np.argmax(ac[a:b]))
    return 60.0 / (lag * hop)


def bpm_hist(d, hop, lo=50, hi=200):
    """直方图投票：统计相邻峰间隔。"""
    thr = d.mean() + d.std()
    pk = [i for i in range(1, len(d)-1)
          if d[i] > thr and d[i] >= d[i-1] and d[i] > d[i+1]]
    gaps = np.diff(pk) * hop
    gaps = gaps[(gaps > 60/hi) & (gaps < 60/lo)]
    if len(gaps) == 0:
        return 0.0
    bpms = 60.0 / gaps
    hist, edges = np.histogram(bpms, bins=np.arange(lo, hi, 0.5))
    return float(edges[np.argmax(hist)] + 0.25)


def best_phase(d, hop, period, tmax=None):
    """在给定周期下扫相位，用中位偏差+命中率联合选最优。"""
    onsets = []
    thr = d.mean() + d.std()
    for i in range(1, len(d)-1):
        if d[i] > thr and d[i] >= d[i-1] and d[i] > d[i+1]:
            t = i * hop
            if tmax is None or t < tmax:
                onsets.append(t)
    if not onsets:
        return None
    best = None
    for ph in np.arange(0, period, 0.0005):
        dev = [abs(((t - ph + period/2) % period) - period/2) for t in onsets]
        med = float(np.median(dev))
        hit = sum(1 for x in dev if x < 0.03) / len(dev)
        # 目标函数：精度优先，命中率次之
        score = med - 0.02 * hit
        if best is None or score < best[0]:
            best = (score, ph, med, hit, len(onsets))
    return best


def main():
    print("=" * 70)
    print("参考片 BGM 节拍分析")
    print("=" * 70)
    y = decode(BGM)
    dur = len(y) / SR
    print(f"  文件 {os.path.basename(BGM)}   {dur:.3f}s   {len(y)} samples")

    e, d, hop = analyze(y)
    print(f"  包络 {len(e)} 帧（20ms 窗 / 5ms 跳）")

    b1 = bpm_ac(d, hop)
    b2 = bpm_hist(d, hop)
    print(f"\n  BPM 估计:")
    print(f"    自相关    {b1:.2f}   (拍周期 {60/b1*1000:.2f}ms)")
    print(f"    直方图投票 {b2:.2f}   (拍周期 {60/b2*1000:.2f}ms)")

    print(f"\n  [核心核对] 参考片快切镜 900ms -> BPM {60/0.900:.3f}")
    for name, b in (("自相关", b1), ("投票", b2)):
        r = 0.900 / (60/b)
        print(f"    {name} {b:.2f} -> 900ms = {r:.4f} 拍")

    # 用 900ms 作为周期直接验相位
    print(f"\n  [以 900ms 为周期扫相位]")
    r900 = best_phase(d, hop, 0.900)
    if r900:
        _, ph, med, hit, n = r900
        print(f"    相位 {ph:.4f}s  中位偏差 {med*1000:.1f}ms  "
              f"命中 {hit*100:.0f}%  (共 {n} 个起点)")

    # 用估出的 BPM 验相位
    for name, b in (("自相关", b1), ("投票", b2)):
        r = best_phase(d, hop, 60/b)
        if r:
            _, ph, med, hit, n = r
            print(f"  [以 {b:.2f}BPM 为周期扫相位] 相位 {ph:.4f}s  "
                  f"中位偏差 {med*1000:.1f}ms  命中 {hit*100:.0f}%")

    # ---- 参考片硬切是否落在 900ms 网格上 ----
    jp = os.path.join(BASE, "output", "_ref_analysis", "shots_ref.json")
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as f:
            cuts = json.load(f)["cut_times"]
        print(f"\n  [参考片 {len(cuts)} 个硬切对 900ms 网格]")
        # 只要快切段的
        fc = [c for c in cuts if c >= 14.0]
        for P in (0.900, 60/b1, 60/b2):
            best = None
            for ph in np.arange(0, P, 0.001):
                dev = [abs(((c - ph + P/2) % P) - P/2) for c in fc]
                med = float(np.median(dev))
                if best is None or med < best[0]:
                    best = (med, ph)
            print(f"    周期 {P*1000:.1f}ms: 最佳相位 {best[1]:.3f}s  "
                  f"中位偏差 {best[0]*1000:.1f}ms")

    # ---- 结构：能量/低频找 Drop ----
    print(f"\n  [结构] 逐秒能量（找 Drop / 段落边界）")
    sec = int(SR)
    for i in range(0, int(dur), 2):
        seg = y[i*sec:(i+2)*sec]
        if len(seg) == 0:
            break
        rms = np.sqrt(np.mean(seg ** 2))
        bar = "▇" * int(min(rms * 260, 50))
        print(f"    {i:>3}-{i+2:>3}s {rms:.4f} {bar}")

    out = os.path.join(BASE, "bgm", "ref_beats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"file": os.path.basename(BGM), "duration": dur,
                   "bpm_ac": b1, "bpm_hist": b2,
                   "period_900": 0.900,
                   "phase_900": r900[1] if r900 else None},
                  f, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
