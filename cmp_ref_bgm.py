"""参考片音频 vs 我们 BGM 的同源性核对（cmp_ref_bgm.py）
====================================================================
用户说"现在是 bgm 相同"，但必须实测确认，不能假设。
两件事：
  1. 参考片 BPM / 相位（用项目已有的 beat_analyze 口径）
  2. 参考片音频 vs bgm/gigachad_phonk.aac 是不是同一首
     —— 用解码后波形互相关的峰值位置判断（同源则峰值 ≈ 0）
  3. 参考片切点是否落在"同一套拍网格"上
"""
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
REF = r"C:\Users\Lyonl\Desktop\2022年的剪辑 belike.mp4"
BGM = os.path.join(BASE, "bgm", "gigachad_phonk.aac")
SR = 22050


def decode_audio(path, sr=SR):
    """解码成单声道 float32 numpy。绕开 soundfile（它读不了 aac）。"""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if not r.stdout:
        print("解码失败:", path, r.stderr[-300:])
        return None
    return np.frombuffer(r.stdout, dtype="<f4").astype(np.float64)


def envelope(y, sr=SR, win=0.02, hop=0.005):
    """能量包络（与 beat_analyze 同口径：20ms 窗 / 5ms 跳）。"""
    w = int(win * sr)
    h = int(hop * sr)
    n = (len(y) - w) // h
    e = np.array([np.sqrt(np.mean(y[i * h:i * h + w] ** 2))
                  for i in range(n)])
    return e, hop


def onset(e):
    """起点强度：正差分。"""
    d = np.diff(e)
    d[d < 0] = 0
    return d


def est_bpm(d, hop):
    """自相关估 BPM（与项目口径一致）。"""
    d = d - d.mean()
    ac = np.correlate(d, d, mode="full")[len(d) - 1:]
    # 60-200 BPM 对应的 lag
    lo, hi = int(60 / 200 / hop), int(60 / 60 / hop)
    seg = ac[lo:hi]
    lag = lo + int(np.argmax(seg))
    return 60.0 / (lag * hop)


def main():
    print("=" * 66)
    print("参考片音频分析")
    print("=" * 66)
    yr = decode_audio(REF)
    yb = decode_audio(BGM)
    if yr is None or yb is None:
        return 1
    print(f"  参考片音频 {len(yr)/SR:.3f}s")
    print(f"  我们的BGM  {len(yb)/SR:.3f}s  {os.path.basename(BGM)}")

    # ---- 各自估 BPM ----
    for name, y in (("参考片", yr), ("我们的BGM", yb)):
        e, hop = envelope(y)
        d = onset(e)
        bpm = est_bpm(d, hop)
        print(f"  {name:10} BPM ≈ {bpm:.2f}   (拍周期 {60/bpm*1000:.1f}ms)")

    # ---- 同源性：互相关 ----
    # 取两边共同长度的中段，做互相关，看峰值位置
    L = int(min(len(yr), len(yb), 40 * SR))
    x = yr[:L] - yr[:L].mean()
    y = yb[:L] - yb[:L].mean()
    # 归一化互相关
    n = 1 << (2 * L - 1).bit_length()
    fx = np.fft.rfft(x, n)
    fy = np.fft.rfft(y, n)
    cc = np.fft.irfft(fx * np.conj(fy), n)
    cc = np.concatenate([cc[-(L - 1):], cc[:L]])
    lags = np.arange(-(L - 1), L)
    norm = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    cc /= (norm or 1)
    pk = int(np.argmax(np.abs(cc)))
    print(f"\n  [同源性] 最大归一化互相关 = {cc[pk]:+.4f} "
          f"@ lag {lags[pk]/SR*1000:+.1f}ms")

    # ---- 参考片切点是否落在 128BPM 拍网格上 ----
    import json
    jp = os.path.join(BASE, "output", "_ref_analysis", "ref_analysis.json")
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as f:
            js = json.load(f)
        cuts = js["cut_times"]
        PERIOD = 60 / 128.0
        # 只在"切点密集段"（Drop 后）检验，前奏/主歌的切点是叙事性的
        fast = [c for c in cuts if c > 13.0]
        if fast:
            best = None
            for ph in np.arange(0, PERIOD, 0.0005):
                dev = [(c - ph) % PERIOD for c in fast]
                dev = [min(d, PERIOD - d) for d in dev]
                med = float(np.median(dev))
                hit = sum(1 for d in dev if d < 0.03 or d > PERIOD - 0.03)
                if best is None or med < best[1]:
                    best = (ph, med, hit / len(fast))
            print(f"\n  [切点对 128BPM 网格] 密集段 {len(fast)} 个切点")
            print(f"    最佳相位 {best[0]:.4f}s  中位偏差 {best[1]*1000:.1f}ms  "
                  f"命中率 {best[2]*100:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
