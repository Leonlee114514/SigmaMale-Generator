"""用 librosa 的拍点反推目标段落的最佳相位，并与手写相位对比。

目的：librosa 在 Drop 段的中位偏差 23.2ms，比手写的 37.0ms 好。
     但两者的"命中率"都是 71.4%，说明差异只在**相位的小数位**。
     这里把 librosa 的拍点约束到 128BPM 的均匀网格上，求最贴合底鼓的相位，
     得到一个可写进 edit_slop 的常数。
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import analyze_sections as asec    # noqa: E402
import edit_slop as es             # noqa: E402

BGM = os.path.join(BASE, "bgm", "gigachad_phonk.aac")
PERIOD = es.PERIOD


def kicks(t, sub, k=2.0, look=30, dedup=0.10):
    flux = np.maximum(np.diff(sub, prepend=sub[:1]), 0.0)
    out = []
    for i in range(1, len(flux) - 1):
        a, b = max(0, i - look), min(len(flux), i + look)
        thr = flux[a:b].mean() + k * flux[a:b].std()
        if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if not out or t[i] - out[-1] > dedup:
                out.append(t[i])
    return np.asarray(out)


x = asec.decode(BGM, asec.SR)
S, freqs = asec.stft_mag(x)
sub = asec.band_energy_db(S, freqs, 30, 120)
t_axis = (np.arange(len(sub)) * asec.HOP + asec.N_FFT / 2) / asec.SR
kk = kicks(t_axis, sub)

DROP_A, DROP_B = 30.5, 41.0
kk_d = kk[(kk >= DROP_A) & (kk < DROP_B)]


def score(phase, period=PERIOD, tol=0.04):
    r = np.mod(kk_d - phase, period)
    d = np.minimum(r, period - r)
    return float((d < tol).mean()), float(np.median(d) * 1000)


print("=" * 72)
print("Drop 段最佳相位搜索（目标：贴合底鼓）")
print("=" * 72)
print(f"Drop 段底鼓数 {len(kk_d)}\n")

# 细扫相位
best = (0.0, -1.0, 999.0)
for ph in np.linspace(0, PERIOD, 20000, endpoint=False):
    h, m = score(ph)
    if h > best[1] or (h == best[1] and m < best[2]):
        best = (float(ph), h, m)
print(f"细扫最佳相位 = {best[0]:.6f}s   命中率={best[1]*100:.1f}%  "
      f"中位偏差={best[2]:.1f}ms")

# 与现用相位对比
for name, ph in (("edit_slop 现用", es.PHASE),
                 ("细扫最佳", best[0])):
    h, m = score(ph)
    print(f"  {name:16} 相位={ph:.6f}  命中率={h*100:5.1f}%  "
          f"中位偏差={m:6.1f}ms")

# ---- 用 librosa 交叉确认 ----
import librosa                     # noqa: E402
y = asec.decode(BGM, 22050).astype(np.float32)
tempo, lb = librosa.beat.beat_track(y=y, sr=22050, units="time")
lb = np.asarray(lb, float)
lb_d = lb[(lb >= DROP_A) & (lb < DROP_B)]
print(f"\nlibrosa: BPM={float(np.atleast_1d(tempo)[0]):.2f}  "
      f"Drop 段拍点 {len(lb_d)} 个")
if len(lb_d) >= 2:
    # 用 librosa 拍点拟合 128BPM 网格的相位
    bl, bh, bm = 0.0, -1.0, 999.0
    for ph in np.linspace(0, PERIOD, 20000, endpoint=False):
        r = np.mod(lb_d - ph, PERIOD)
        d = np.minimum(r, PERIOD - r)
        h = float((d < 0.04).mean())
        m = float(np.median(d) * 1000)
        if h > bh or (h == bh and m < bm):
            bl, bh, bm = float(ph), h, m
    print(f"  由 librosa 拍点反推相位 = {bl:.6f}s  "
          f"命中率={bh*100:.1f}%  中位偏差={bm:.1f}ms")
    print(f"  与细扫最佳之差 = {(bl-best[0])*1000:+.1f}ms")

print(f"\n建议写入 edit_slop 的 PHASE = {best[0]:.6f}")
