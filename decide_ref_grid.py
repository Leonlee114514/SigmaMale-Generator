"""参考片网格最终裁定（decide_ref_grid.py）
====================================================================
数据：
  - 谱通量自相关：67.12 / 136.00（2:1 关系）
  - 硬切间隔：900ms（867ms 是 30fps 量化抖动，26 帧）
  - 136BPM 相位扫描：中位偏差 46.5ms / 命中 30%
  - 67BPM  相位扫描：中位偏差 98.7ms / 命中 10%

疑点：900ms 对 441.18ms(136BPM) = 2.04 拍，不是整数；
      对 893.97ms(67BPM) = 1.007 拍，接近整数。

这里做精确裁定：
  1. 假设真拍是 136BPM -> 快剪镜 = 2 拍 = 882.35ms
     与实测 900ms 差 17.65ms（= 0.53 帧），累积 15 镜会漂 265ms
  2. 假设真拍是 67BPM  -> 快剪镜 = 1 拍 = 893.97ms
     与实测 900ms 差 6ms，累积 15 镜漂 90ms
  3. 用硬切的**绝对时刻**（不是间隔）拟合，哪个更贴合？
  4. 用谱通量峰值的**绝对时刻**拟合，哪个更贴合？

判定标准：绝对时刻的中位偏差 + 命中率。
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
jp = os.path.join(BASE, "output", "_ref_analysis", "shots_ref.json")
bp = os.path.join(BASE, "bgm", "ref_beats.json")
with open(jp, encoding="utf-8") as f:
    J = json.load(f)
with open(bp, encoding="utf-8") as f:
    B = json.load(f)

cuts = J["cut_times"]
fc = [c for c in cuts if c >= 14.0]
flux_pk = B["flux_peaks"]


def fit(times, period, tol=0.03):
    """返回 (相位, 中位偏差, 命中率)。"""
    best = None
    for ph in np.arange(0, period, 0.0005):
        dev = [abs(((t - ph + period/2) % period) - period/2) for t in times]
        med = float(np.median(dev))
        hit = sum(1 for x in dev if x < tol) / len(dev)
        score = med - 0.015 * hit
        if best is None or score < best[0]:
            best = (score, ph, med, hit)
    return best[1], best[2], best[3]


print("=" * 74)
print("参考片拍网格裁定")
print("=" * 74)
print(f"\n快剪硬切 ({len(fc)} 个): {[round(c,3) for c in fc]}")
print(f"间隔(ms): {[round(a*1000) for a in np.diff(fc)]}")
print(f"  900ms 在 30fps = {900/1000*30:.2f} 帧 | 867ms = {867/1000*30:.2f} 帧")

print("\n" + "-" * 74)
print("假设 A：真拍 = 136 BPM（441.18ms），快剪镜 = 2 拍")
P136 = 60 / 136.0
print(f"  2 拍 = {2*P136*1000:.2f}ms   实测 900ms，差 {900-2*P136*1000:+.2f}ms")
ph, med, hit = fit(fc, 2 * P136)
print(f"  [用 2 拍周期拟合硬切] 相位 {ph:.4f}s  中位偏差 {med*1000:.1f}ms  "
      f"命中 {hit*100:.0f}%")
ph2, med2, hit2 = fit(fc, P136)
print(f"  [用 1 拍周期拟合硬切] 相位 {ph2:.4f}s  中位偏差 {med2*1000:.1f}ms  "
      f"命中 {hit2*100:.0f}%")
phf, medf, hitf = fit(flux_pk, P136)
print(f"  [用 1 拍周期拟合谱通量峰] 相位 {phf:.4f}s  中位偏差 {medf*1000:.1f}ms  "
      f"命中 {hitf*100:.0f}%")

print("\n" + "-" * 74)
print("假设 B：真拍 = 67.12 BPM（893.97ms），快剪镜 = 1 拍")
P67 = 60 / 67.12
print(f"  1 拍 = {P67*1000:.2f}ms   实测 900ms，差 {900-P67*1000:+.2f}ms")
ph3, med3, hit3 = fit(fc, P67)
print(f"  [用 1 拍周期拟合硬切] 相位 {ph3:.4f}s  中位偏差 {med3*1000:.1f}ms  "
      f"命中 {hit3*100:.0f}%")
ph3b, med3b, hit3b = fit(fc, P67/2)
print(f"  [用 0.5 拍周期拟合硬切] 相位 {ph3b:.4f}s  中位偏差 {med3b*1000:.1f}ms  "
      f"命中 {hit3b*100:.0f}%")

print("\n" + "-" * 74)
print("假设 C：直接用量化网格 —— 快剪镜正好 = 26 帧（867ms）@30fps")
P26 = 26 / 30.0
print(f"  26 帧 = {P26*1000:.2f}ms")
ph4, med4, hit4 = fit(fc, P26)
print(f"  [26 帧周期拟合硬切] 相位 {ph4:.4f}s  中位偏差 {med4*1000:.1f}ms  "
      f"命中 {hit4*100:.0f}%")
for nf in (13, 26, 52):
    P = nf / 30.0
    r = fit(fc, P)
    print(f"  [{nf:>2} 帧 = {P*1000:7.2f}ms] 相位 {r[0]:.4f}s  "
          f"中位偏差 {r[1]*1000:5.1f}ms  命中 {r[2]*100:.0f}%")

print("\n" + "=" * 74)
print("谱通量峰值 前 30 个 + 与 136BPM 网格的距离")
print("=" * 74)
phf, _, _ = fit(flux_pk, P136)
for t in flux_pk[:30]:
    d = abs(((t - phf + P136/2) % P136) - P136/2) * 1000
    mark = "✓" if d < 30 else " "
    print(f"  {t:7.3f}s   距网格 {d:6.1f}ms {mark}")
