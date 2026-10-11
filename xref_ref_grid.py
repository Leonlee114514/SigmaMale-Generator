"""网格裁定的独立交叉验证（xref_ref_grid.py）
====================================================================
decide_ref_grid.py 用自己写的 fit() 得出 67.12BPM/1拍 最优（命中88%）。
为了防止"自己的实现自己验"这种自证问题，这里换一套独立口径：

  1. 用 librosa 的 beat tracker 直接跑参考片音频（项目已装 librosa 1.0.0）
     —— 完全独立的实现
  2. 用"网格残差-周期扫描"：对每个候选周期，统计硬切落点的残差方差
     （不扫相位，用解析法求最优相位）—— 数学上等价但实现不同
  3. 穷举"每镜帧数"假设：9/13/14/26/27/28 帧 @30fps，看哪个最整齐

三者都指向同一个周期才可信。
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
BGM = os.path.join(BASE, "bgm", "ref_bgm.m4a")
SR = 22050

jp = os.path.join(BASE, "output", "_ref_analysis", "shots_ref.json")
with open(jp, encoding="utf-8") as f:
    J = json.load(f)
fc = [c for c in J["cut_times"] if c >= 14.0]


def decode(path, sr=SR):
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, dtype="<f4").astype(np.float64)


print("=" * 76)
print("独立交叉验证 1：librosa beat tracker（完全独立的实现）")
print("=" * 76)
try:
    import librosa
    y = decode(BGM).astype(np.float32)
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=512,
                                           start_bpm=120.0)
    tempo = float(np.atleast_1d(tempo)[0])
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=512)
    print(f"  librosa tempo = {tempo:.2f} BPM   周期 {60/tempo*1000:.2f}ms")
    print(f"  检出 {len(bt)} 个拍点")
    d = np.diff(bt)
    print(f"  拍间隔 中位 {np.median(d)*1000:.1f}ms  "
          f"min {d.min()*1000:.0f}ms  max {d.max()*1000:.0f}ms")
    if len(bt) > 2:
        print(f"  前 20 个拍点: {[round(x,3) for x in bt[:20]]}")
except Exception as ex:
    print("  librosa 失败:", ex)

print("\n" + "=" * 76)
print("独立交叉验证 2：解析法求最优相位（残差方差最小）")
print("=" * 76)
print("  对周期 P，最优相位可用圆均值解析求出，不用网格扫描：")
print("    ph = atan2(mean(sin(2πt/P)), mean(cos(2πt/P))) * P / (2π)")
print("  残差 R = sqrt(-2*ln(Rbar)) * P / (2π)   (圆统计的 std)\n")

arr = np.array(fc)
print(f"  {'周期ms':>8} {'对应BPM':>8} {'最优相位':>9} {'残差std':>9} "
      f"{'命中<30ms':>10}")
best = None
for P_ms in np.arange(300, 1200, 1.0):
    P = P_ms / 1000.0
    ang = 2 * math.pi * arr / P
    C, S = np.cos(ang).mean(), np.sin(ang).mean()
    Rbar = math.hypot(C, S)
    ph = (math.atan2(S, C) * P / (2 * math.pi)) % P
    if Rbar <= 1e-9:
        continue
    rstd = math.sqrt(-2 * math.log(Rbar)) * P / (2 * math.pi)
    dev = np.abs(((arr - ph + P/2) % P) - P/2)
    hit = float((dev < 0.030).mean())
    if best is None or rstd < best[0]:
        best = (rstd, P_ms, ph, hit)
    if abs(P_ms - 900) < 0.5 or abs(P_ms - 893.9) < 3 or abs(P_ms - 882.4) < 3 \
            or abs(P_ms - 866.7) < 0.5 or abs(P_ms - 441.2) < 2:
        print(f"  {P_ms:8.1f} {60000/P_ms:8.2f} {ph:9.4f} {rstd*1000:9.1f} "
              f"{hit*100:9.0f}%")
print(f"\n  ★ 残差最小: 周期 {best[1]:.1f}ms (BPM {60000/best[1]:.2f})  "
      f"相位 {best[2]:.4f}s  残差std {best[0]*1000:.1f}ms  命中 {best[3]*100:.0f}%")

print("\n" + "=" * 76)
print("独立交叉验证 3：穷举'每镜帧数'假设（30fps 下）")
print("=" * 76)
print(f"  {'帧/镜':>6} {'周期ms':>8} {'BPM':>8} {'残差std':>9} {'命中<1帧':>10}")
for nf in range(8, 35):
    P = nf / 30.0
    ang = 2 * math.pi * arr / P
    C, S = np.cos(ang).mean(), np.sin(ang).mean()
    Rbar = math.hypot(C, S)
    if Rbar <= 1e-9:
        continue
    ph = (math.atan2(S, C) * P / (2 * math.pi)) % P
    rstd = math.sqrt(-2 * math.log(Rbar)) * P / (2 * math.pi)
    dev = np.abs(((arr - ph + P/2) % P) - P/2)
    hit = float((dev < (1/60)).mean())     # 半帧容差
    star = ""
    if nf in (13, 26, 27, 28):
        star = " ←"
    print(f"  {nf:>6} {P*1000:8.2f} {60/P:8.2f} {rstd*1000:9.1f} "
          f"{hit*100:9.0f}%{star}")

print("\n" + "=" * 76)
print("独立交叉验证 4：所有硬切（含铺垫段的 2 个）一起拟合")
print("=" * 76)
allc = np.array(J["cut_times"])
print(f"  全部 {len(allc)} 个硬切: {[round(x,2) for x in allc]}")
for label, P in (("893.92ms(67.12BPM)", 0.89392), ("900ms", 0.900),
                 ("882.35ms(136BPM/2)", 0.88235), ("866.67ms(26帧)", 26/30)):
    ang = 2 * math.pi * allc / P
    C, S = np.cos(ang).mean(), np.sin(ang).mean()
    Rbar = math.hypot(C, S)
    ph = (math.atan2(S, C) * P / (2 * math.pi)) % P
    rstd = math.sqrt(-2 * math.log(max(Rbar, 1e-9))) * P / (2 * math.pi)
    dev = np.abs(((allc - ph + P/2) % P) - P/2)
    hit = float((dev < 0.030).mean())
    print(f"  {label:22} 相位 {ph:.4f}s  残差std {rstd*1000:6.1f}ms  "
          f"命中 {hit*100:3.0f}%")
