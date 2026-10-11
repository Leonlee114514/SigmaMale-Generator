"""用 librosa 交叉验证手写节拍分析（回答"要不要造轮子"）。

现在本机装上了 librosa 1.0.0，可以直接拿它和 beat_analyze.py 对同一首 BGM
做对比。判据不是"谁的数字好看"，而是**谁给出的网格更能压住底鼓**——
这是剪辑真正关心的量，也是可客观测量的量。

librosa 用的是 onset strength + 动态规划 beat tracking（工业界标准做法），
我手写的是 IOI 直方图投票 + 相位扫描。对比能看出差距在哪。
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import analyze_sections as asec    # noqa: E402
import edit_slop as es             # noqa: E402

BGM = os.path.join(BASE, "bgm", "gigachad_phonk.aac")


def align_score(onsets, beats, tol=0.05):
    """每个底鼓击打点到最近拍点的距离，返回 (命中率, 中位偏差ms)。"""
    if len(onsets) == 0 or len(beats) == 0:
        return 0.0, 999.0
    b = np.asarray(beats, float)
    d = np.array([np.min(np.abs(b - t)) for t in onsets])
    return float((d < tol).mean()), float(np.median(d) * 1000)


# ---- 底鼓时刻（作为"标准答案"的观测量）----
x = asec.decode(BGM, asec.SR)
S, freqs = asec.stft_mag(x)
sub = asec.band_energy_db(S, freqs, 30, 120)
flux = np.maximum(np.diff(sub, prepend=sub[:1]), 0.0)
t_axis = (np.arange(len(sub)) * asec.HOP + asec.N_FFT / 2) / asec.SR
look, k = 30, 2.0
kick = []
for i in range(1, len(flux) - 1):
    a, b = max(0, i - look), min(len(flux), i + look)
    thr = flux[a:b].mean() + k * flux[a:b].std()
    if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
        if not kick or t_axis[i] - kick[-1] > 0.10:
            kick.append(t_axis[i])
kick = np.asarray(kick)
print("=" * 78)
print("librosa vs 手写节拍分析")
print("=" * 78)
print(f"底鼓击打点（作为观测量）: {len(kick)} 个")

# ---- 手写实现 ----
import beat_analyze as ba          # noqa: E402
info = ba.analyze(BGM)
mine_beats = np.asarray(info["beats"], float)
h_mine, e_mine = align_score(kick, mine_beats)
print(f"\n[手写 beat_analyze（全局相位）]")
print(f"  BPM={info['bpm']}  相位={info['offset']}s  拍点数={len(mine_beats)}")
print(f"  底鼓命中率(±50ms)={h_mine*100:5.1f}%   中位偏差={e_mine:6.1f}ms")

# ---- 手写实现（分段精修相位）----
# ⚠ 这才是 edit_slop 实际使用的网格：相位是在**目标段落内**拟合的。
#   拿全局相位去比 Drop 段是不公平的 —— 全局相位被前奏/主歌带偏，
#   而剪辑只用 Drop 段。必须按"实际使用的配置"去比。
ref_beats = np.array([es.PHASE + k * es.PERIOD for k in range(400)])
h_ref, e_ref = align_score(kick, ref_beats)
print(f"\n[手写（分段精修相位 {es.PHASE}s）—— edit_slop 实际用的]")
print(f"  BPM={es.BPM}  相位={es.PHASE}s")
print(f"  底鼓命中率(±50ms)={h_ref*100:5.1f}%   中位偏差={e_ref:6.1f}ms")

# ---- librosa ----
# ⚠ soundfile 读不了 .aac（"Format not recognised"）。
#   但 librosa 的分析函数都接受**内存里的 numpy 数组**，所以不必依赖
#   soundfile 的解码能力 —— 直接复用我们自己的 ffmpeg 解码结果即可。
#   这比"先把 aac 转成 wav 再读"更干净：少一个中间文件，也不会因
#   重采样两次而引入额外误差。
import librosa                     # noqa: E402
y = asec.decode(BGM, 22050).astype(np.float32)
print(f"\n[librosa {librosa.__version__}]")
print(f"  输入: ffmpeg 解码的 {len(y)/22050:.2f}s @22050Hz（绕开 soundfile）")
tempo, lb = librosa.beat.beat_track(y=y, sr=22050, units="time")
lb = np.asarray(lb, float)
tempo_v = float(np.atleast_1d(tempo)[0])
h_lib, e_lib = align_score(kick, lb)
print(f"  BPM={tempo_v:.2f}  拍点数={len(lb)}")
print(f"  底鼓命中率(±50ms)={h_lib*100:5.1f}%   中位偏差={e_lib:6.1f}ms")

# ---- 也试一下 librosa 的 onset + tempo 估计（另一种口径）----
onset_env = librosa.onset.onset_strength(y=y, sr=22050)
t2 = librosa.feature.tempo(onset_envelope=onset_env, sr=22050)
print(f"  [另一口径] tempo={float(np.atleast_1d(t2)[0]):.2f} BPM")

# ---- 关键：Drop 段（30.5s 之后）单独比 ----
print(f"\n[只看 Drop 段 30.5-41s —— 这是真正要剪的区间]")
m = (kick >= 30.5) & (kick < 41.0)
kk = kick[m]
h1, e1 = align_score(kk, ref_beats)
h2, e2 = align_score(kk, lb)
h3, e3 = align_score(kk, mine_beats)
print(f"  手写全局相位     命中率={h3*100:5.1f}%  中位偏差={e3:6.1f}ms")
print(f"  手写分段精修相位 命中率={h1*100:5.1f}%  中位偏差={e1:6.1f}ms")
print(f"  librosa         命中率={h2*100:5.1f}%  中位偏差={e2:6.1f}ms")

print("\n" + "=" * 78)
if h2 > h1 + 0.08:
    print(f"结论: librosa 在目标段落更好（{h2*100:.0f}% vs {h1*100:.0f}%），"
          f"应改用 librosa 作为主用方案")
elif h1 > h2 + 0.08:
    print(f"结论: 手写分段精修更好（{h1*100:.0f}% vs {h2*100:.0f}%）")
else:
    print(f"结论: 两者在目标段落**相当**（{h1*100:.0f}% vs {h2*100:.0f}%）。")
    print(f"      真正的问题不在 BPM 估计，而在**相位必须在目标段落内拟合** ——")
    print(f"      全局相位({info['offset']}s)在 Drop 段命中率只有 {h3*100:.0f}%。")
print("=" * 78)
