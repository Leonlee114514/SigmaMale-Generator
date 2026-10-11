"""验证 beat_analyze 的核心算法：用**已知 BPM** 的合成信号回测。

为什么要先做这一步：
    在没有真实 BGM 的情况下，如果直接拿它去分析真实文件，
    无法判断"分析出来的 BPM 对不对"—— 因为标准答案是未知的。
    用合成信号回测，标准答案是**构造时就确定的**，这才算真验证。

回测要点：
    - 覆盖多个 BPM（80/100/128/140），128 是 phonk 的典型值
    - 覆盖不同相位偏移（0 / 0.1 / 0.25 / 0.37 秒）
    - 验证 BPM 误差 < 2，相位误差 < 25ms
    - 边界：纯静音不应报出 BPM
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import beat_analyze as ba


def synth_clicks(bpm, duration=20.0, offset=0.0, sr=ba.SR, noise=0.004):
    """合成"鼓点"信号：在每拍位置放一个快速衰减的脉冲。"""
    n = int(duration * sr)
    x = np.random.randn(n).astype(np.float32) * noise
    period = 60.0 / bpm
    t = offset
    while t < duration:
        i = int(t * sr)
        # 35ms 的衰减脉冲，带一点低频（模拟 kick）
        L = int(0.035 * sr)
        if i + L < n:
            env = np.exp(-np.arange(L) / (0.006 * sr))
            tone = np.sin(2 * np.pi * 70 * np.arange(L) / sr)
            x[i:i + L] += (env * tone).astype(np.float32) * 0.9
        t += period
    return x


def run_case(bpm, offset):
    sr = ba.SR
    x = synth_clicks(bpm, duration=20.0, offset=offset)
    db, hop = ba.energy_envelope(x, sr)
    strength = ba.onset_strength(db)
    onsets = ba.detect_onsets(strength, hop, sr)
    est_bpm, conf = ba.estimate_bpm(onsets, hop, sr)
    est_off = ba.align_phase(onsets, est_bpm, hop, sr)

    bpm_err = abs(est_bpm - bpm)
    # 相位误差按拍周期取模，避免"差一整拍"被算成大误差
    period = 60.0 / bpm
    ph_err = abs((est_off - offset + period / 2) % period - period / 2)

    ok = bpm_err < 2.0 and ph_err < 0.025
    print(f"  [{'OK  ' if ok else 'FAIL'}] 真实 BPM={bpm:6.1f} 偏移={offset:.3f}"
          f"  ->  测得 BPM={est_bpm:6.1f} 偏移={est_off:.3f}"
          f"   (BPM误差 {bpm_err:.2f}, 相位误差 {ph_err*1000:.1f}ms)")
    return ok


print("=" * 70)
print("节拍算法回测（合成信号，标准答案已知）")
print("=" * 70)

results = []
print("\n[1] 不同 BPM（相位固定 0）")
for bpm in (80, 100, 128, 140, 174):
    results.append(run_case(bpm, 0.0))

print("\n[2] 不同相位偏移（BPM 固定 128，phonk 典型值）")
for off in (0.0, 0.1, 0.25, 0.37, 0.45):
    results.append(run_case(128, off))

print("\n[3] 边界：纯静音")
x = np.random.randn(ba.SR * 10).astype(np.float32) * 0.002
db, hop = ba.energy_envelope(x, ba.SR)
st = ba.onset_strength(db)
on = ba.detect_onsets(st, hop, ba.SR)
bpm, conf = ba.estimate_bpm(on, hop, ba.SR)
silent_ok = bpm == 0.0
print(f"  [{'OK  ' if silent_ok else 'FAIL'}] 纯静音 -> BPM={bpm}")
results.append(silent_ok)

print()
print("=" * 70)
n_ok = sum(results)
print(f"通过 {n_ok}/{len(results)}")
if n_ok < len(results):
    print("存在失败项，需要调整算法参数")
print("=" * 70)
sys.exit(0 if n_ok == len(results) else 1)
