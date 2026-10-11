"""验证卡点剪辑的数学：慢放锚点是否真的落在 BGM 的强拍上。

这是整个卡点功能的核心正确性问题：
    如果锚点算错了，导出视频就会"卡不准"，而这在视觉上很难一眼看出来
    （画面上没人会去数秒表），所以必须用程序验证。

验证方法：
    构造一个已知 BPM/offset 的节拍字典 -> 算变速方案 -> 检查每个锚点
    与最近的强拍时间差是否在容差内。
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import beat_sync as bs


def make_beat_info(bpm, offset, duration=200.0):
    period = 60.0 / bpm
    beats, t = [], offset
    while t < duration:
        beats.append(round(t, 4))
        t += period
    return {
        "file": "synthetic.mp3", "duration": duration,
        "bpm": bpm, "offset": offset,
        "beats": beats, "downbeats": beats[::4],
    }


def check_anchors(anchors, downbeats, slow_window=0.0, tol=0.02):
    """检查慢放锚点是否压在强拍上。

    ⚠ 语义：anchors 是慢放段的**起点**，压住强拍的是段**中点**。
      所以要比对的是 anchor + slow_window/2，而不是 anchor 本身。
      早期测试直接拿 anchor 去比，导致全部假失败。
    """
    if not anchors:
        return [], 0.0
    dbs = np.asarray(downbeats, dtype=float)
    errs = []
    for a in anchors:
        mid = a + slow_window / 2.0
        errs.append(float(np.min(np.abs(dbs - mid))))
    return errs, float(np.max(errs))


print("=" * 72)
print("卡点锚点对齐验证")
print("=" * 72)

all_ok = True

print("\n[1] 不同 BPM 下，慢放段中点是否压在强拍上")
for bpm in (100, 128, 140):
    bi = make_beat_info(bpm, 0.0)
    for src_dur in (30.0, 45.0, 60.0):
        plan = bs.fit_to_beats(src_dur, bi, n_downbeats=4)
        errs, mx = check_anchors(plan["anchors"], bi["downbeats"],
                                 plan["slow_window"])
        ok = mx < 0.02 if errs else False
        all_ok &= ok
        print(f"  [{'OK  ' if ok else 'FAIL'}] BPM={bpm:3} 素材={src_dur:4.0f}s"
              f" -> 目标={plan['target_duration']:6.3f}s"
              f" 锚点={len(plan['anchors']):2} 最大偏差={mx*1000:6.3f}ms")

print("\n[2] 变速后总时长是否等于目标时长（误差 <1%）")
for bpm in (100, 128, 140):
    bi = make_beat_info(bpm, 0.0)
    plan = bs.fit_to_beats(40.0, bi, n_downbeats=4)
    tgt, got = plan["target_duration"], plan["total_out"]
    err = abs(got - tgt) / tgt * 100
    ok = err < 1.0
    all_ok &= ok
    print(f"  [{'OK  ' if ok else 'FAIL'}] BPM={bpm:3}"
          f" 目标={tgt:.3f}s 实得={got:.3f}s 误差={err:.3f}%")

print("\n[3] 不同 offset 下锚点仍应压在强拍上")
for off in (0.0, 0.123, 0.456):
    bi = make_beat_info(128, off)
    plan = bs.fit_to_beats(35.0, bi, n_downbeats=4)
    errs, mx = check_anchors(plan["anchors"], bi["downbeats"],
                             plan["slow_window"])
    ok = mx < 0.02 if errs else False
    all_ok &= ok
    print(f"  [{'OK  ' if ok else 'FAIL'}] offset={off:.3f}"
          f" 锚点={len(plan['anchors']):2} 最大偏差={mx*1000:6.3f}ms")

print("\n[4] n_downbeats 对总时长的影响（应线性）")
bi = make_beat_info(128, 0.0)
dbeat = 60.0 / 128 * 4
for n in (2, 4, 6, 8):
    plan = bs.fit_to_beats(60.0, bi, n_downbeats=n)
    expect = dbeat * n
    ok = abs(plan["target_duration"] - expect) < 1e-6
    all_ok &= ok
    print(f"  [{'OK  ' if ok else 'FAIL'}] n={n} -> {plan['target_duration']:.3f}s"
          f"  (期望 {expect:.3f}s)")

print("\n[5] 闪白点：只强拍 vs 每拍，数量关系应为 1:4")
bi = make_beat_info(128, 0.0)
db_pts = bs.flash_points(bi, n_downbeats=4, per_beat=False)
b_pts = bs.flash_points(bi, n_downbeats=4, per_beat=True)
ratio = len(b_pts) / max(len(db_pts), 1)
ok = 3.5 < ratio < 4.5
all_ok &= ok
print(f"  [{'OK  ' if ok else 'FAIL'}] 强拍点={len(db_pts)}  每拍点={len(b_pts)}"
      f"  比值={ratio:.2f}")

print("\n[6] 模式 B：BGM 截取是否落在强拍边界上")
for vdur in (13.0, 15.0, 17.3, 20.0):
    bi = make_beat_info(128, 0.0, duration=120)
    f = bs.fit_audio_to_video(vdur, bi)
    print(f"  vdur={vdur:5.1f}s -> {f[:70]}...")

print("\n[7] 异常输入：BPM=0 应明确报错而不是给错结果")
bi0 = {"bpm": 0.0, "offset": 0.0, "beats": [], "downbeats": []}
try:
    bs.fit_to_beats(30.0, bi0)
    print("  FAIL  BPM=0 却没有报错")
    all_ok = False
except ValueError as e:
    print(f"  OK    正确报错: {e}")

print()
print("=" * 72)
print("结果:", "全部通过" if all_ok else "存在失败项")
print("=" * 72)
sys.exit(0 if all_ok else 1)
