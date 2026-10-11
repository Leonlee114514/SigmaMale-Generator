"""动作阈值标定（calibrate_actions.py）
====================================================================
用**缓存的特征**扫阈值，看清"改阈值会得到几个事件"，而不是盲试。

为什么需要它：
    动作检测的阈值有两种典型失败，肉眼看不出来：

      1. **永远不可能触发** —— 信号的取值范围根本不跨阈值。
         实测：`耸肩` 阈值写 0.95，但信号 `-(肩到鼻的竖直距离/脸高)`
         恒为负（最大值 -0.291），所以 0.95 是**永远够不到的**。
         `双手举` 同理。这类错误不会报错，只是"这个动作从来不出现"，
         很容易被当成"素材里没有这个动作"。

      2. **一直触发** —— 阈值落在信号的常见取值里。
         实测：`微笑` 用绝对阈值 0.45 时，有人中性表情就在 0.4~0.5，
         于是整段静止的脸被判成一次 4.7 秒的微笑。

    两种失败的共同点是**只看结果发现不了**，必须看信号的分布和分位数。

用法：
    # 先确保有特征缓存（第一次会自动生成）
    python action_detect.py 素材.mp4 --cache output/动作分析/features.npz

    # 看所有动作在当前阈值下的表现 + 分位数扫描表
    python calibrate_actions.py

    # 只看某几个动作
    python calibrate_actions.py --actions 微笑,张嘴,耸肩
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_detect import (ACTIONS, load_cache, smooth, runs_above,  # noqa: E402
                           validity, robust_baseline)

# 扫描用的分位点。用**分位数**而不是等距数值：
# 阈值该定在"信号只有 5% 时间超过它"这种位置，而分位数直接表达了这件事，
# 等距网格在长尾分布上会浪费大半格子。
PCTS = [50, 70, 80, 90, 95, 97, 99, 99.5, 99.9]


def signal_of(F, cfg):
    """按动作定义算出**待阈值化的信号**（含 rel 基线校正与平滑）。"""
    raw = F[cfg["sig"]]
    valid = validity(F, cfg["sig"])
    sig = raw - robust_baseline(raw, valid) if cfg.get("rel") else raw
    return smooth(sig, cfg["smooth"]), valid


def main():
    ap = argparse.ArgumentParser(description="动作阈值标定")
    ap.add_argument("--cache", default="output/动作分析/features.npz")
    ap.add_argument("--actions", help="只看这些动作（逗号分隔）")
    ap.add_argument("--min-dur-scale", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=6, help="每行最多列几个事件")
    a = ap.parse_args()

    if not os.path.exists(a.cache):
        print(f"没有特征缓存 {a.cache}")
        print("先跑一次：python action_detect.py 素材.mp4 --cache " + a.cache)
        return 1
    data = load_cache(a.cache)
    F, fps = data["F"], data["fps"]
    print(f"读入特征 {data['n']} 帧 / {data['n']/fps:.1f}s（{a.cache}）")

    names = list(ACTIONS)
    if a.actions:
        want = {s.strip() for s in a.actions.split(",") if s.strip()}
        names = [n for n in names if n in want]

    for name in names:
        cfg = ACTIONS[name]
        s, valid = signal_of(F, cfg)
        v = s[valid]
        if v.size == 0:
            print(f"\n{name}: 无有效帧")
            continue
        ml = max(2, int(cfg["min_dur"] * a.min_dur_scale * fps))
        cl = int(cfg["cd"] * fps)
        tag = "（相对基线）" if cfg.get("rel") else "（绝对值）"

        print()
        print("=" * 78)
        print(f"{name}{tag}  信号 {cfg['sig']}  平滑窗 {cfg['smooth']}  "
              f"最短 {cfg['min_dur']}s")
        print("=" * 78)
        print(f"  取值: 最小 {v.min():.3f}  中位 {np.median(v):.3f}  "
              f"最大 {v.max():.3f}")

        # ⚠ 一行就能看出"永远够不到"：若阈值在信号范围之外，事件数恒为 0
        cur = cfg["thr"]
        in_range = v.min() <= cur <= v.max()
        print(f"  当前阈值 {cur:.2f}  " +
              ("落在信号范围内 ✓" if in_range else "★ 落在信号范围**之外**"))
        if not in_range:
            print("    → 本素材中该动作**不可能触发**（事件数必然是 0）。")
            print("      两种可能，必须分清：")
            print("      (a) 阈值语义与信号符号不符 —— 是真 bug")
            print("          （实测：耸肩阈值 0.95，但信号恒为负，最大值才 -0.31）")
            print("      (b) 这段素材里确实没做这个动作 —— 不用改")
            print("      判据：把阈值设到 p90~p99 之间跑一遍，")
            print("      若事件时长/位置合理，说明动作存在、只是阈值定错了。")

        print()
        print(f"  {'阈值':>8}{'来源':>10}{'事件数':>7}{'最长':>8}{'合计':>8}"
              f"  事件时长（前 {a.top} 个）")
        print("  " + "-" * 76)
        rows = [("current", cur)] + [(f"p{p}", float(np.percentile(v, p)))
                                     for p in PCTS]
        for tag2, thr in rows:
            evs = runs_above(s, thr, ml, cl, valid)
            if evs:
                ds = [(b - a_) / fps for a_, b, _ in evs]
                shown = ", ".join(f"{d:.2f}" for d in sorted(ds, reverse=True)[:a.top])
            else:
                ds, shown = [], "(无)"
            mark = "  ← 当前" if tag2 == "current" else ""
            print(f"  {thr:>8.2f}{tag2:>10}{len(evs):>7}"
                  f"{(max(ds) if ds else 0):>8.2f}{sum(ds):>8.2f}  {shown}{mark}")
        print()
        print("  · 选阈值的原则：事件数在 2~8 个、最长 0.3~1.5s 之间比较可用；")
        print("    事件数太多说明切太碎；最长时长好几秒说明这是「状态」而不是")
        print("    「动作」—— 这时该看的是**峰值时刻**（供切点吸附用），")
        print("    而不是整段时长。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
