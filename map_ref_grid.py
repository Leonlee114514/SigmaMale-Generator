"""参考片结构 vs 我们的 128BPM 网格的映射核对（map_ref_grid.py）
====================================================================
已知：
  参考片 18 镜：
    0  0.000- 4.333  4.333s   （前奏/铺垫，长镜）
    1  4.333-14.333 10.000s   （主歌，长镜）
    2 14.333-15.233  0.900s  ┐
    ...                       ├ 15 个 900ms 快切镜 + 尾镜
    16 26.833-27.700 0.867s  ┘
    17 27.700-30.000  2.300s  （结尾 LUV2 字幕，黑白）

检查：
  A) 900ms 对应参考片 BPM？  -> 66.67 BPM
  B) 900ms 换算到 128BPM 是几拍？ -> 900/468.75 = 1.92 拍
  C) 快剪段起点 14.333s 在参考片 BGM 里的位置
  D) 我们成片的对应段：快剪起点 8.4375s，长度又是多少？
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
PERIOD = 60 / 128.0          # 0.46875s
REF_PERIOD = 0.900           # 参考片一个快切镜

jp = os.path.join(BASE, "output", "_ref_analysis", "shots_ref.json")
with open(jp, encoding="utf-8") as f:
    J = json.load(f)

print("=" * 70)
print("参考片结构 vs 我们的 128BPM")
print("=" * 70)

print(f"\n[参考片] 快切镜时长 900ms  -> 隐含 BPM {60/0.900:.2f}")
print(f"[我们]   128BPM 拍周期 {PERIOD*1000:.2f}ms")
print(f"         900ms / {PERIOD*1000:.2f}ms = {0.900/PERIOD:.3f} 拍")

# 我们成片的结构
N_NORMAL_BEATS = 17
normal = N_NORMAL_BEATS * PERIOD
drop = normal + PERIOD * 0.25          # 慢放段
fast_start = normal + PERIOD
N_FAST = 22
fast_len = N_FAST * PERIOD
total = normal + PERIOD + fast_len

print(f"\n[我们成片]")
print(f"  正常段   0.000-{normal:.3f}s  ({N_NORMAL_BEATS} 拍)")
print(f"  慢放冲击 {normal:.3f}-{fast_start:.3f}s  (0.25x, 1 拍)")
print(f"  快剪段   {fast_start:.3f}-{total:.3f}s  ({N_FAST} 拍 = {fast_len:.3f}s)")
print(f"  总计     {total:.3f}s")

# 参考片快剪段
shots = J["shots"]
fast = [s for s in shots if 0.8 < s["dur"] < 1.0]
print(f"\n[参考片]")
print(f"  铺垫段   0.000-{shots[1]['t1']:.3f}s  (2 个长镜)")
print(f"  快剪段   {fast[0]['t0']:.3f}-{fast[-1]['t1']:.3f}s  "
      f"({len(fast)} 个镜 × 900ms = {len(fast)*0.9:.2f}s)")
print(f"  结尾镜   {shots[-1]['t0']:.3f}-{shots[-1]['t1']:.3f}s  "
      f"({shots[-1]['dur']:.3f}s, LUV2 字幕)")
print(f"  总计     {shots[-1]['t1']:.3f}s")

print("\n" + "=" * 70)
print("映射方案对比")
print("=" * 70)

# 参考片快剪段 = 14.333-27.700 = 13.367s
ref_fast_len = fast[-1]["t1"] - fast[0]["t0"]
print(f"\n参考片快剪段长度 {ref_fast_len:.3f}s")
print(f"我们快剪段长度   {fast_len:.3f}s")
print(f"比例 {ref_fast_len/fast_len:.3f}")

n128 = ref_fast_len / PERIOD
print(f"\n参考片快剪段折算到 128BPM = {n128:.3f} 拍")

print(f"""
关键结论
--------
参考片快剪段 {len(fast)} 镜 × 900ms，我们的是 {N_FAST} 镜 × {PERIOD*1000:.0f}ms。
两者镜头数不同（{len(fast)} vs {N_FAST}），所以**不是简单的一对一替换**。

复刻要点：
  1. 参考片快剪段 {ref_fast_len:.2f}s 对应 {n128:.2f} 拍（128BPM）
     -> 取整后可用 {round(n128)} 拍 = {round(n128)*PERIOD:.3f}s
  2. 参考片铺垫段 {shots[1]['t1']:.2f}s = {shots[1]['t1']/PERIOD:.2f} 拍（128BPM）
     -> 取整后 {round(shots[1]['t1']/PERIOD)} 拍 = {round(shots[1]['t1']/PERIOD)*PERIOD:.3f}s
  3. 结尾镜 {shots[-1]['dur']:.2f}s = {shots[-1]['dur']/PERIOD:.2f} 拍
     -> {round(shots[-1]['dur']/PERIOD)} 拍
""")

print(f"完整复刻后的时间轴（128BPM 量化）:")
b1 = round(shots[1]["t1"] / PERIOD)
b2 = round(ref_fast_len / PERIOD)
b3 = round(shots[-1]["dur"] / PERIOD)
print(f"  铺垫段   {b1} 拍 = {b1*PERIOD:.3f}s")
print(f"  快剪段   {b2} 拍 = {b2*PERIOD:.3f}s  ({b2} 镜)")
print(f"  结尾镜   {b3} 拍 = {b3*PERIOD:.3f}s")
print(f"  总计     {b1+b2+b3} 拍 = {(b1+b2+b3)*PERIOD:.3f}s")
