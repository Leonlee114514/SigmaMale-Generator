"""客观验证闪白是否真的压在拍点上。

不看观感 —— 观感无法评审也无法回归。做法：
    解码成片 → 逐帧平均亮度 → 找"明显比邻帧亮"的帧 → 检查间距是否为拍周期。
如果闪白没落在拍点上，间距会乱；如果没触发，一个亮点都找不到。
"""
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import edit_slop as es       # noqa: E402
import film_name as fn       # noqa: E402

# 待验证的成片：优先用命令行参数，否则用 output 里最新的一个成片。
# ⚠ 早期版本把路径写死成临时测试文件（output/_test/...），
#   清理临时目录后就报"解码失败"。
# ⚠ 后来改成"演示成片 demo_卡点成片_*.mp4"，那批文件也早没了；
#   而再往下的兜底是"按 mtime 取最新 mp4" —— 那会取到**刚录的素材**
#   （`录制_*.mp4` 也在 output/ 里），然后报"闪白 0 个"之类看不懂的错。
#   现在统一用 film_name.newest_film：判据是"有成片必有的 .plan.json 伴生文件"。
def _default_video():
    return fn.newest_film(es.OUT_DIR)


VID = sys.argv[1] if len(sys.argv) > 1 else _default_video()
if not VID or not os.path.exists(VID):
    print("找不到待验证的成片。用法: python verify_flash.py <成片.mp4>")
    sys.exit(1)


def frame_brightness(path, w=160, h=284):
    """用 ffmpeg 输出灰度原始帧，算每帧均值。"""
    cmd = [es.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
           "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("解码失败")
    n = len(r.stdout) // (w * h)
    a = np.frombuffer(r.stdout[:n * w * h], dtype=np.uint8)
    a = a.reshape(n, w * h).astype(np.float32)
    return a.mean(axis=1)


b = frame_brightness(VID)
fps = 30.0
print("=" * 72)
print("闪白压点验证")
print("=" * 72)
print(f"成片 {os.path.basename(VID)}  共 {len(b)} 帧  {len(b)/fps:.2f}s")

# ---- 找闪光帧 ----
# ⚠ 判据必须是"接近纯白"，不能用"比邻帧亮"。
#   快剪段每拍都在硬切，而两个源的亮度差极大（测试卡 ~150 vs 素材 ~60），
#   于是**每一次硬切都会产生亮度跳变**，用相对判据会把 22 次切点全误判成闪光
#   （实测检出 30 个"事件"、间距 342ms，看着像"闪了两次"）。
#   闪白的物理特征是 eq 把亮度 +0.85，暗画面也会被推到接近 255，
#   所以用**绝对阈值**最可靠。
FLASH_LEVEL = 235
spikes = [i for i, v in enumerate(b) if v > FLASH_LEVEL]

print(f"\n检出闪光帧 {len(spikes)} 个（判据：帧均值亮度 > {FLASH_LEVEL}）")
if spikes:
    times = np.asarray(spikes) / fps
    print("  闪光帧时刻: " + ", ".join(f"{t:.3f}" for t in times[:30]))

    # 0.04s 的闪白在 30fps 下会跨到 2 帧（0.04*30=1.2），
    # 所以要先把连续帧聚成"闪光事件"，再测事件间距。
    events, cur = [], [times[0]]
    for t in times[1:]:
        if t - cur[-1] < 0.10:
            cur.append(t)
        else:
            events.append(cur[0])
            cur = [t]
    events.append(cur[0])
    events = np.asarray(events)

    print(f"\n聚合后闪光事件 {len(events)} 个（每拍应 1 个）")
    print("  事件时刻: " + ", ".join(f"{t:.3f}" for t in events[:26]))
    gaps = np.diff(events)
    if len(gaps):
        print(f"\n  事件间距: 均值 {gaps.mean()*1000:.1f}ms  "
              f"标准差 {gaps.std()*1000:.1f}ms")
        print(f"  拍周期应为 {es.PERIOD*1000:.1f}ms")
        dev = np.abs(gaps - es.PERIOD) * 1000
        print(f"  与拍周期偏差: 均值 {dev.mean():.1f}ms  最大 {dev.max():.1f}ms")

    before = [t for t in events if t < es.DROP_T_VIDEO - 0.05]
    after = [t for t in events if t >= es.DROP_T_VIDEO - 0.05]
    print(f"\n  Drop 前({es.DROP_T_VIDEO:.3f}s 之前)闪光 = {len(before)} 个  "
          f"<- 应为 0（需求：前奏不加）")
    print(f"  Drop 后闪光 = {len(after)} 个")
    ok = (len(before) == 0 and len(events) >= 15
          and (len(gaps) == 0 or abs(gaps.mean() - es.PERIOD) < 0.06))
    print(f"\n  判定: {'通过' if ok else '不通过'}")
else:
    print("  未检出闪光 —— 闪白可能没生效")
