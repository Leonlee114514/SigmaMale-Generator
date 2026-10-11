"""导出分镜时刻表：每镜的时间范围 + 效果 + 源，便于精确抽帧核对。"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))
import edit_slop as es       # noqa: E402

FOOT = os.path.join(BASE, "output", "my_footage.mp4")
MAT = r"C:/Users/Lyonl/Desktop/2022年的剪辑 belike.mp4"
f_dur, m_dur = es.probe_dur(FOOT), es.probe_dur(MAT)
segs, _, _ = es.build_timeline(f_dur, m_dur)

label = {"boom": "微波炉", "mirror": "镜像", "split": "分屏镜像",
         "zoom": "缩放脉冲", "neg": "负片闪", "plain": "原样"}

print(f"{'#':>3} {'起':>7} {'止':>7} {'时长':>6} {'源':>5} {'效果':>8} 备注")
print("-" * 60)
t = 0.0
for i, (src, t_in, out_dur, speed, kind, rev) in enumerate(segs):
    t0, t1 = t, t + out_dur
    t = t1
    if t0 < 8.0:
        continue          # 正常段太长，跳过
    srcn = "实拍" if src == 0 else "素材"
    note = "倒放" if rev else ""
    print(f"{i:3} {t0:7.3f} {t1:7.3f} {out_dur:6.3f} {srcn:>5} "
          f"{label.get(kind, kind):>8} {note}")

print("\n便于抽帧的采样点（每镜中点）：")
t = 0.0
pts = []
for i, (src, t_in, out_dur, speed, kind, rev) in enumerate(segs):
    mid = t + out_dur / 2
    t += out_dur
    if mid >= 8.2 and kind != "plain":
        pts.append((mid, kind))
print("  " + "  ".join(f"{m:.2f}:{k}" for m, k in pts[:14]))
