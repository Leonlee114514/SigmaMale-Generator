# -*- coding: utf-8 -*-
"""拼交错素材（场景版）—— 从《表情包层出不穷》里挑"有场景感"的贝特曼镜头。

与 `make_material.py`（表情版）的区别
------------------------------------
| | 表情版 | **场景版（本脚本）** |
|---|---|---|
| 源 | `魔性表情_0m21s.mp4` 20.6s / 360×640 | **`表情包层出不穷_2m51s.mp4` 170.9s / 640×360** |
| 内容 | **整条都是脸部特写**，无场景 | 0–21.7s 是脸部特写；**21.7s 之后才有场景** |
| 观感 | "贝特曼一直在做表情" | 全景/中景/近景混排，有场景感 |

**为什么要换**：用户反馈"为啥全是贝特曼做表情的素材啊"——
根因不是裁剪，是**源本身**就是表情合集。

**⚠ 两条硬约束（都是实测出来的）**

1. **片段要短（~0.7s）**。成片每个素材镜只推进 `0.633s`（`0.469 × 1.35`），
   而整片只消耗素材前 ~5.7s。若每段 1.5s，9 个素材镜只覆盖约 4 段 →
   同一段被切好几刀，观感反而重复。**每段 0.7s ≈ 一个素材镜**，
   于是 9 个镜各落一段，变化最多。
2. **裁切要留主体**。源是 640×360（16:9），裁成 1:1 会去掉 44% 宽度。
   实测有些时间点裁完只剩一堵墙（如 44.5s、84.5s），所以下面这些起点
   是**逐个看过 1:1 裁切结果**之后选的，不是按"看起来像场景"估的。

**前置步骤**：无（源本身就是横版原始素材，不需要 face_crop 锁头）。
"""
import os
import shutil
import subprocess

FF = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
      r"\resources\ffmpeg\ffmpeg.exe")
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "output", "_pipeline", "素材候选")
SRC = os.path.join(D, "表情包层出不穷_2m51s.mp4")
DST = os.path.join(D, "新素材_贝特曼场景.mp4")
DST_OUT = os.path.join(BASE, "output", "交错素材_贝特曼场景.mp4")

# 源 640×360 → 中心裁 360×360（x 偏移 = (640-360)/2 = 140）
CROP_X, CROP_S = 140, 360

# ⚠⚠ 段长必须 **≤ 素材镜的推进步长 0.633s**，否则一段会被切好几刀。
#
#   素材镜的入点是 0.000 / 0.633 / 1.266 / 1.898 / …（步长 = 0.469×1.35）。
#   若段长 0.7s，则入点 0.000 与 0.633 都落在**第 1 段**里 → 同一段被切 3 刀
#   （实测 v7 第一段"卧沙发"占了 3 个镜）。
#   取 0.63s（略短于步长）后：**一个素材镜正好落一段**，9 个镜 = 9 个不同场景。
DUR = 0.63

# (起点, 说明) —— 顺序就是最终素材里的顺序，**前几段最重要**（只有前面进片）
CUTS = [
    (24.0, "卧沙发·全景"),
    (30.5, "白椅·全景"),
    (57.0, "酒吧·暗调"),
    (63.5, "走廊·暗调"),
    (78.0, "办公室·中景"),
    (84.5, "对手戏·中景"),
    (51.0, "脸部·近景"),
    (96.0, "房间·中景"),
    (105.0, "中景"),
    # ⚠ 原来这里取 129.0s，抽帧一看是**带血的脸**（126–147s 整段都是血腥镜头）。
    #   虽然它落在进片窗口（5.70s）之外、成片里看不到，但素材一变就可能被用到，
    #   没必要留雷 → 换成干净的 153.0s。
    (153.0, "脸部·近景"),
    (150.0, "中景"),
    (162.0, "暗房间·全景"),
]

parts, labels = [], []
for i, (a, _) in enumerate(CUTS):
    parts.append(f"[0:v]trim=start={a}:end={a + DUR:.3f},setpts=PTS-STARTPTS,"
                 f"crop={CROP_S}:{CROP_S}:{CROP_X}:0,setsar=1,fps=30[v{i}]")
    labels.append(f"[v{i}]")
parts.append("".join(labels) + f"concat=n={len(CUTS)}:v=1:a=0[out]")

cmd = [FF, "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
       "-filter_complex", ";".join(parts), "-map", "[out]",
       "-c:v", "libx264", "-crf", "18", "-preset", "slow",
       "-pix_fmt", "yuv420p", "-r", "30", DST]
r = subprocess.run(cmd, capture_output=True, text=True)
print("退出码", r.returncode, (r.stderr or "")[-400:])
if r.returncode != 0:
    raise SystemExit(1)

shutil.copyfile(DST, DST_OUT)
print(f"已另存 -> {DST_OUT}")

total = len(CUTS) * DUR
USED = 5.695          # 成片实际消耗到哪（微波炉对共用入点，不是每镜都推进）
print(f"\n新素材 {os.path.basename(DST)}  总长 {total:.2f}s（{len(CUTS)} 段 × {DUR}s）")
print(f"（成片只消耗前 {USED:.2f}s，所以**排在前面的段才是真正进片的**）\n")
acc = 0.0
for a, note in CUTS:
    if acc >= USED:
        tag = "（进不了片）"
    elif acc + DUR <= USED:
        tag = "← 进片"
    else:
        tag = "← 进片（一部分）"
    print(f"  {acc:5.2f}-{acc + DUR:5.2f}s  源{a:6.1f}s  {note}  {tag}")
    acc += DUR
if total < USED:
    print(f"\n⚠ 素材只有 {total:.2f}s，短于成片需要的 {USED:.2f}s！")
