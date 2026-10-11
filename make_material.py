# -*- coding: utf-8 -*-
"""拼交错素材（make_material.py）—— 贝特曼表情版的「配方」。

改 `CUTS` 就能换一组表情重拼，不用重来一遍。

**为什么要按这个顺序排**

交错素材是从头**顺序消耗**的：每个素材镜推进 0.633s，22 刀里约 11 个素材镜，
所以一部片**只用前约 5 秒**（不是 7s —— 见下方 `CUTS` 注释里的实测值）。
旧素材 30 秒里前 19 秒全是同一个抽雪茄镜头，成片里因此几乎只有雪茄 ——
素材不缺内容，是**好东西放错了位置**。所以这里把最经典的表情排在最前。

⚠ **排序即"选题"**：排在 5.06s 之后的段落，成片里根本不会出现。
2026-10-09 的改版就是按这条做的 —— 用户指认开头那段不是贝特曼本人，
于是把「大笑 / 脸部特写」这两段本人镜头前移顶上。

**前置步骤**（本脚本不自动做，因为只做一次）：

1. 源先过一遍 `face_crop.py` 把**头位固定住**：
   `python face_crop.py <源> --aspect 1 --lock -o 候选A_方形.mp4`
   ⚠ 必须 `--lock`：源是多镜头拼接，脸的高度每个镜头都不同，
     直接对原源固定裁会切掉半张脸（第一版就这么废的）。
     （`--aspect 1` 本身不生效，真正管用的是 `--lock`。）

2. ⚠ 而 face_crop 的输出是 **360x360 像素但 SAR 9:16（变形像素）**
   —— 它用非方形像素保住了 9:16 的显示比例。若直接当 1:1 素材用，
   归一化链会按 DAR 当成 9:16、铺满整帧，而不是"方形居中 + 上下模糊带"。
   所以下面每个片段都要 `scale=360:640,setsar=1` 先还原成方形像素，再裁 1:1。
"""
import os
import shutil
import subprocess

FF = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
      r"\resources\ffmpeg\ffmpeg.exe")
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "output", "_pipeline", "素材候选")
SRC = os.path.join(D, "候选A_方形.mp4")          # 已做人脸居中方形裁剪
DST = os.path.join(D, "新素材_贝特曼表情_v2.mp4")
# 成品另存一份到 output/（网页默认素材指向它）
DST_OUT = os.path.join(BASE, "output", "交错素材_贝特曼表情_v2.mp4")

# (起, 止, 说明) —— 顺序就是最终素材里的顺序
#
# ⚠ 这些时间点是**逐个抽帧核对过**的锚点，不是看缩略图估的：
#     5.4s=嘟嘴   9.5s=smirk   13.0s=大笑   19.6s=皱眉（经典 sigma face）
#   要避开的：~15-18s 是雨衣中景（做成 360x360 后表情全没），
#             20.2s 之后已经黑了（源实际到 20.5s 左右结束，不是 21s）。
#
# 2026-10-09 改版（用户："0.0 至 2.4 的这些素材替换掉"）：
#   · 原第 1 段（源 6.0-8.2s「侧头点头 + 抬手指」）**整段去掉** ——
#     用户指认它 "根本就不是贝特曼"。
#   · 把原第 6 段「大笑」（源 12.5-13.3，0.8s）与原第 7 段「脸部特写」
#     （源 0.5-2.0，1.5s）**前移到最前**，合计 2.3s ≈ 用户说的 2.4s。
#   · 这两段从原位置移走（不是复制）—— 素材里不留重复内容。
#   → 新素材总长 8.0s（旧 10.2s）。仍然够用：成片只消耗前 5.06s。
CUTS = [
    (12.5, 13.3, "大笑（原第 6 段，前移）"),
    (0.5, 2.0, "脸部特写（原第 7 段，前移）"),
    (8.3, 9.4, "皱眉"),
    (5.0, 5.9, "嘟嘴"),
    (19.3, 20.2, "皱眉 —— 经典 sigma face"),
    (9.3, 10.6, "抿嘴笑 / smirk"),
    (3.0, 4.5, "表情特写"),
]

# ⚠⚠ 这里有两层"坑"，都踩过了，记下来：
#
#   1) 源是 360x640 **竖版**（bilibili 那个本身就是竖版剪辑）。要 1:1 只能竖直裁。
#      **但不能用固定 Y 裁原始源** —— 它是多个镜头的拼接，脸的高度每个镜头都不同，
#      固定裁会把有的镜头切掉半张脸（第一版就是这么废掉的）。
#      → 先用 face_crop.py `--lock` 把**头位固定住**，之后固定裁才成立。
#
#   2) face_crop 的输出是 360x360 像素但 **SAR 9:16（变形像素）** ——
#      它用非方形像素保住了 9:16 的显示比例。若直接当 1:1 素材用，
#      归一化链会按 DAR 当成 9:16 处理、铺满整帧，而不是"方形居中 + 上下模糊带"，
#      与现有素材的观感就不一致了。
#      → 必须 `scale=360:640,setsar=1` 先还原成方形像素，再裁 1:1。
CROP_Y = 175          # 在"360x640 方形像素"坐标系里的裁切起点（实测脸最合适）

parts, labels = [], []
for i, (a, b, _) in enumerate(CUTS):
    parts.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS,"
                 f"scale=360:640,setsar=1,crop=360:360:0:{CROP_Y},"
                 f"fps=30[v{i}]")
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

# 另存一份到 output/ —— 网页默认素材读的是那里。
# ⚠ 用**新文件名**（_v2），不覆盖旧的：旧素材被 v5 等既有 plan.json 引用着，
#   就地覆盖会让那些方案与文件内容悄悄不一致。
shutil.copyfile(DST, DST_OUT)
print(f"已另存 -> {DST_OUT}")

total = sum(b - a for a, b, _ in CUTS)
# 成片实际消耗到哪 —— 不是"素材镜数 × 0.633"：微波炉对（正放+倒放）
# **共用同一个入点**，所以只推进一次。当前配置实测 0–5.695s。
USED = 5.695
print(f"\n新素材 {os.path.basename(DST)}  总长 {total:.1f}s")
print(f"（成片只消耗前 {USED:.2f}s —— 微波炉对共用入点，不是每镜都推进）\n")
acc = 0.0
for a, b, note in CUTS:
    if acc >= USED:
        used = "（进不了片）"
    elif acc + (b - a) <= USED:
        used = "← 进片"
    else:
        used = "← 进片（只用到一部分）"
    print(f"  {acc:5.1f}-{acc + (b - a):5.1f}s  {note}  {used}")
    acc += b - a
if total < USED:
    print(f"\n⚠ 素材只有 {total:.2f}s，短于成片需要的 {USED:.2f}s！"
          f"末尾会被取模回绕、内容重复。")
