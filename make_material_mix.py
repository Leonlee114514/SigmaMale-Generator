# -*- coding: utf-8 -*-
"""拼交错素材（混剪版）—— 从**四个源**里挑贝特曼镜头，按模板生成对应段长。

为什么要跨源拼
--------------
用户看过各候选源的对比图后选定：**第 1/2/3/5 行四个源都要**
（第 4 行《美国精神硬人》是**特效版**，脸已被硬汉变形过 → 不能用）。

| 源 | 文件 | 时长/尺寸 | 提供什么 |
|---|---|---|---|
| ①脸 | `素材候选/魔性表情_30fps.mp4` | 20.6s / 360×640(9:16) | 脸部表情（**原片**）——⚠ **当前未使用**，见约束 3 |
| ②景 | `素材候选/表情包层出不穷_2m51s.mp4` | 170.9s / 640×360 | 沙发/办公室/酒吧/走廊等场景 |
| ③净 | `sigma face/【4K 60FPS】…Patrick Bateman.mp4` | 99.5s / 852×480 | 干净原片：西装、走廊、近景 |
| ④牌 | `sigma face/2022年的剪辑 belike.mp4` | 30.0s / 360×360 | 招牌镜头：**面具**、礼服、门廊剪影 |

**⚠ 四条硬约束（都是实测出来的，改之前先读）**

1. **两个模板必须各做一份素材。** 素材镜的入点是等步长推进的，段长必须
   **≤ 入点步长**（否则一镜跨两段、画面中途跳一下），又必须
   **≥ 单镜时长**（否则一镜被切成两段）。两个模板的步长不同且不可通约：

   | 模板 | 单镜时长 | 入点步长 | 段长 = |
   |---|---|---|---|
   | 卡点 `edit_slop` | 0.46875 (=60/128) | 0.6328125 (=×1.35) | **0.6328125** |
   | 复刻 `edit_ref`  | 0.8939213 (=60/67.12) | 同左 | **0.8939213** |

   ⚠ 复刻版的**单镜时长恰好等于步长**，所以段长必须**精确**等于 0.8939213
   （取整到 0.894 会让每镜尾巴带出下一段的一帧）。卡点版镜短于步长，有余量。

2. **顺序即"进片优先级"**。卡点版消耗素材前 ~5.7s（≈9 段）、复刻版前 ~7.2s
   （≈8 段）。排在后面的段根本进不了片 —— 所以好料必须排前面。

3. **每个源都要单独处理画幅**，因为四个源的画幅各不相同，而素材统一是
   **4:3（480×360）**：
   - 比 4:3 **宽**的源（②③ 16:9）→ 居中裁 480×360（保留 75% 宽）
   - 比 4:3 **窄**的源（① 9:16、④ 1:1）→ **整幅放入 + 同帧模糊填边**
     （`_fill()`），不留黑边
   ⚠ ① 号源用**原片** `魔性表情_0m21s.mp4`（脚本自动归一化到
   `魔性表情_30fps.mp4`，24→30fps 补帧，几何不变）。
   2026-10-10 之前用的是 `候选A_方形.mp4`（face_crop 锁头派生版），
   用户反馈「没必要整个画面都用贝特曼的脸撑满」—— 对比实测：
   锁头版把脸放大了且上下带黑边；原片是头+肩+西装。所以换回原片。
   ⚠ ① 号源（9:16 竖版）**当前没有任何段在用**：2026-10-11 用户反馈
   「有个素材是竖屏的，确实看上去怪怪的」，要求换横屏 —— 它的两段
   （脸部特写 / 大笑）已换成 ③ 与 ② 的横屏镜头。条目与 30fps 归一化保留着，
   要换回去只需把 CUTS 里那两段的源下标改回 0。
   ⚠ 原片是 24fps，必须归一化到 30fps：素材要按**帧号**精确裁剪，
     24fps 会让段帧数算不准（见下面 trim 处说明）。

4. **挑镜头必须逐点看过裁切结果**：横版源裁 4:3 要去掉 25% 宽度，
   ② 源还带**烧死的中文字幕**、④ 源左上角带 **bilibili 水印**、③ 源个别点带
   英文字幕或血腥镜头。下表的时间点都是逐个抽帧核对过的。

5. **取用窗口必须整窗落在同一个镜头内**（镜内无切点）。
   素材镜是从素材里切一段连续画面，若那 0.63s/0.89s 里源片自己换了镜头，
   成片里就会"一镜中途换场景"。四个源都是快剪素材，切点很密
   （① 8 个 / ② 43 个 / ③ 55 个 / ④ 22 个），随手取的时间点很容易踩到。
   所以起点不是"看着好看"定的，而是先用切点分析找出每个镜头的区间，
   再在区间内取点。改时间点请重跑这套分析。
"""
import argparse
import math
import os
import shutil
import subprocess

FPS = 30

# 素材画幅：**4:3（480×360）**
# ⚠ 2026-10-11 从 1:1(360×360) 放宽到 4:3。原因：用户反馈
#   「整屏全是人脸看起来怪怪的」—— 1:1 方裁会把 16:9 的源裁掉 44% 宽度，
#   只剩中间一张脸（②③ 都变成脸部特写）。4:3 只裁 25%，取景明显放宽。
#   窄于 4:3 的源（① 9:16、④ 1:1）用**同帧模糊填充**补边，不留黑边。
MAT_W, MAT_H = 480, 360

FF = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
      r"\resources\ffmpeg\ffmpeg.exe")
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "output", "_pipeline", "素材候选")


def _fill(tag, blur=18):
    """把当前流等比缩放进 MAT_W×MAT_H，四周用**同帧模糊**填充（不留黑边）。

    ⚠ 标签必须按段号唯一（`{tag}` 会替换成段号），否则 filter_complex 里
      同名标签会冲突。
    ⚠ 返回串的**最后一段不能带输出标签** —— 外层会在后面接
      `,setsar=1,fps=30[vN]`。
    """
    return (f"split=2[{tag}bg][{tag}fg];"
            f"[{tag}bg]scale={MAT_W}:{MAT_H}:force_original_aspect_ratio=increase,"
            f"crop={MAT_W}:{MAT_H},gblur=sigma={blur}[{tag}bgb];"
            f"[{tag}fg]scale={MAT_W}:{MAT_H}:force_original_aspect_ratio=decrease"
            f"[{tag}fgs];"
            f"[{tag}bgb][{tag}fgs]overlay=(W-w)/2:(H-h)/2")

# ---------------------------------------------------------------- 段长 / 输出
# 数值来源：edit_slop.PERIOD(60/128) × FAST_FACTOR(1.35)；edit_ref.REF_PERIOD(60/67.12)
#   advances = 素材游标**推进次数**（微波炉对正放+倒放共用入点，只推进一次），
#              也就是实际用到的段数。实测：卡点版 9 次（10 个素材镜）、复刻版 8 次。
TEMPLATES = {
    "slop": dict(seg=60.0 / 128 * 1.35,
                 out=os.path.join(BASE, "output", "交错素材_贝特曼混剪.mp4"),
                 advances=9, name="卡点版"),
    "ref": dict(seg=60.0 / 67.12,
                out=os.path.join(BASE, "output", "交错素材_贝特曼混剪_复刻.mp4"),
                advances=8, name="复刻版"),
}

# ① 号源的 30fps 归一化版（由本脚本生成；见文件头约束 3）
# ⚠ 原片是 24fps，而素材要按**帧号**精确裁剪（见下面 trim 处说明），
#   所以先统一到 30fps CFR。几何不变，只是补帧。
SQ_SRC = os.path.join(D, "魔性表情_0m21s.mp4")
SQ_DST = os.path.join(D, "魔性表情_30fps.mp4")

# ① 的裁切窗口（旧 1:1 方案用；现在 4:3 是整幅放入 + 模糊填边，不再裁）
# ⚠ ①（9:16 竖版）**目前没有任何段在用** —— 2026-10-11 用户反馈
#   「有个素材是竖屏的，确实看上去怪怪的」，要求换成横屏的。
#   它的两段（脸部特写 / 大笑）已换成 ③ 与 ② 的横屏镜头。
#   条目保留着：换回去只要把 CUTS 里的源下标改回 0 即可（30fps 版会自动生成）。
SOURCES = [
    # ① 原片 360×640（9:16，比 4:3 窄）→ **整幅放入** + 左右模糊填（当前未使用）
    dict(path=SQ_DST, prep="{fill}"),
    # ② 640×360（16:9，比 4:3 宽）→ 居中裁 432×324（4:3）
    #   ⚠ 必须**裁掉顶部 36px**：该源右上角有烧死的「周大福」水印
    #     （实测在画面最上方约 15px 高，取 36px 留足余量）。
    dict(path=os.path.join(D, "表情包层出不穷_2m51s.mp4"),
         prep=f"crop=432:324:104:36,scale={MAT_W}:{MAT_H}"),
    # ③ 852×480（16:9）→ 居中裁 580×435（4:3）再缩到 480×360
    #   ⚠ 同样必须**裁掉顶部 45px**：该源右上角有烧死的中文字幕
    #     （「黑洞打了黑…」，实测约占最上方 25px）。
    dict(path=os.path.join(BASE, "sigma face",
                           "【4K 60FPS】“My name is Patrick Bateman”.mp4"),
         prep=f"crop=580:435:136:45,scale={MAT_W}:{MAT_H}"),
    # ④ 360×360（1:1，比 4:3 窄）：先裁掉左上角 bilibili 水印
    #   （实测 x≈10–105 / y≈13–33，360×360 坐标）→ 再整幅放入 + 左右模糊填
    dict(path=os.path.join(BASE, "sigma face", "2022年的剪辑 belike.mp4"),
         prep="crop=325:325:15:35,{fill}"),
]

# (源下标, **起始帧号**, 说明) —— 顺序即进片优先级
#
# ⚠ 用**帧号**而不是秒：四个源都是 30fps CFR 且 start=0，帧号是精确的，
#   而秒会因为浮点边界把某些段多裁 1 帧（见下面 trim 处的说明）。
#
# ⚠ 每个取用点都满足：
#   ① **整窗（27 帧 = 复刻版段长）落在同一个镜头内** —— 否则成片里"一镜中途
#      换场景"。四个源切点都很密（①12 ②43 ③55 ④22 个），随手取点极易踩到；
#      所以这些帧号是先分析出每个镜头的帧区间、再在区间内取的。
#   ② 清晰度不低于阈值（避开失焦/糊帧）；③ 无烧死字幕 / 水印（见约束 4）。
#   这 12 个点是用切点分析 + 逐帧摊开图逐个核过的，别凭感觉改。
CUTS = [
    (3, 511, "面具（招牌）"),      # 17.033s ④ 青绿眼罩，镜头区间恰好 511–538
    (2, 612, "脸部·近景"),         # 20.400s ③ 干净原片
    (1, 1905, "走廊"),             # 63.500s ② 场景
    (2, 689, "西装·中景"),         # 22.967s ③ 横屏中景（原 ① 脸部特写，竖版换掉）
    (2, 1311, "西装·近景"),        # 43.700s ③ 干净原片
    (3, 565, "西装·微笑"),         # 18.833s ④ 镜头区间恰好 565–592
    # ⚠ 原为 ③72.0s「走廊·暗」——抽帧发现是**一帧失焦的城市光斑**（废帧），
    #   且落在两个模板的进片窗口内（卡点第 7 段 / 复刻第 7 段）。
    #   换成 ③52.0s「街头·西装」：干净、有场景感、与相邻段不重复。
    (2, 1560, "街头·西装"),        # 52.000s ③ 干净原片（替换废帧）
    (2, 1015, "室内·西装"),        # 33.833s ③ 横屏中景（原 ②169.067 只有下半身）
    (2, 1199, "椅中·近景"),        # 39.967s ③ 干净原片
    # ---- 以上 9 段 = 卡点版的进片范围（0–5.70s）；以下为余量 ----
    #      ⚠ 复刻版每镜 0.894s，会一直吃到第 8 段，所以下面也要干净
    (1, 4860, "沙发·中景"),        # 162.000s
    (2, 2280, "走廊·白"),          # 76.000s
    (1, 1725, "酒吧·暗调"),        # 57.500s
]


def ensure_30fps():
    """把 ① 号源（原片，24fps）一次性归一化到 30fps CFR（幂等）。

    几何不变（只补帧），但让「按帧号裁剪」成为可能。
    """
    if os.path.exists(SQ_DST) and (os.path.getmtime(SQ_DST)
                                   > os.path.getmtime(SQ_SRC)):
        return
    print(f"生成 ① 号源的 30fps 版 -> {os.path.basename(SQ_DST)}")
    r = subprocess.run(
        [FF, "-hide_banner", "-loglevel", "error", "-y", "-i", SQ_SRC,
         "-vf", "fps=30,setsar=1", "-c:v", "libx264", "-crf", "16",
         "-preset", "slow", "-pix_fmt", "yuv420p", SQ_DST],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("  归一化失败:", (r.stderr or "")[-300:])
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(
        description="按模板段长拼混剪素材（一镜一段）")
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="slop",
                    help="目标模板：slop=卡点版(段长0.6328) / ref=复刻版(段长0.8939)")
    ap.add_argument("--seg", type=float, default=None,
                    help="段长（秒）；默认取模板值，一般不用改")
    ap.add_argument("--out", default=None, help="输出路径；默认按模板命名")
    a = ap.parse_args()

    tpl = TEMPLATES[a.template]
    DUR = a.seg if a.seg else tpl["seg"]
    OUT = a.out or tpl["out"]
    tmp = os.path.join(D, "新素材_贝特曼混剪.mp4")

    ensure_30fps()

    # ---- 自检：段长必须落在 [单镜时长, 入点步长] 内 ----
    shot = 0.46875 if a.template == "slop" else tpl["seg"]
    step = tpl["seg"]
    if DUR > step + 1e-9:
        print(f"⚠ 段长 {DUR:.7f} > 入点步长 {step:.7f}：一镜会跨两段")
    if DUR < shot - 1e-9:
        print(f"⚠ 段长 {DUR:.7f} < 单镜时长 {shot:.7f}：一镜会被切成两段")

    # ---- 段帧数：必须让段边界与素材镜入点网格 (k×DUR) 严格对齐 ----
    # ⚠ 入点步长往往不是帧长的整数倍（复刻版 0.8939213×30 = 26.8176 帧）。
    #   若每段都取 27 帧，段边界落在 27k 帧上，而镜 k 的入点在 26.8176k 帧上，
    #   累计漂移到第 6 镜就多出 1 帧 —— 成片里表现为镜首闪一下上一段的画面。
    #   所以段 k 的帧数取 ceil((k+1)·DUR·fps) - ceil(k·DUR·fps)（26 或 27 帧）。
    def seg_frames(k):
        return (int(math.ceil((k + 1) * DUR * FPS - 1e-9))
                - int(math.ceil(k * DUR * FPS - 1e-9)))

    inputs = []
    for s in SOURCES:
        inputs += ["-i", s["path"]]

    # ⚠ 用 **start_frame/end_frame**（帧号）而不是 start/end（秒）。
    #   按秒裁时 `t + n/30` 的浮点误差可能刚好跨过下一帧的 PTS，
    #   使某些段多收 1 帧 —— 实测卡点版因此变成 232 帧（应为 228），
    #   段边界逐段漂移，成片里镜首会闪出上一段的画面。
    #   四个源都是 30fps CFR 且 start=0，帧号 = round(t×30)，按帧裁是精确的。
    parts, labels = [], []
    for i, cut in enumerate(CUTS):
        si, f0 = cut[0], cut[1]
        base = cut[3] if len(cut) > 3 else SOURCES[si]["prep"]   # 可按段覆盖
        prep = base.replace("{fill}", _fill(f"s{i}"))            # 填边标签按段唯一
        n = seg_frames(i)
        parts.append(
            f"[{si}:v]trim=start_frame={f0}:end_frame={f0 + n},"
            f"setpts=PTS-STARTPTS,{prep},setsar=1,fps={FPS}[v{i}]")
        labels.append(f"[v{i}]")
    parts.append("".join(labels) + f"concat=n={len(CUTS)}:v=1:a=0[out]")

    cmd = [FF, "-hide_banner", "-loglevel", "error", "-y", *inputs,
           "-filter_complex", ";".join(parts), "-map", "[out]",
           "-c:v", "libx264", "-crf", "18", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-r", "30", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("退出码", r.returncode, (r.stderr or "")[-400:])
    if r.returncode != 0:
        raise SystemExit(1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shutil.copyfile(tmp, OUT)

    TAG = ["①脸", "②景", "③净", "④牌"]
    fr = [seg_frames(k) for k in range(len(CUTS))]
    total = sum(fr) / FPS
    print(f"\n[{tpl['name']}] {os.path.basename(OUT)}")
    print(f"  {len(CUTS)} 段 / {sum(fr)} 帧 = {total:.3f}s  来自 {len(SOURCES)} 个源")
    print(f"  单镜 {shot:.5f}s / 入点步长 {step:.7f}s / 段长 {DUR:.7f}s")
    print(f"  段帧数 {sorted(set(fr))} 帧（按入点网格分配，保证边界对齐）")
    need = tpl["advances"] * step
    print(f"  成片需消耗前 {need:.2f}s（素材游标推进 {tpl['advances']} 次）"
          f"{'  ✓ 够' if total >= need else '  ✗ 不够!'}\n")

    # 逐段列出：段号 / 帧数 / 时间 / 内容 / 落在哪个素材镜
    acc = 0.0
    for i, cut in enumerate(CUTS):
        si, f0, note = cut[0], cut[1], cut[2]
        tag = (f"← 素材镜#{i + 1}" if i < tpl["advances"] else "（进不了片）")
        d = fr[i] / FPS
        print(f"  段{i + 1:>2} {fr[i]:>2}帧 {acc:5.2f}-{acc + d:5.2f}s  "
              f"{TAG[si]} 源{f0/30:7.3f}s(#f0)  {note}  {tag}")
        acc += d
    print("\n（素材镜 k 的入点 = k×段长，恰好落在第 k 段的第 1 帧 = 一镜一段）")


if __name__ == "__main__":
    main()
