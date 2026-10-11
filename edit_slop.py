"""slop 卡点成片器（edit_slop.py）
====================================================================
按参考模板（美国精神病人 / 2022年的剪辑 belike）复刻的自动剪辑：

    时间轴（全部对齐 128BPM 拍网格，phase=0.0677s）
    ├─ 0.00-7.97s  正常速度  实拍（抬头/喝水等动作）        17 拍
    ├─ 7.97-8.44s  0.25x 慢放冲击（瞬间降速，Drop 那一拍）   1 拍
    └─ 8.44-18.75s 1.35x 快剪  实拍 ↔ 素材 交替，每拍一刀     22 拍
       ↑ Drop 起点 = BGM 30.068s（第 64 拍 = 第 16 小节第 0 拍）

    特效（按需求逐条对应）
    ├─ 硬切为主：concat 直接切，不叠转场
    ├─ 重鼓点闪白 0.04s：eq 亮度用 mod 表达式周期性触发，只在 Drop 之后
    ├─ 轻微抖动：crop 偏移用 gte(t,drop) 门控，前奏不加
    ├─ 色差偏移(RGB分离)：水平 5-10px，全局叠加
    └─ VHS 胶片颗粒：noise 强度 12-20，全局，从开头到结尾

关键实现决定（都是踩过坑换来的）：
  1. **先把两个源各自归一化到统一画布，再从归一化结果里 trim**。
     如果反过来（每个片段单独做模糊填充），24 个片段就要跑 24 次
     split+scale+gblur+overlay，耗时翻十几倍。
  2. **闪白用 mod 周期表达式，而不是列 22 个 between()**。
     后者表达式极长且要转义 22 组逗号；前者只有一项、且天然跟随拍网格。
  3. **抖动必须门控**。"只加在 Drop 段"不是靠"只对某些片段应用"，
     而是在全局 crop 里乘一个 gte(t,drop) 的开关 —— 这样它仍然只作用于
     后半段，但不需要把滤镜链拆成两段再拼接。

用法：
    python edit_slop.py --footage 实拍.mp4 --material 素材.mp4
    python edit_slop.py --footage 实拍.mp4 --material 素材.mp4 --aspect square
    python edit_slop.py --footage 实拍.mp4 --material 素材.mp4 --dry-run
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))

import slop_templates as st        # noqa: E402

# 内容感知取段的助手 —— 与 edit_ref.py 用的是**同一套**（见 content_aware.py 文件头）
import content_aware as ca         # noqa: E402

# 成片命名 —— 与 slop_pipeline.py 用的是**同一份**规则
# （`卡点成片.bat` 走本脚本的兜底名，流水线走 --out；两处必须一致）
import film_name as fn             # noqa: E402

FFMPEG = st.FFMPEG
OUT_DIR = os.path.join(BASE, "output")
BGM_DIR = os.path.join(BASE, "bgm")

# ---------------------------------------------------------------- 从分析得到的常量
# 这些值由 analyze_sections.py / diag_refine.py 实测得出，不是猜的：
#   BPM 128（两种独立方法一致，A/B=1.003）
#   相位 0.0677s（只用 Drop 段拟合；整段拟合会被 break 段污染到 54% 对齐率）
#   Drop 起点 30.068s（= 第 64 拍 = 第 16 小节第 0 拍，偏差仅 13.7ms）
BPM = 128.0
PERIOD = 60.0 / BPM                  # 0.46875s
# ⚠ 相位的搜索目标里**必须包含精度（中位偏差）**，不能只看命中率。
#   命中率会**饱和**：Drop 段最多只能压住 71.4% 的底鼓（有些击打本就不在
#   主拍上），所以 0.0677 和 0.1033 两个相位给出**完全相同的命中率 71.4%**，
#   但中位偏差差 8 倍（37.0ms vs 4.4ms）。只看命中率会挑到那个差的。
#   现用值由 refine_phase.py 细扫 20000 个候选、以中位偏差为次要判据选出。
PHASE = 0.103266
DROP_BEAT = 64
BGM_START_BEAT = 47                  # 视频起点 = 第 47 拍 ≈ 22.134s
N_NORMAL_BEATS = 17                  # 正常段 17 拍 = 7.97s
N_FAST_BEATS = 22                    # 快剪段 22 拍 = 10.31s
SLOW_FACTOR = 0.25                   # 慢放冲击倍率
FAST_FACTOR = 1.35                   # 快剪倍率

# Drop 在**成片时间轴**上的位置。
# ⚠ 别混用两条时间轴：beat_time() 给的是 BGM 时间轴（30.068s），
#   而 eq/crop 里的 t 是**成片**时间轴。成片第 0 帧 = BGM 第 47 拍，
#   所以成片里的拍点相位正好是 0（t=0 就是拍点），Drop 落在
#   (64-47) 拍 = 7.96875s。早期版本误把 BGM 时间喂给 eq，闪白会完全错位。
DROP_T_VIDEO = (DROP_BEAT - BGM_START_BEAT) * PERIOD


def beat_time(n):
    return PHASE + n * PERIOD


# ---------------------------------------------------------------- 画布归一化
def normalize_chain(idx, tag, canvas, fps=30):
    """把第 idx 路输入归一化到统一画布（模糊背景填充），返回 (滤镜串, 输出标签)。

    ⚠ 只做一次，后面的 trim 都从它的输出取 —— 这是性能关键（见文件头说明 1）。
    """
    W, H = canvas
    return (
        f"[{idx}:v]split=2[{tag}bg][{tag}fg];"
        f"[{tag}bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},gblur=sigma=28[{tag}bgb];"
        f"[{tag}fg]scale={W}:-2[{tag}fgs];"
        f"[{tag}bgb][{tag}fgs]overlay=(W-w)/2:(H-h)/2,"
        f"setsar=1,fps={fps}[{tag}]"
    )


# ---------------------------------------------------------------- 分镜表
# 快剪段的效果编排：(效果类型, 占几拍)。合计必须 = N_FAST_BEATS。
#
# 为什么用"表"而不是"循环里取模"：
#   素材缺剪辑感，根源是"每拍只是换了个源、画面本身没变"。
#   要出剪辑感，得让**每一镜有自己的处理方式**，而且要有节奏变化
#   （连续两镜同类型会显得平）。写死一张表最直观，也最好调。
#
# 各类型说明：
#   boom  微波炉：**同一段源**正放 + 倒放，占 2 拍。画面"弹回去"，最抓眼。
#   rev   单拍倒放：整镜倒着放，占 1 拍。比 boom 省一拍、也更好塞进节奏里。
#   mirror 单镜左右翻转，五官错位感。
#   split  分屏镜像：左半原图 + 右半镜像，对称构图。
#          ⚠ 2026-09-24 用户看过实片后否掉了（"不好看"）：两边的头被推到
#            画面外缘、中间留一条暗缝，在 9:16 竖屏里信息量反而更低。
#            原语保留在 slop_templates 里，但**不再排进这张表**。
#   zoom   节拍点突然放大再收回（关键帧缩放）。
#   neg    该镜开头 0.12s 色彩反转（负片闪）。
#   plain  不做处理，用来留呼吸感 —— 全是特效反而糊。
#
# 2026-09-24 改版：**分屏镜像下架**（用户看过实片后否决："不好看" ——
#   两边的头被推到画面外缘、中间留一条暗缝，9:16 竖屏里信息量反而更低）。
#   原语仍保留在 slop_templates 里，只是不再排进这张表。
#
#   把原来的 4 个 split(1) 换成 2 个 boom(2)：总拍数不变（22），
#   微波炉从 4 个变 6 个。**用户 2026-09-25 选定这一版（记为 A）**。
#   另一版（4 个单拍倒放 rev）的产物已移进 output/_pipeline_v2/ 备查，
#   换回只需把下面的 boom 换回 rev（拍数仍 22）。
# 2026-10-09 改版（用户："微波炉太多了…缩放脉冲也是"）：
#   微波炉 6 组 → **4 组**（去掉第 2、5 组），缩放脉冲 4 个 → **2 个**
#   （去掉第 2、4 个），腾出的 6 拍全部改成 `plain`（原样）。
#   总拍数仍是 22，且**相邻两镜仍不同类**（plain 的 2 拍连排是有意的"呼吸感"）。
#   被去掉的 4 组位置：原第 6 项 boom→plain(2)、第 8 项 zoom→plain(1)、
#                     原第 11 项 zoom→plain(1)、原第 13 项 boom→plain(2)。
FAST_PLAN = [
    ("boom", 2), ("mirror", 1), ("zoom", 1), ("boom", 2),
    ("neg", 1), ("plain", 2), ("mirror", 1), ("plain", 1),
    ("boom", 2), ("mirror", 1), ("plain", 1), ("neg", 1),
    ("plain", 2), ("mirror", 1), ("zoom", 1), ("boom", 2),
]


def build_timeline(footage_dur, material_dur, plan=None, slices=1):
    """生成分镜列表。

    每项 (源索引, 素材入点, 输出时长, 速度, 效果类型, 是否倒放)。
    速度语义（与 beat_sync.py 一致）：setpts 的乘数 = 1/速度。
    输出时长 × 速度 = 消耗的素材时长。

    `slices` = **每拍切几刀**（切片密度）。
      1 = 每拍一刀（0.469s，跟着底鼓）
      2 = 每半拍一刀（0.234s，"切得极碎"）
    分镜表里的"占几拍"会按比例展开成子切片，效果类型不变 ——
    所以调密度不会打乱效果编排，只是让同一镜里换更多次画面。

    `boom` 会展开成**两个分镜**（正放 + 倒放），它们共用同一段素材，
    所以素材只被消费一次、输出两次。
    """
    plan = plan or FAST_PLAN
    shots = []
    sub = PERIOD / max(1, int(slices))     # 单个切片的输出时长

    # ---- 1) 正常段：实拍，17 拍，1.0x（正常段不切碎，保持铺垫感）----
    t_in = 0.0
    for _ in range(N_NORMAL_BEATS):
        shots.append((0, t_in, PERIOD, 1.0, "plain", False))
        t_in += PERIOD

    # ---- 2) 慢放冲击：Drop 那一拍，0.25x ----
    # 放在实拍素材上（延续上一镜），作为"进入快剪"的缓冲
    shots.append((0, t_in, PERIOD, SLOW_FACTOR, "plain", False))
    t_in += PERIOD * SLOW_FACTOR

    # ---- 3) 快剪段：按分镜表编排，实拍 ↔ 素材 交替 ----
    m_in = 0.0
    src_toggle = 1          # 从素材开始（上一镜是实拍，换源才形成交替）
    for kind, beats in plan:
        n_slices = beats * max(1, int(slices))
        if kind == "boom":
            # 微波炉：正放一半 + 倒放一半（按切片数对半分）。
            #
            # ⚠⚠ **整组必须共用同一个源** —— 这是"弹回去"成立的前提。
            #   早先的实现把 `src_toggle` 放在组内也翻转，于是变成
            #   "正放素材 + 倒放实拍"：两半是**两段无关的镜头**，
            #   "微波炉"名存实亡。实测（卡点_竖屏_v2）：第 1 个 boom 的
            #   正放帧 8.70s 是交错素材、倒放帧 9.10s 是实拍人脸。
            #   这种错不报错、时长也对，只是效果没做出来 —— 与项目里
            #   反复出现的那类静默问题同源。
            #
            #   修法：进组时**只取一次**源、只推进一次源指针，
            #   再让后半按镜像顺序倒着取（slices>1 时也正好成回文）。
            src = src_toggle
            src_toggle = 1 - src_toggle
            half = n_slices // 2
            step = sub * FAST_FACTOR
            base = t_in if src == 0 else m_in
            if src == 0:
                t_in += step * half
            else:
                m_in += step * half
            for j in range(n_slices):
                if j < half:
                    in_pt, rev = base + j * step, False
                else:
                    k = j - half
                    in_pt, rev = base + (half - 1 - k) * step, True
                shots.append((src, in_pt, sub, FAST_FACTOR, "plain", rev))
        else:
            # 单拍倒放（rev）：取段方式与普通镜**完全一样**，只是倒着放。
            # 之所以不放进 `shot_chain`：倒放改的是**取段顺序**，不是画面滤镜，
            # 由 `rev` 标志去触发 `reverse`。kind 仍写成 "rev"（而不是 "plain"），
            # 这样 verify_contentaware 的 `not seg.get("rev")` 会把它排除在
            # 内容匹配之外 —— 时间反过来的画面拿去比内容是无意义的。
            rev = (kind == "rev")
            for _ in range(n_slices):
                src = src_toggle
                src_toggle = 1 - src_toggle
                if src == 0:
                    in_pt = t_in
                    t_in += sub * FAST_FACTOR
                else:
                    in_pt = m_in
                    m_in += sub * FAST_FACTOR
                shots.append((src, in_pt, sub, FAST_FACTOR, kind, rev))

    return shots, t_in, m_in


def shot_chain(kind, reversed_, canvas=None, out_len=None):
    """返回该分镜的**画面处理**滤镜（不含 trim/setpts）。

    这些滤镜都放在 setpts 之后、fps 之前 —— 因为它们大多依赖帧的
    内容/时间，放在变速之后语义更直观（"变速后的这一镜要翻转"）。

    `canvas` 只有 zoom 和 split 用得到：
      - zoompan 的 `s` 必须是**明确尺寸**（实测传 `iwxih` 会报
        `Unable to parse "s" option value`）。
      - hstack 分屏后宽度会翻倍（两路 1080 并排 = 2160），
        **必须 scale 回画布**，否则 concat 报尺寸不匹配：
        `Input link in0:v0 (2160x1920) do not match (1080x1920)`
    """
    if kind == "mirror":
        return st.mirror("hflip")
    if kind == "split":
        return st.split_mirror(size=canvas)
    if kind == "zoom":
        fx = st.zoom_pulse(period=PERIOD, size=canvas)
        # ⚠⚠⚠ zoompan 会**按自己的 fps 重新生成时间戳，完全忽略输入 PTS**。
        #   而本镜的时间轴在上一步 setpts 里已经被压缩过（快剪段 1.35x），
        #   于是 zoompan 输出的是"未压缩"的时长 —— 整段被拉长约 1.35 倍，
        #   后果是**它之后的所有分镜整体后移**。
        #
        #   实测（纯色合成源端到端，红=实拍 蓝=素材，逐格核对）：
        #     plain 40/40 ✓   rev 40/40 ✓   mirror 40/40 ✓   zoom 29/40 ✗
        #   编排里 22 个 zoom 时累积错位 +21 帧（0.7s）—— 后半段整个脱离拍点。
        #
        #   最阴的地方：**它不报错**。总帧数照样 562、切点照样"压在拍上"、
        #   时长照样 18.75s，只有"每一格放的是哪一段源"悄悄错了位。
        #   （这也是"抽帧看到的画面和方案 json 对不上"的真正原因。）
        #
        #   兜底：zoompan 之后**按目标输出时长截断**并重置 PTS。
        #   实测加这一句后 29/40 -> 40/40 全对。
        if out_len:
            fx += f",trim=duration={out_len:.6f},setpts=(PTS-STARTPTS)"
        return fx
    if kind == "neg":
        # 只在该镜开头 0.12s 反转 —— "闪一下"而不是整镜负片
        # setpts 已把本镜时间归零，所以直接用 t 即可
        return st.negate_flash(period=PERIOD, dur=0.12)
    return None



def build_filter(segs, canvas, footage_dur, material_dur, cfg):
    """拼出完整 filter_complex。"""
    parts = [normalize_chain(0, "fw", canvas), normalize_chain(1, "fm", canvas)]

    # ---- 显式 split（必需，否则 concat 会报尺寸不匹配）----
    # ⚠⚠ 这是一个**非常隐蔽**的坑，实测定位过程值得记下：
    #   归一化链的输出 `[fw]` 被 29 个 trim 消费。直觉上让 ffmpeg 自动
    #   插入 split 即可（同一标签多处引用是允许的），但实测报：
    #       Input link in0:v0 parameters (size 1280x720, SAR 1:1)
    #       do not match the corresponding output link (1080x1920, SAR 1:1)
    #   即 concat 收到的第一个输入**退回成了原始分辨率 1280x720**。
    #   原因是：`overlay` 是**多输入滤镜**，它的输出尺寸是由主输入推导出来的
    #   （不是固定参数）。当它被自动 split 时，尺寸推导链断掉，
    #   协商失败就退回源尺寸。
    #
    #   已排除的可能（都实测过）：
    #     ✗ 标签复用本身（简单 scale 链复用标签是 OK 的）
    #     ✗ overlay 之后再加一个显式 scale（照样失败）
    #     ✗ format=yuv420p 钉死（照样失败）
    #     ✓ **显式 split** —— 唯一有效的解法
    #   所以：只要一个多输入滤镜（overlay/blend/concat）的输出要喂给多个下游，
    #   **必须自己写 split，不能依赖自动插入**。
    n_fw = sum(1 for s in segs if s[0] == 0)
    n_fm = sum(1 for s in segs if s[0] == 1)
    if n_fw:
        labs = [f"fwa{i}" for i in range(n_fw)]
        parts.append(f"[fw]split={n_fw}" + "".join(f"[{l}]" for l in labs))
    if n_fm:
        labs = [f"fma{i}" for i in range(n_fm)]
        parts.append(f"[fm]split={n_fm}" + "".join(f"[{l}]" for l in labs))

    # ---- 逐片段 trim + setpts（时长对齐到帧网格）----
    #
    # ⚠⚠ 这里解决的是一个**会毁掉卡点精度**的问题，值得完整说明：
    #
    #   拍周期 0.46875s，在 30fps 下 = **14.0625 帧**（不是整数）。
    #   如果每个片段直接按理想时长切，setpts 后每段实际落在帧网格上的
    #   时长会被四舍五入，误差**逐段累积**。实测 40 段累计漂移 0.15s
    #   （成片 18.60s vs 理想 18.75s），后果是：
    #     - 后半段画面已经偏离 BGM 拍点
    #     - 闪白用 mod(t,period) 判定，但 t 是**输出**时间轴，
    #       漂移后有的拍点整帧错过（实测 22 拍漏 1 拍）
    #
    #   解法：**每个片段的输出起点都吸附到"理想网格的最近帧"**，
    #   而不是"上一段结束的最近帧"。关键区别在于后者会累积误差，
    #   前者每段都重新对齐理想值，误差永远不超过半帧（≈17ms）。
    #   时长随之微调，倍率按"实际输出/素材"反算，保证音画同步。
    fps = float(st.SPEC["fps"])
    ideal, acc = [0.0], 0.0
    for s in segs:
        acc += s[2]
        ideal.append(acc)
    snapped = [round(v * fps) / fps for v in ideal]

    counters = {0: 0, 1: 0}
    labels = []
    for i, (src, t_in, out_dur, speed, kind, rev) in enumerate(segs):
        k = counters[src]
        counters[src] += 1
        tag = f"fwa{k}" if src == 0 else f"fma{k}"
        avail = footage_dur if src == 0 else material_dur

        src_len = out_dur * speed              # 消耗的素材时长（不变）
        out_len = snapped[i + 1] - snapped[i]  # 吸附后的输出时长
        if out_len <= 1e-6:
            out_len = out_dur
        real_speed = src_len / out_len         # 反算实际倍率

        # ⚠⚠ 素材入点必须**向下对齐到帧网格**，否则整段内容晚最多 1 帧。
        #
        #   踩过的坑：入点直接写 `t_in`（= 段序号 × 拍周期），而拍周期
        #   0.46875s 在 30fps 下是 **14.0625 帧**（不是整数）。
        #   `trim=start=t_in` 取的是"PTS ≥ t_in 的第一帧"，于是每段都往后
        #   挪到下一个帧边界 —— 实测偏差与"向上取整量"**完全吻合**：
        #
        #       段   t_in×30   向上取整   预期晚   实测偏差
        #        1   14.0625      15      31.2ms   +31.2ms
        #        2   28.1250      29      29.2ms   +29.2ms
        #        3   42.1875      43      27.1ms   +27.1ms
        #        4   56.2500      57      25.0ms   +25.0ms
        #
        #   23 个切点**全部**晚 1 帧左右，不是随机误差 —— 是系统性偏置。
        #   向下对齐后，trim 从 ≤ t_in 的那一帧开始，随后的 fps=30 会取
        #   离输出时刻最近的那帧，内容就落在正确的时间上了。
        t0_raw = t_in % max(avail - src_len, 0.05) if avail > src_len else 0.0
        # ⚠ epsilon 不能省。`floor(t*fps)/fps` 得到的值在二进制里是
        #   7.966666666666667，而 f-string 的 `:.4f` 会把它写成 "7.9667"
        #   —— 比该帧的实际 PTS 还大，trim 又会把这一帧丢掉，等于没对齐。
        #   减 1e-6 保证「小于等于目标帧」的语义。
        t0 = math.floor(t0_raw * fps) / fps - 1e-6
        t0 = max(0.0, t0)
        lab = f"s{i}"

        # 该镜的画面处理（镜像 / 分屏 / 缩放 / 负片），放在 setpts 之后
        fx = shot_chain(kind, rev, canvas, out_len)

        # ⚠ trim 的 duration 要多给 1 帧余量 —— 但**只在短切片下才起作用**。
        #
        #   实测（probe_pad.py，只改这一个变量）：
        #       slices=1（每镜 0.469s）  pad=0 → 562 帧 ✓   pad=1 → 562 帧 ✓
        #       slices=2（每镜 0.234s）  pad=0 → 518 帧 ✗   pad=1 → 562 帧 ✓
        #                                           ↑ 44 镜各少 1 帧，成片短 1.48s
        #
        #   原因是 `trim=duration` 取到的**最后一帧时间戳是 k/fps**，
        #   所以它实际覆盖的时间跨度比 duration 少最多 1 帧。
        #   长切片时这点差距被后续的帧吸附吸收掉了；短切片时吸收不掉，
        #   于是每镜少 1 帧、逐镜累积。
        #
        #   ⚠ 这里踩过一次归因错误：我曾断言"不给余量就丢 16 帧"，
        #     后来单独复测 slices=1 发现两者都是 562 帧，一度以为
        #     该余量完全没必要、把它删了 —— 结果 slices=2 立刻短了 1.48s。
        #     **只测一个切片密度就下"通用"结论，是错的。**
        #     与 README 里那条教训同源：变量没覆盖全，结论就不能外推。
        #
        # ⚠ setpts 之后必须跟 fps 把该段归一成 CFR：setpts 只改时间戳、
        #   不增减帧，压缩段的帧间隔是不均匀的，concat 拿到会在段边界丢帧。
        # ⚠ reverse 必须放在 setpts **之前**：reverse 反转的是整段的帧序，
        #   放在 setpts 之后语义会乱（时间戳已被压缩）。
        chain = f"trim=start={t0:.4f}:duration={src_len + 1.0/fps:.4f}"
        if rev:
            chain += ",reverse"
        chain += f",setpts=(PTS-STARTPTS)/{real_speed:g}"
        if fx:
            chain += f",{fx}"
        # ⚠ 每镜都必须 setsar=1。
        #   hstack（分屏镜像）会改**采样宽高比**：两路 1:1 并排后变成 SAR 2:1，
        #   于是 concat 报 `size 1080x1920, SAR 2:1 do not match SAR 1:1`。
        #   只补 scale 不够（尺寸对了但 SAR 还是错的），必须显式钉住 SAR。
        #   放在每镜末尾最省心 —— 不用逐个滤镜去判断"它会不会改 SAR"。
        chain += f",setsar=1,fps={fps:g}"

        parts.append(f"[{tag}]{chain}[{lab}]")
        labels.append(lab)

    # ---- 拼接（硬切：concat 本身就是直接切）----
    cat = "".join(f"[{l}]" for l in labels)
    parts.append(f"{cat}concat=n={len(labels)}:v=1:a=0[cat]")

    # ---- 全局特效链 ----
    chain = []

    # 调色：参考片是低饱和偏冷、略带红。用轻量方案，不抢戏。
    if cfg["grade"]:
        chain.append("eq=saturation=0.92:contrast=1.06:brightness=-0.02,"
                     "colorbalance=rs=0.03:bs=-0.04")

    # 降饱和 / 黑白：slop 里常用"降饱和 + 偶尔黑白"制造廉价感。
    # 0 = 不动；1 = 全黑白。
    if cfg.get("bw", 0) > 0:
        chain.append(st.bw(cfg["bw"]))

    # 色差偏移（RGB 分离）：水平 5-10px，全局叠加，淡淡的彩边
    if cfg["rgb"] > 0:
        chain.append(st.rgb_split(shift=cfg["rgb"]))

    # VHS 胶片颗粒：强度 12-20，全局，不能太强否则糊脸
    if cfg["grain"] > 0:
        chain.append(f"noise=alls={cfg['grain']}:allf=t+u")

    # 切换瞬间的模糊过渡。
    #
    # ⚠⚠ 用**空间模糊**（gblur），不要用时间混合（tmix）。
    #
    #   踩过的大坑：一开始用 `tmix=frames=4`（时间维度混合）做"拖影"，
    #   想法是"运动模糊更真实"。但实测它**会把整个切点往后拖**：
    #
    #       模糊帧数   切点中位偏差   40ms 内命中
    #          0        +31.2ms        13/20      ← 关掉最好
    #          2        +64.6ms         1/20
    #          3        +91.7ms         0/20
    #          4        +92.7ms         1/20
    #
    #   一启用就立刻晚 33ms（1 帧），3 帧以上饱和在 +91ms（约 3 帧）。
    #   原因：tmix 的输出是**前 N 帧的加权平均**，画面内容天然滞后。
    #   对卡点成片来说这是致命的 —— 卡点就是靠"切得准"，
    #   为了一个过渡效果牺牲 90ms 的时间精度，本末倒置。
    #
    #   gblur 只做**空间**低通，不缓冲帧，所以零延迟。
    #   观感上不如拖影"运动"，但切点准才是这类片子的命根子。
    #
    # ⚠ gblur 的 sigma **不接受 t 表达式**（实测报
    #   `Undefined constant or missing '('`），
    #   但 `enable` 是时间线选项、可以用表达式 —— 所以门控照旧。
    #   同理门控要**提前收尾**：帧吸附会让切点比拍网格早最多半帧，
    #   窗口若覆盖到切点帧，那一帧就会发虚。
    mode = cfg.get("cut_blur_mode", "spatial")
    if mode != "off" and cfg.get("cut_blur_sigma", 0) > 0:
        lead = cfg.get("cut_blur_dur", 0.12)
        lo = max(0.0, PERIOD - lead)
        hi = PERIOD - 2.0 / float(st.SPEC["fps"])      # 提前 2 帧收尾
        if hi > lo:
            if mode == "temporal":
                # 保留旧实现供对比（已知会拖后 ~90ms，不推荐）
                n = int(cfg.get("cut_blur", 4))
                w = " ".join(str(i + 1) for i in range(n))
                chain.append(
                    f"tmix=frames={n}:weights='{w}'"
                    f":enable='gt(mod(t,{PERIOD:.6f}),{lo:.6f})"
                    f"*lt(mod(t,{PERIOD:.6f}),{hi:.6f})'")
            else:
                sg = cfg["cut_blur_sigma"]
                chain.append(
                    f"gblur=sigma={sg:g}:steps=2"
                    f":enable='gt(mod(t,{PERIOD:.6f}),{lo:.6f})"
                    f"*lt(mod(t,{PERIOD:.6f}),{hi:.6f})'")

    # 闪白：只在 Drop 之后，每拍开头 0.04s
    #
    # ⚠⚠ 三条实测踩出来的规则（probe_expr.py 的矩阵测试结论）：
    #   1. **eq 的变量是小写 `t`，不是大写 `T`**。写 T 会报
    #      `Undefined constant or missing '('`，而且报错信息会误导人
    #      以为是逗号问题（实测 `gte(T,0.5)` 挂、`gte(t,0.5)` 通）。
    #   2. **eq 的引号内逗号不需要转义**。`gte(t,0.5)` / `mod(t,0.46875)`
    #      原样写即可；反而转义成 `\,` 也会失败。
    #   3. **成片时间轴的拍点相位是 0**（因为成片第 0 帧就切在拍点上），
    #      所以直接用 `mod(t,period)`，不需要减相位 —— 顺带避开了
    #      setpts 那个减号坑。
    #   对比：crop 的表达式**必须加引号**，否则逗号会被当成滤镜分隔符；
    #        setpts 则**必须转义成 `\,`** 且变量是大写 `T`。
    #       三个滤镜三套规则，不能互相套用。
    if cfg["flash"] > 0:
        chain.append(
            f"eq=brightness='{cfg['flash']:g}*gte(t,{DROP_T_VIDEO:.4f})"
            f"*lt(mod(t,{PERIOD:.6f}),{cfg['flash_dur']:g})':eval=frame")

    # 抖动：轻微，且用 gte(t,drop) 门控 -> 只作用于 Drop 段
    # ⚠ crop 的 x/y 表达式必须用**单引号**包住（否则逗号被当滤镜分隔符）
    if cfg["shake"] > 0:
        i = int(cfg["shake"])
        chain.append(
            f"crop=iw-{2*i}:ih-{2*i}:"
            f"'{i}*gte(t,{DROP_T_VIDEO:.4f})*(1+sin(n*0.9))':"
            f"'{i}*gte(t,{DROP_T_VIDEO:.4f})*(1+cos(n*0.7))'")
        chain.append(f"scale={canvas[0]}:{canvas[1]}")

    # ⚠ 末尾必须再 setsar=1 一次。
    #   全局链里的 `shake` 用 `crop=iw-10:ih-10` 改过尺寸，之后的
    #   `scale=1080:1920` **不会**把 SAR 重置回 1:1 —— 实测成片
    #   输出 SAR 1712:1719（约 0.4% 横向形变）。
    #   虽然肉眼看不出，但它是错的，而且会在二次转码/拼接时累积。
    chain.append("format=yuv420p")
    chain.append("setsar=1")
    parts.append("[cat]" + ",".join(chain) + "[vout]")
    return ";".join(parts)


def build_audio(bgm, dur, cfg):
    """BGM 侧：从指定拍点截取，末尾淡出。"""
    t0 = beat_time(BGM_START_BEAT)
    fade = cfg["fade"]
    return (f"atrim=start={t0:.4f}:duration={dur:.4f},asetpts=N/SR/TB,"
            f"afade=t=out:st={max(0.0, dur-fade):.4f}:d={fade:g},"
            f"volume={cfg['volume']:g}")


def probe_dur(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    for line in (r.stderr or "").splitlines():
        if "Duration" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def probe_size(path):
    """读分辨率。本机没有 ffprobe，只能解析 ffmpeg -i 的 stderr。"""
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    for line in (r.stderr or "").splitlines():
        if "Stream #" in line and "Video:" in line:
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return 0, 0


def has_audio(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    return "Audio:" in (r.stderr or "")


# ---------------------------------------------------------------- 内容感知取段
def apply_content_aware(segs, peaks, src_dur, slices, peak_pos, intro_pick,
                        weak=ca.WEAK_ACTIONS, pins=None):
    """把实拍位的入点压到动作峰值上 —— 与 edit_ref.py 是**同一套逻辑**。

    段落结构和复刻版不同，但取段逻辑共用 `content_aware` 的函数：

    · **正常段 + 慢放冲击 = 一段连续的源窗口**
      长度 = 17 拍 + 慢放那 1 拍消耗的源 = `N_NORMAL_BEATS×PERIOD + PERIOD×SLOW_FACTOR`
      （慢放那一拍输出 1 拍、只消耗 `PERIOD×SLOW_FACTOR` 的源，所以两段在源上接着）
      → 整段平移，位置由 `pick_intro_offset` 按「给快剪段留下什么」择优。

    · **快剪段的实拍切片**：每片输出 `sub`、速度 `FAST_FACTOR`
      → **消耗的源 = sub × FAST_FACTOR**（默认 0.469×1.35 ≈ 0.633s）

    ⚠⚠ 两个最容易搞错的地方：

      1. **`od` 要用「源消耗」，不是「输出时长」。**
         窗口是开在**源**上的；用输出时长会让窗口整体偏小 1.35 倍，
         「峰值落在镜内 35%」实际变成落在源里 26%，画面上峰值明显偏前。
         但 **`peak_pos` 不用换算**：整窗被均匀压缩，
         源里 35% 处的帧在输出里仍在 35% 处。

      2. **素材侧不参与内容感知。** `m_in` 仍是顺序游标 ——
         要对交错素材做动作检测得先跑参考片，不在本次范围。

    `pins`（可选）= `{槽位下标(0-based): 动作名}`，来自 `--action-plan`。
    直接透传给 `select_anchors` / `pick_intro_offset`（**同一套逻辑**，见
    content_aware 的纪律说明）。落实与否写进逐镜报告的 `pin` / `pin_ok`，
    并汇总到 `stat["n_pin_ok"]`，由 `verify_contentaware.py` 做机器验收 ——
    **指定没落实必须能被看出来**，不允许静默降级。

    返回 (新分镜表, 逐镜报告, 统计)。除实拍位入点外，其它字段一律不动。
    """
    out = list(segs)
    sub = PERIOD / max(1, int(slices))
    od = sub * FAST_FACTOR
    intro_len = N_NORMAL_BEATS * PERIOD + PERIOD * SLOW_FACTOR

    fast_fw = [k for k, s in enumerate(segs)
               if s[0] == 0 and abs(s[3] - FAST_FACTOR) < 1e-9]
    n_shots = len(fast_fw)
    if n_shots == 0:
        return segs, [], {}

    # ---- 0) 把「微波炉对」并成一个槽位 ----
    #   ⚠ 微波炉 = 同一段源正放 + 倒放，两半**必须共用一个入点**。
    #     而下面的吸附是**逐镜独立**做的：不管的话两半会各挑各的峰值，
    #     回文被拆开、"弹回去"就没了。
    #     实测（卡点_竖屏_v4）：对 (29,30) 被吸成 6.115 / 4.790 两个值，
    #     画面变成两段无关的镜头 —— 和修 `build_timeline` 之前是同一个症状，
    #     只是这次"肇事者"换成了内容感知。
    #   识别方法：下一个实拍位的**原始**入点与本位完全相同
    #   （build_timeline 给一对的两半写的就是同一个 in_pt，是可靠指纹）。
    #
    #   ⚠⚠ **不能**再加 `segs[nxt][5]`（rev=True）这个条件 —— 踩过：
    #     验收专用片是用 `--no-fx` 渲染的，而 `--no-fx` 会把所有 rev 置为
    #     False，于是配对识别全部失效、两半被分到不同入点。后果是
    #     **"验收专用片"的方案与成片的方案不是同一份**（实测镜 22/23 被分到
    #     1.039 / 10.758 两个入点），验收等于在验另一部片子 ——
    #     而 `verify_contentaware` 会因此报"未胜出"，看起来像选段错了。
    #     只用入点判断是充分的：非配对镜的入点每镜推进 0.633s，
    #     只有微波炉对的两半才完全相同。
    slots = []
    i = 0
    while i < n_shots:
        k = fast_fw[i]
        nxt = fast_fw[i + 1] if i + 1 < n_shots else None
        if (nxt is not None
                and abs(segs[k][1] - segs[nxt][1]) < 1e-9):
            slots.append((k, nxt))
            i += 2
        else:
            slots.append((k,))
            i += 1
    n_slots = len(slots)

    # ---- 1) 正常段（连续长镜）该放哪 ----
    info = {}
    if intro_pick == "auto":
        off, sel, info = ca.pick_intro_offset(
            peaks, intro_len, max(0.0, src_dur - intro_len),
            od, src_dur, peak_pos, n_slots, 0.0, weak, pins)
    else:
        off = 0.0
        sel = [p for p in peaks if p["t"] <= intro_len + 1e-6]
    for k in range(N_NORMAL_BEATS + 1):        # 17 拍 + 慢放那一拍
        src, it, od_i, sp, kind, rev = out[k]
        if src == 0:
            out[k] = (src, it + off, od_i, sp, kind, rev)

    # ---- 2) 快剪段实拍位：在空档里挑锚点（以**槽位**为单位）----
    spans = ca._spans_for_intro(off, intro_len, src_dur, od, 0.0)
    cand = ca._cands_in_spans(peaks, spans, od, peak_pos)
    chosen = ca.select_anchors(cand, n_slots, od, weak, pins)

    report = []
    used = []                # 已用过的源窗口 —— 兜底时要避开（见 fallback_in_pt）
    for pos, slot in enumerate(slots):
        k0 = slot[0]
        it = out[k0][1]
        want = (pins or {}).get(pos)      # 该槽位被指定的动作（None = 没指定）
        if pos < len(chosen):
            tgt = chosen[pos]["t"]
            new_it = tgt - peak_pos * od
            pk, act, cov = tgt, chosen[pos]["action"], True
            note = ""
        else:
            # ⚠ 兜底**不能直接钳进空档**：空档只剩一个时，所有剩余的镜头位
            #   都会被钳到同一帧 —— 成片里就是"同一个动作连续回放好几遍"。
            #   详见 content_aware.fallback_in_pt 的说明。
            new_it = ca.fallback_in_pt(it + off, od, spans, src_dur, used)
            pk, act, cov = None, None, False
            note = "空档内峰值不够，已避开用过的位置顺序取用"
        # 指定动作是否落实 —— 不落实必须在报告里说清楚（下游 verify 也会查）
        pin_ok = None if want is None else (act == want)
        if want is not None and not pin_ok:
            note = (note + "；" if note else "") + \
                   f"指定「{want}」未落实（实际「{act or '无'}」）"
        new_it = max(0.0, min(new_it, max(0.0, src_dur - od)))
        used.append((new_it, new_it + od))
        # 一个槽位里的所有分镜（微波炉的一对）写同一个入点 —— 回文才成立
        for k in slot:
            src, _, od_i, sp, kind, rev = out[k]
            out[k] = (src, new_it, od_i, sp, kind, rev)
            report.append(dict(shot=k, slot=pos, kind=kind,
                               planned=segs[k][1] + off, in_pt=new_it,
                               peak=pk, action=act, covered=cov,
                               pin=want, pin_ok=pin_ok, note=note))

    n_distinct = len({p["action"] for p in chosen})
    if n_distinct < len(chosen):
        max_dist, avail_dist = ca.max_distinct_actions(peaks, od, peak_pos, spans)
    else:
        max_dist, avail_dist = n_distinct, n_distinct
    # 指定动作落实统计 —— 按**槽位**计（boom 对有两镜，只能算一次）；
    # 越界槽位不计入分母（它们根本没参与挑选，写进分母会让人以为"漏做了一个"）
    by_slot = {r["slot"]: r for r in report}
    n_pin_ok = sum(1 for r in by_slot.values() if r["pin_ok"])
    n_pin_valid = sum(1 for k in (pins or {}) if 0 <= k < n_slots)
    stat = dict(od=od, intro_len=intro_len, intro_off=off, n_shots=n_shots,
                n_slots=n_slots,
                n_distinct=n_distinct, max_distinct=max_dist, avail_distinct=avail_dist,
                n_pin=n_pin_valid, n_pin_ok=n_pin_ok,
                spans=spans, info=info, sel=sel)
    return out, report, stat


def main():
    ap = argparse.ArgumentParser(description="slop 卡点成片器")
    ap.add_argument("--footage",
                    help="实拍素材 mp4。不指定时：有 --clips 就用内容池原片，"
                         "否则用 output/ 里最新的 mp4")
    ap.add_argument("--material", required=True,
                    help="交错素材 mp4（如美国精神病人剪辑）")
    ap.add_argument("--clips",
                    help="clip_extract.py 产出的 reel_map.json —— "
                         "接入**内容感知取段**（与 edit_ref.py 同一套逻辑）："
                         "实拍位不再顺序取，改成压在动作峰值上，且每类动作只用一次")
    ap.add_argument("--peak-pos", type=float, default=0.35,
                    help="动作峰值落在镜内的相对位置（0=镜首，1=镜尾，默认 0.35）")
    ap.add_argument("--intro-pick", default="auto", choices=["auto", "start"],
                    help="正常段位置：auto=按「给快剪段留下什么」择优；"
                         "start=固定从素材 0 秒开始")
    ap.add_argument("--action-plan",
                    help="指定快剪段各实拍位放哪个动作，如 \"3=举手,7=转头\"。"
                         "编号是**槽位**（快剪段第几个实拍位，微波炉对算一个），"
                         "从 1 开始；未指定的槽位照旧自动挑。"
                         "⚠ 必须与 --clips 同用（动作数据来自 reel_map）。"
                         "指定没落实会明确告警，不会被静默忽略")
    ap.add_argument("--dump-plan",
                    help="把分镜表 + 内容感知报告写成 JSON —— "
                         "供 verify_contentaware.py 做**独立验收**。"
                         "不要让验收脚本自己重算一遍方案，那样验的是同一份逻辑、"
                         "等于没验")
    ap.add_argument("--bgm", help="BGM，缺省用 bgm/ 里第一个")
    ap.add_argument("--out", help="输出路径")
    ap.add_argument("--aspect", default="vertical",
                    choices=["vertical", "square"],
                    help="vertical=1080x1920（默认，平台标准）；"
                         "square=1080x1080（参考片是 360x360 方）")
    ap.add_argument("--grain", type=int, default=14,
                    help="VHS 颗粒强度 12-20（默认 14，太大糊脸）")
    ap.add_argument("--rgb", type=int, default=6,
                    help="色差水平偏移 px 5-10（默认 6，0=关）")
    ap.add_argument("--shake", type=int, default=5,
                    help="抖动幅度 px（默认 5，很轻；0=关）")
    ap.add_argument("--flash", type=float, default=0.85,
                    help="闪白亮度增量（0=关）")
    ap.add_argument("--flash-dur", type=float, default=0.05,
                    help="闪白时长秒（默认 0.05）。需求写的是 0.04s，"
                         "但 30fps 下 0.04s = 1.2 帧，边界帧容易整帧错过；"
                         "concat 混合变速后时间戳还有亚帧漂移，"
                         "实测 0.04s 会有 1/22 拍漏闪，0.05s 起稳定覆盖")
    ap.add_argument("--fade", type=float, default=1.2, help="结尾淡出秒数")
    ap.add_argument("--volume", type=float, default=0.9, help="BGM 音量")
    ap.add_argument("--bw", type=float, default=0.0,
                    help="降饱和/黑白程度 0-1（0=不降，1=全黑白）。"
                         "slop 常用 0.35 左右制造廉价感")
    ap.add_argument("--cut-blur", type=int, default=0,
                    help="[旧参数，仅配合 --cut-blur-mode temporal 使用] "
                         "时间混合的帧数")
    ap.add_argument("--cut-blur-mode", default="spatial",
                    choices=["off", "spatial", "temporal"],
                    help="切换瞬间的模糊过渡方式。"
                         "spatial=空间模糊(默认，零延迟)；"
                         "off=关闭；"
                         "temporal=时间混合(拖影更好看，但实测会把切点拖后 ~90ms)")
    ap.add_argument("--cut-blur-sigma", type=float, default=8.0,
                    help="空间模糊强度 sigma（默认 8，0=关）")
    ap.add_argument("--cut-blur-dur", type=float, default=0.12,
                    help="模糊持续秒数（默认 0.12，只盖住切的那一下）")
    ap.add_argument("--slices", type=int, default=1,
                    help="每拍切几刀（切片密度）。1=每拍一刀(0.469s，跟底鼓)；"
                         "2=每半拍一刀(0.234s，切得极碎)。"
                         "调密度不会打乱效果编排，只让同一镜里换更多次画面")
    ap.add_argument("--face-crop", type=float, nargs="?", const=2.0,
                    metavar="ZOOM",
                    help="先把实拍素材做人脸特写再剪辑（默认放大 2.0 倍）。"
                         "特写后脸能占画面 40%%+，信息密度远高于原始全身画面。"
                         "⚠ 素材必须**不含烧死的 HUD** —— 用 "
                         "sigma_realtime.py --bare 录的才行")
    ap.add_argument("--face-safe-y", metavar="TOP:BOTTOM",
                    help="人脸特写的安全带（配合 --face-crop）。"
                         "手上只有带 HUD 的旧素材时用，例：702:1161")
    ap.add_argument("--face-lock", action="store_true",
                    help="人脸特写开启**锁头运镜**（配合 --face-crop）："
                         "头部固定在画面同一位置，背景绕着头转")
    ap.add_argument("--no-fx", action="store_true",
                    help="关闭分镜效果（微波炉/镜像/分屏/缩放/负片），"
                         "回到朴素的每拍一刀")
    ap.add_argument("--no-grade", action="store_true", help="不调色")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印方案与滤镜链，不渲染")
    a = ap.parse_args()

    # ---- 实拍素材来源优先级：--clips 的内容池原片 > --footage > output 里最新的 mp4 ----
    clipmap = None
    if a.clips:
        if not os.path.exists(a.clips):
            print(f"映射表不存在: {a.clips}")
            return 1
        clipmap = ca.load_clipmap(a.clips)
        if clipmap is None:
            return 1
        # 内容感知的峰值坐标全部是**原片时间轴**上的，实拍源必须是同一支原片
        if a.footage and os.path.abspath(a.footage) != os.path.abspath(clipmap["source"]):
            print(f"提示：已用 --clips，实拍源改用内容池原片，"
                  f"忽略 --footage {os.path.basename(a.footage)}")
        a.footage = clipmap["source"]

    # ---- --action-plan：解析"第几个实拍位放哪个动作" ----
    #   ⚠ 格式错、动作名不存在、缺 --clips 一律**当场报错退出** ——
    #     用户写了指定却什么都没发生，是本项目最忌讳的那类静默失败。
    pins = {}
    if a.action_plan:
        if not clipmap:
            print("✗ --action-plan 需要配合 --clips 使用"
                  "（动作数据来自 clip_extract.py 产出的 reel_map.json）")
            return 1
        try:
            pins = ca.parse_action_plan(a.action_plan)
        except ValueError as e:
            print(f"✗ {e}")
            return 1
        avail = sorted({p["action"] for p in clipmap["peaks"]})
        unknown = {k + 1: v for k, v in pins.items() if v not in avail}
        if unknown:
            print(f"✗ --action-plan 里有本素材中不存在的动作：{unknown}")
            print(f"  可用动作（{len(avail)} 种）：{'、'.join(avail)}")
            return 1
        print(f"指定动作：{'、'.join(f'槽位{k+1}={v}' for k, v in sorted(pins.items()))}")
    if not a.footage:
        # 双击场景的兜底：用 output/ 里最新的 mp4。
        # ⚠ 卡点成片.bat 的「直接回车」分支走的就是这里 ——
        #   而 --footage 以前写的是 required=True，那个分支必然被 argparse 挡掉。
        cands = [os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR)
                 if f.lower().endswith(".mp4") and not f.startswith("_")]
        cands = [p for p in cands if os.path.isfile(p)]
        if not cands:
            print(f"没有指定 --footage，且 {OUT_DIR} 里没有可用的 mp4")
            return 1
        a.footage = max(cands, key=os.path.getmtime)
        print(f"未指定 --footage，自动选用 output 里最新的："
              f"{os.path.basename(a.footage)}")

    if not os.path.exists(a.footage):
        print(f"实拍素材不存在: {a.footage}")
        return 1
    if not os.path.exists(a.material):
        print(f"交错素材不存在: {a.material}")
        return 1

    # ---- 可选：先把实拍素材做人脸特写 ----
    # 原始录像里脸只占约 16% 画面高，缩到竖屏后脸部信息量极低。
    # 特写后脸能占 40%+，同样的画布信息密度完全不同。
    if a.face_crop:
        fc_script = os.path.join(BASE, "face_crop.py")
        if not os.path.exists(fc_script):
            print("找不到 face_crop.py，跳过人脸特写")
        else:
            src_dur = probe_dur(a.footage)
            face_out = os.path.join(
                os.path.dirname(os.path.abspath(a.footage)),
                os.path.splitext(os.path.basename(a.footage))[0] + "_face.mp4")
            cmd = [sys.executable, fc_script, a.footage, "-o", face_out,
                   "--zoom", str(a.face_crop)]
            if a.face_lock:
                cmd.append("--lock")
            if a.face_safe_y:
                cmd += ["--safe-y", a.face_safe_y]
            print(f"人脸特写中（放大 {a.face_crop:g}x）...")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode == 0 and os.path.exists(face_out):
                # ⚠ 带 --clips 时，动作峰值坐标是**原片时间轴**上的。
                #   特写片一旦改了帧数/时长/fps，所有坐标会整体错位，
                #   而画面看上去完全正常（"不报错但结果错"）。
                #   所以这里比对时长，超一帧就终止，不允许带错位渲染。
                new_dur = probe_dur(face_out)
                drift = abs(new_dur - src_dur)
                if clipmap and drift > 1.0 / 30 + 1e-6:
                    print(f"  ✗ 特写片 {new_dur:.3f}s 与原片 {src_dur:.3f}s "
                          f"相差 {drift*1000:.0f}ms（> 1 帧）")
                    print(f"    内容感知的峰值坐标全部基于原片时间轴，"
                          f"带错位渲染等于白做 —— 已终止，不出片。")
                    return 1
                print(f"  特写素材 -> {face_out}")
                print(f"    时长 {new_dur:.3f}s / 原片 {src_dur:.3f}s "
                      f"（偏差 {drift*1000:.0f}ms"
                      f"{' ✓ 时序一致' if drift <= 1/30 + 1e-6 else ''}）")
                a.footage = face_out
                if clipmap:
                    # 与 edit_ref.py 同理：峰值坐标基于原片；这里的 face_out 时序一致，
                    # 所以坐标仍然有效，只是源文件换成了特写片。
                    clipmap["source"] = os.path.abspath(face_out)
            else:
                print(f"  人脸特写失败，改用原素材")
                for line in (r.stderr or "").strip().splitlines()[-3:]:
                    print("   ", line[:150])

    bgm = a.bgm
    if not bgm:
        import glob
        cands = []
        for e in ("*.mp3", "*.m4a", "*.wav", "*.aac", "*.opus", "*.flac"):
            cands += glob.glob(os.path.join(BGM_DIR, e))
        if not cands:
            print(f"{BGM_DIR} 里没有 BGM")
            return 1
        bgm = sorted(cands)[0]
    if not os.path.exists(bgm):
        print(f"BGM 不存在: {bgm}")
        return 1

    canvas = (1080, 1080) if a.aspect == "square" else (1080, 1920)
    cfg = {"grain": a.grain, "rgb": a.rgb, "shake": a.shake,
           "flash": a.flash, "flash_dur": a.flash_dur,
           "fade": a.fade, "volume": a.volume,
           "bw": a.bw, "cut_blur": a.cut_blur,
           "cut_blur_mode": a.cut_blur_mode,
           "cut_blur_sigma": a.cut_blur_sigma,
           "cut_blur_dur": a.cut_blur_dur,
           "grade": not a.no_grade}

    f_dur = probe_dur(a.footage)
    m_dur = probe_dur(a.material)

    # ---- 素材信息（尽量详细，避免"用的不是我给的那个文件"这类问题）----
    # ⚠ 这里刻意打印**完整路径**和分辨率。早期只打印文件名，
    #   当 --footage 缺省而回退到"output 里最新 mp4"时，用户无从判断
    #   到底用了哪个文件 —— 出了"我的素材没进去"的问题根本没法自查。
    def _desc(path, dur):
        w, h = probe_size(path)
        return (f"{os.path.basename(path)}  {w}x{h}  {dur:.2f}s"
                f"{'  含音轨(会被 BGM 替换)' if has_audio(path) else ''}")

    print(f"实拍素材 {_desc(a.footage, f_dur)}")
    print(f"         路径 {os.path.abspath(a.footage)}")
    print(f"交错素材 {_desc(a.material, m_dur)}")
    print(f"         路径 {os.path.abspath(a.material)}")
    print(f"BGM      {os.path.basename(bgm)}")

    plan = None if a.no_fx else FAST_PLAN
    segs, used_f, used_m = build_timeline(f_dur, m_dur, plan=plan,
                                          slices=a.slices)

    # ⚠⚠ `--no-fx` 以前是**失效的**（真 bug）：
    #   它只把 `plan` 置成 None，而 build_timeline 里 `plan = plan or FAST_PLAN`
    #   又把 None 还原成 FAST_PLAN，时间轴完全一样；
    #   而真正的效果开关 `shot_chain(kind, rev, canvas)` 无条件执行 ——
    #   于是这个参数**只改了一句打印**，.bat 里宣传的"关掉分镜效果"从来没生效。
    #   现在在这里真的把逐镜效果和倒放清掉（只动观感字段，不动时间轴）。
    if a.no_fx:
        segs = [(s[0], s[1], s[2], s[3], "plain", False) for s in segs]

    # ---- 内容感知：把实拍位入点压到动作峰值上 ----
    ca_report, ca_stat = [], {}
    if clipmap:
        segs, ca_report, ca_stat = apply_content_aware(
            segs, clipmap["peaks"], f_dur, a.slices, a.peak_pos, a.intro_pick,
            pins=pins)
        # 入点变了 → "消耗了多少实拍"要重算。
        # ⚠ 消耗 = 输出时长 × 速度，不是输出时长本身（快剪段是 1.35x）。
        used_f = max((s[1] + s[2] * s[3] for s in segs if s[0] == 0), default=0.0)

    total = sum(s[2] for s in segs)
    fast = [s for s in segs if s[3] == FAST_FACTOR]

    print(f"\n时间轴（BGM {beat_time(BGM_START_BEAT):.3f}s 起，"
          f"共 {total:.3f}s / {total/PERIOD:.0f} 拍）")
    print(f"  正常段   0.000-{N_NORMAL_BEATS*PERIOD:.3f}s  1.00x  实拍  "
          f"{N_NORMAL_BEATS} 拍")
    print(f"  慢放冲击 {N_NORMAL_BEATS*PERIOD:.3f}-"
          f"{(N_NORMAL_BEATS+1)*PERIOD:.3f}s  {SLOW_FACTOR}x  ← Drop "
          f"@ BGM {beat_time(DROP_BEAT):.3f}s")
    print(f"  快剪段   {(N_NORMAL_BEATS+1)*PERIOD:.3f}-{total:.3f}s  "
          f"{FAST_FACTOR}x  {len(fast)} 个分镜"
          f"（每刀 {PERIOD/a.slices*1000:.0f}ms）")
    print(f"  Drop 落在 BGM 第 {DROP_BEAT} 拍 = 第 {DROP_BEAT//4} 小节第 "
          f"{DROP_BEAT%4} 拍")

    # ---- 方案导出（供独立验收）----
    # ---- 成片路径：**在 dump 之前定下来**，好写进方案的 render 段 ----
    #   ⚠ 兜底名也必须能自解释：`卡点成片.bat` 直接调本脚本且**不传 --out**，
    #     原来走的是 `slop_edit_20260925_221530.mp4` 这种时间戳名 ——
    #     完全看不出是什么模式。改用与流水线**同一份**命名规则
    #     （film_name.py），否则两条入口的名字迟早会不一致。
    out = a.out or fn.next_versioned(
        OUT_DIR, fn.film_stem("卡点", a.aspect,
                              face_crop=a.face_crop, face_lock=a.face_lock,
                              face_safe_y=a.face_safe_y,
                              slices=a.slices, bw=a.bw, no_fx=a.no_fx))

    # 与 edit_ref.py 用同一套字段，这样 verify_contentaware.py 两个模板都能验。
    # ⚠ `has_bw` = False：本模板没有**逐镜黑白**（--bw 是全局降饱和），
    #   验收脚本据此跳过黑白项，而不是把"没有黑白镜"误判成"黑白镜全错了"。
    if a.dump_plan:
        acc, shots_out = 0.0, []
        for i, s in enumerate(segs):
            shots_out.append({"i": i, "src": s[0], "in_pt": round(s[1], 4),
                              "out_dur": round(s[2], 4), "speed": s[3],
                              # kind/rev 要写进去：验收脚本靠它判断"这一镜的画面
                              # 有没有被变换过"。镜像/分屏/缩放/倒放会彻底破坏
                              # 像素对应，那种镜做内容匹配是无意义的（不是选段错）。
                              "kind": s[4], "rev": bool(s[5]),
                              "out_start": round(acc, 4)})
            acc += s[2]
        plan = {"source": os.path.abspath(a.footage),
                "material": os.path.abspath(a.material),
                "canvas": list(canvas),
                "period": PERIOD,
                "content_aware": bool(clipmap),
                "has_bw": False,
                "no_fx": bool(a.no_fx),
                "peak_pos": a.peak_pos,
                # ---- 指定动作（--action-plan）----
                #   原样记下来，供**独立验收**核对"指定的槽位是否真的拿到了那个动作"。
                #   键是**槽位号（1-based）**，与用户写的完全一致 ——
                #   不要让验收脚本自己去猜编号口径。没指定时为 {} / None。
                "action_plan": {str(k + 1): v for k, v in sorted(pins.items())},
                "action_plan_ok": (ca_stat.get("n_pin_ok", 0) if pins else None),
                # ---- 渲染参数：让工具**不必再解析文件名**去猜模式 ----
                #   ⚠ 为什么要记：本项目栽过三次"验收/交付的对象悄悄不是同一个"，
                #     根因都是"来源没被记录，只能靠文件名或缓存去猜"。
                #     现在名字和这里都记，两边可互相校验。
                #   ⚠ 没有 `grade` 键：预调色是**流水线层面**的事
                #     （先跑 grade.py 再重指向映射表），编辑器在这一层看不到。
                #     它由文件名里的 `调色` 词体现，实际用的源看上面的 `source`。
                #     （注意别和本脚本的 `--no-grade` 搞混 —— 那是编辑器自己
                #       那道轻调色，是另一回事。）
                "render": {
                    "template": "卡点",
                    "aspect": a.aspect,
                    "face_crop": a.face_crop,
                    "face_lock": bool(a.face_lock),
                    "face_safe_y": a.face_safe_y,
                    "slices": a.slices,
                    "bw": a.bw,
                    "no_fx": bool(a.no_fx),
                    # ⚠ 这里必须用 `a.clips`（原始**路径**），
                    #   不能写 `clipmap` —— 那是 load_clipmap 之后的**字典**，
                    #   拿它去 abspath 会直接 TypeError（踩过）。
                    "clips": os.path.abspath(a.clips) if a.clips else None,
                    "out": os.path.basename(out),
                },
                "snap": ca_report,
                "shots": shots_out}
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_plan)),
                    exist_ok=True)
        with open(a.dump_plan, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"\n方案 -> {a.dump_plan}")

    # ---- 内容感知取段报告 ----
    # 与 edit_ref.py 用同一套助手，但段落结构不同（这里快剪段是 1.35x 变速），
    # 所以报告里把"镜长 od"的来历一起打出来，避免看的人自己换算错。
    if clipmap:
        from collections import Counter
        inf = ca_stat.get("info") or {}
        print(f"\n内容感知 开：{len(clipmap['peaks'])} 个动作峰值"
              f"（原片 {os.path.basename(clipmap['source'])}）"
              f"   峰值落位 {a.peak_pos:.0%}")
        where = ("按「给快剪段留下什么」择优" if a.intro_pick == "auto"
                 else "固定从 0 开始")
        print(f"正常段选窗（{where}）: {ca_stat['intro_off']:.3f} → "
              f"{ca_stat['intro_off'] + ca_stat['intro_len']:.3f}s"
              f"  （连续源 {ca_stat['intro_len']:.3f}s，含慢放那一拍消耗的 "
              f"{PERIOD*SLOW_FACTOR:.3f}s）")
        if inf:
            print(f"  判据是它**留下**什么，不是它**装进**什么：该位置能给快剪段留下 "
                  f"{inf.get('n_chosen', 0)} 个锚点 / "
                  f"{inf.get('n_distinct', 0)} 种不同动作"
                  f"（锚点总强度 {inf.get('strength', 0):.3f}）")
        if inf.get("floor_relaxed"):
            print(f"  ⚠ 为落实指定动作，已**放开「不冷清」护栏**"
                  f"（该位置丰富度 {inf.get('intro_score', 0):.3f} < 门槛 "
                  f"{inf.get('rich_floor', 0):.3f}）—— 用户指定优先于观感护栏")
        cov = [r for r in ca_report if r["covered"]]
        print(f"快剪段实拍位 {len(ca_report)} 个 → 命中动作峰值 {len(cov)}，"
              f"退回顺序取用 {len(ca_report) - len(cov)}")
        print(f"  镜长（源消耗）{ca_stat['od']:.3f}s "
              f"= 输出 {ca_stat['od']/FAST_FACTOR:.3f}s × {FAST_FACTOR}x 变速")
        print(f"    {'分镜':>5}{'入点':>9}{'峰值时刻':>10}  动作")
        for r in ca_report:
            pk = "—" if r["peak"] is None else f"{r['peak']:.3f}"
            print(f"    {r['shot']:>5}{r['in_pt']:>9.3f}{pk:>10}  "
                  f"{r['action'] or r['note']}")

        # ---- 槽位一览：`--action-plan` 的编号就是这里的「槽位」 ----
        #   微波炉对（正放+倒放）在画面上是**一个**单元，所以算**一个**槽位 ——
        #   编号按槽位而不是按分镜，用户写起来才符合直觉。
        by_slot = {}
        for r in ca_report:
            by_slot.setdefault(r["slot"], r)
        print(f"  实拍位（槽位）一览 —— --action-plan 用这个编号，共 {len(by_slot)} 个：")
        print(f"    {'槽位':>4}{'首分镜':>7}  {'效果':<9}{'入点':>9}  {'动作':<9}指定")
        for k in sorted(by_slot):
            r = by_slot[k]
            n_in_slot = sum(1 for x in ca_report if x["slot"] == k)
            eff = "boom" if n_in_slot > 1 else (r["kind"] or "plain")
            eff = {"plain": "原样", "rev": "倒放", "split": "分屏"}.get(eff, eff)
            pin = ("—" if r["pin"] is None
                   else f"指定 {r['pin']} " + ("✓" if r["pin_ok"] else "✗ 未落实"))
            print(f"    {k + 1:>4}{r['shot']:>7}  {eff:<9}{r['in_pt']:>9.3f}  "
                  f"{(r['action'] or '—'):<9}{pin}")
        if pins:
            print(f"  指定落实 {ca_stat['n_pin_ok']}/{ca_stat['n_pin']}"
                  + (" ✓" if ca_stat["n_pin_ok"] == ca_stat["n_pin"]
                     else "  ⚠ 有未落实项（见上表），已退回自动"))
            over = sorted(k + 1 for k in pins if k >= ca_stat["n_slots"])
            if over:
                print(f"  ⚠ 槽位 {over} 超出范围"
                      f"（快剪段只有 {ca_stat['n_slots']} 个实拍位），已忽略")
        acts = [r["action"] for r in ca_report if r["action"]]
        if acts:
            cnt2 = Counter(acts)
            rep = [k for k, v in cnt2.items() if v > 1]
            print(f"  动作类型 {len(set(acts))} 种 / {len(acts)} 个镜"
                  + (f" —— 复用：{'、'.join(rep)}" if rep
                     else " —— **每类只用一次** ✓"))
            if rep:
                sp = " ∪ ".join("[%.2f, %.2f]" % (lo, hi)
                                for lo, hi in ca_stat["spans"])
                print(f"    ⚠ 凑不满 {len(acts)} 种，这是**当前配置下的上限**：")
                print(f"      约束 = 镜头不重叠 + 峰值固定落在镜内 "
                      f"{a.peak_pos:.0%}（无结尾预留，结尾是淡出）")
                print(f"      可放锚点的空档 {sp} 内最多放得下 "
                      f"{ca_stat['max_distinct']} 种不同动作（精确 DP 算的）")

    # ---- 分镜效果编排 ----
    if a.no_fx:
        print("\n分镜效果: 已关闭（--no-fx），朴素每拍一刀")
    else:
        from collections import Counter
        cnt = Counter(k for k, _ in FAST_PLAN)
        label = {"boom": "微波炉(正放+倒放)", "mirror": "镜像",
                 "split": "分屏镜像", "zoom": "缩放脉冲",
                 "neg": "负片闪", "rev": "单拍倒放", "plain": "原样"}
        print("\n分镜效果编排:")
        for k, n in cnt.most_common():
            print(f"    {label.get(k, k):18} × {n}")
        glob = []
        if a.cut_blur_mode != "off" and a.cut_blur_sigma > 0:
            glob.append(f"切换模糊 sigma={a.cut_blur_sigma:g}（{a.cut_blur_mode}）")
        if a.bw > 0:
            glob.append(f"降饱和 {a.bw:g}")
        if a.flash > 0:
            glob.append("闪白")
        if a.shake > 0:
            glob.append("抖动")
        if a.rgb > 0:
            glob.append("色差")
        if a.grain > 0:
            glob.append("颗粒")
        print(f"  全局: {' / '.join(glob) if glob else '无'}")

    # ---- 素材够不够（不够会循环取用，必须显著提示）----
    # ⚠ 早期版本这里只打一行不起眼的提示，用户看到成片里素材在重复，
    #   但不知道为什么。素材长度不足是"素材好像没进去/重复出现"的
    #   头号原因，必须说得足够清楚。
    short = []
    if used_f > f_dur + 0.05:
        short.append(f"实拍素材 需要 {used_f:.2f}s，实有 {f_dur:.2f}s"
                     f"（差 {used_f - f_dur:.2f}s）")
    if used_m > m_dur + 0.05:
        short.append(f"交错素材 需要 {used_m:.2f}s，实有 {m_dur:.2f}s"
                     f"（差 {used_m - m_dur:.2f}s）")
    if short:
        print("\n" + "!" * 60)
        print("⚠ 素材长度不足 —— 不足的部分会**循环重复**：")
        for s in short:
            print(f"    {s}")
        print("  建议：多录一会儿（实拍至少 15s），或用 --footage 指定更长的素材")
        print("!" * 60)

    fc = build_filter(segs, canvas, f_dur, m_dur, cfg)
    af = build_audio(bgm, total, cfg)

    if a.dry_run:
        print(f"\n[filter_complex] {len(fc)} 字符")
        print(fc[:1200] + ("..." if len(fc) > 1200 else ""))
        print(f"\n[audio] {af}")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", a.footage, "-i", a.material, "-i", bgm,
           "-filter_complex", f"{fc};[2:a]{af}[aout]",
           "-map", "[vout]", "-map", "[aout]",
           "-c:v", st.SPEC["vcodec"], "-preset", st.SPEC["preset"],
           "-crf", st.SPEC["crf"], "-pix_fmt", st.SPEC["pix_fmt"],
           "-r", str(st.SPEC["fps"]),
           "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
           "-shortest", out]

    print(f"\n渲染中（{len(segs)} 个片段）...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    el = time.time() - t0

    if r.returncode != 0 or not os.path.exists(out) \
            or os.path.getsize(out) < 1024:
        print(f"失败 (rc={r.returncode}):")
        for line in (r.stderr or "").strip().splitlines()[-8:]:
            print("   ", line[:160])
        return 1

    od = probe_dur(out)
    print(f"完成 -> {out}")
    print(f"  {canvas[0]}x{canvas[1]}  {od:.2f}s  "
          f"{os.path.getsize(out)/1024/1024:.1f} MB  耗时 {el:.1f}s")
    if abs(od - total) > 0.4:
        print(f"  ⚠ 时长偏差: 期望 {total:.2f}s 实得 {od:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
