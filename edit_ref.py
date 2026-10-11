"""参考片复刻卡点器（edit_ref.py）
====================================================================
目标：**完全复刻** `C:\\Users\\Lyonl\\Desktop\\2022年的剪辑 belike.mp4` 的剪辑结构，
     只把其中的实拍位替换成用户自己的摄像头素材。

与 edit_slop.py 的区别
----------------------
edit_slop.py 是"按 128BPM 那首 gigachad phonk 卡点"的自创编排；
本脚本是**照搬参考片自己的结构**，包括：

  ├─ BGM        = 参考片的音轨（bgm/ref_bgm.m4a，30.04s）
  ├─ 拍网格     = 67.12 BPM / 893.9ms（实测裁定，见 decide_ref_grid.py）
  ├─ 铺垫段     = 9 拍（实拍，替代参考片的 2 个长镜）
  ├─ 快剪段     = 15 镜 × 1 拍（实拍 ↔ Bateman 素材交替）
  │              ⚠ 参考片实测是真的 **每镜正好 1 拍**（中位偏差 15.1ms，命中 88%）
  ├─ 黑白序列   = 按参考片逐镜实测照搬（第 5/6/11/14 镜加黑白滤镜）
  └─ 结尾       = 淡出到黑（参考片是 LUV2 字幕，用户不要）

参考片实测结构（analyze_ref_shots.py / analyze_ref_shots_detail.py）
--------------------------------------------------------------------
  铺垫段   0.000- 14.333s   2 个长镜（Bateman 抽烟、坐床）
  快剪段  14.333- 27.700s   15 镜，间隔 900/867ms（= 1 拍 ± 帧量化）
  结尾镜  27.700- 30.000s   2.300s  LUV2 字幕（黑白底）

  逐镜黑白（实测饱和度判定）:
    镜 0-1  混合（实拍位）   镜 4-5  黑白    镜 10 黑白    镜 13 黑白
    其余    彩色

关键实现决定
------------
  1. **拍网格用参考片自己的 BPM（67.12），不是 128**。
     参考片快剪镜 900ms 与 893.9ms 只差 6ms（< 半帧），
     而 900ms 相对 128BPM 的 468.75ms 是 1.92 拍 —— 非整数。
     详见 decide_ref_grid.py / xref_ref_grid.py 的四套交叉验证。
  2. **铺垫段 9 拍**：用户要求"缩短到 17 拍"是按旧 BGM(128BPM) 的心智说的，
     9 拍 ≈ 8.04s 才是新 BPM 下的等价时长，且正好匹配现有素材 8.10s。
  3. **黑白用 hue=s=0 而不是 format=gray** —— 后者会改像素格式，
     在 concat 边界引发 `pix fmt mismatch`；前者保持 yuv420p 只去色。
  4. **每镜末尾显式 setsar=1** —— 分屏/镜像类滤镜会改 SAR，
     concat 会报 SAR 不匹配（edit_slop.py 里踩过这个坑）。

用法：
    python edit_ref.py --footage 实拍.mp4 --material 参考片.mp4
    python edit_ref.py --footage 实拍.mp4 --material 参考片.mp4 --dry-run
    python edit_ref.py --footage 实拍.mp4 --material 参考片.mp4 --aspect square
"""
import argparse
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))

import slop_templates as st        # noqa: E402

# 成片命名 —— 与 slop_pipeline.py / edit_slop.py 用的是**同一份**规则
import film_name as fn             # noqa: E402

FFMPEG = st.FFMPEG
OUT_DIR = os.path.join(BASE, "output")
BGM_DIR = os.path.join(BASE, "bgm")

# ---------------------------------------------------------------- 参考片实测常量
# 全部来自 analyze_ref*.py / decide_ref_grid.py 的实测，不是估的：
#   BPM 67.12（谱通量自相关；圆统计残差最小 891ms=67.34BPM，取两者中值口径）
#   相位 0.1011s（圆统计解析法，残差 std 14.9ms）
REF_BPM = 67.12
REF_PERIOD = 60.0 / REF_BPM          # 0.89392s
REF_PHASE = 0.1011

# 快剪镜被设计成正好 1 拍 —— 这是复刻的核心事实
FAST_SHOT_BEATS = 1

# 铺垫段：9 拍（用户确认；参考片原为 14.33s 的 2 个长镜）
N_INTRO = 9

# 快剪段镜数：参考片实测 15 镜
N_FAST = 15

# 结尾：淡出到黑（参考片是 2.3s 的 LUV2 字幕，用户不要字幕）
OUTRO_BEATS = 2                      # 2 拍 ≈ 1.79s
# ⚠ 从 3 拍减到 2 拍，是为了让 **BGM 长度够用**（见 BGM_DROP 的说明）：
#   Drop 之后 BGM 只剩 30.037 - 14.35 = 15.687s，
#   而"快剪 15 拍 + 结尾 3 拍 = 18 拍 = 16.09s" 会超 0.4s 被截断。
#   改成 2 拍后 = 17 拍 = 15.20s，留 0.5s 余量 ✓
#   （参考片也是这么卡的：快剪 13.367s + 结尾 2.3s = 15.667s ≈ 15.687s）

# ---------------------------------------------------------------- BGM 的 Drop
# ⚠⚠ **画面的快剪起点必须对齐 BGM 的 Drop，否则"卡点"是假的。**
#
#   踩过的坑：复刻版第一版把铺垫压到 9 拍（8.045s）就切快剪，
#   但 BGM 仍从 0s 开始播 —— 而 BGM 的 Drop 在 **14.35s**。
#   结果：画面 8.045s 已经切给素材了，音乐还在铺垫段，
#   整整错位 6.3 秒。用户一眼就听出来了。
#
#   根因：参考片本来就是"铺垫 14.333s → 快剪"，它的画面切换点
#   **天然对齐**在自己的 Drop（14.35s）上 —— 实测两者只差 17ms（半帧）。
#   我们缩短了铺垫段，却没有同步移动 BGM 起点，于是错位。
#
#   实测 Drop（10ms 粒度能量包络，铺垫段中位 0.0874）：
#       14.340s  0.0929
#       14.360s  0.2419   <<< 突增 2.6 倍
#       14.380s  0.3709
#
#   修法：BGM 起点 = BGM_DROP - 视频快剪起点，
#         这样"视频播到快剪起点"时，BGM 正好播到 Drop。
BGM_DROP = 14.35

# 参考片逐镜黑白（0-based 镜号），来自 analyze_ref_shots_detail.py 的实测：
#   镜 4,5 黑白；镜 10 黑白；镜 13 黑白
#   ⚠ 参考片的"黑白"其实是原素材自带的胶片色调，不是后期滤镜。
#     用户要求"照搬位置"，所以这里在这些镜号上加黑白滤镜。
REF_BW_SHOTS = {4, 5, 10, 13}


def beat_time(n):
    """BGM 时间轴上的第 n 拍（复刻用参考片自己的相位）。"""
    return REF_PHASE + n * REF_PERIOD


# ---------------------------------------------------------------- 画布归一化
def normalize_chain(idx, tag, canvas, fps=30):
    """把第 idx 路输入归一化到统一画布（模糊背景填充）。

    ⚠ 只做一次，后面的 trim 都从它的输出取 —— 性能关键。
      若每片段单独做模糊填充，30 个片段要跑 30 次
      split+scale+gblur+overlay，耗时翻十几倍。
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


def build_timeline(f_dur, m_dur):
    """生成分镜列表。

    每项 (源索引, 素材入点, 输出时长, 效果, 是否黑白)。

    结构（完全按参考片）：
      1) 铺垫段  N_INTRO 拍，实拍，原速
      2) 快剪段  N_FAST 镜 × 1 拍，实拍 ↔ 素材交替
      3) 结尾    1 镜实拍，OUTRO_BEATS 拍，用于淡出

    素材消耗：每镜 1 拍输出 = 1 拍素材消耗（原速，不复变速）。
    实拍侧需要 N_INTRO + ceil(N_FAST/2) + 1 拍 ≈ 9 + 8 + 1 = 18 拍 ≈ 16.1s
    """
    shots = []
    t_in = 0.0

    # ---- 1) 铺垫段：实拍，9 拍 ----
    for _ in range(N_INTRO):
        shots.append((0, t_in, REF_PERIOD, "plain", False))
        t_in += REF_PERIOD

    # ---- 2) 快剪段：15 镜 × 1 拍，实拍 ↔ 素材交替 ----
    # ⚠ 从实拍开始还是从素材开始？参考片镜 0 是彩色办公室镜（素材），
    #   镜 1 也是素材，所以参考片**不是严格交替**。
    #   但用户的需求是"实拍 ↔ 素材交替"，所以这里按交替走，
    #   且从素材开始 —— 上一镜是实拍，换源才形成交替。
    m_in = 0.0
    src = 1
    for i in range(N_FAST):
        bw = (i in REF_BW_SHOTS)          # 黑白按参考片镜号照搬
        if src == 0:
            in_pt = t_in
            t_in += REF_PERIOD
        else:
            in_pt = m_in
            m_in += REF_PERIOD
        shots.append((src, in_pt, REF_PERIOD, "plain", bw))
        src = 1 - src

    # ---- 3) 结尾：实拍，OUTRO_BEATS 拍（用于淡出到黑）----
    shots.append((0, t_in, REF_PERIOD * OUTRO_BEATS, "plain", False))
    t_in += REF_PERIOD * OUTRO_BEATS

    return shots, t_in, m_in


def shot_chain(bw):
    """该镜的画面处理。黑白用 hue=s=0（不能用 format=gray，见文件头 3）。"""
    return "hue=s=0" if bw else None


# ---------------------------------------------------------------- 内容感知取段
# 画面上"几乎看不出来"的动作，见 content_aware.WEAK_ACTIONS ——
# 那是两个模板共用的策略，已随助手函数一起抽到 content_aware.py。


# ---------------------------------------------------------------- 内容感知取段
# 助手函数已抽到 content_aware.py —— edit_slop.py 用的是**同一套**。
# 两处各写一份必然会漂（项目已经吃过这个亏），所以这里只做导入。
# 共用的还有 WEAK_ACTIONS（哪些动作排到最后用）与 load_clipmap。
from content_aware import (INTRO_STEP, WEAK_ACTIONS, _window_of,
                           _cands_in_spans, _spans_for_intro, _free_gaps,
                           _clamp_free, _richness, select_anchors,
                           pick_intro_offset, max_distinct_actions,
                           fallback_in_pt, load_clipmap)     # noqa: F401,E402


def select_shots(segs, peaks, intro_off, intro_len, src_dur, scope, peak_pos):
    """把实拍位的入点分配到内容池里 —— 内容感知取段的全部逻辑。

    三段分开处理：

    · **铺垫段**：连续长镜，整段平移到 [intro_off, intro_off+intro_len]。
      参考片的铺垫段就是一个稳的长镜，逐镜吸附会把它变成 9 个跳接，
      观感全毁 —— 那一段要的是稳，不是卡动作。
      （scope="all" 时不这样，逐镜吸附，显式选择才会走到）

    · **快剪段实拍位**：在铺垫段**之外**的空档里挑若干峰值，选法见
      `select_anchors`。候选可行域是"窗口整段落在空档里"反解出来的，
      所以放置 `p = 峰值 − peak_pos×镜长` 必然精确，**不需要任何钳位**。

      ⚠ 旧实现在这里踩过一个坑：先按强度全局挑最强的 7 个、再按时间摆。
        结果第一个镜吸到了铺垫段里的「微笑」（它强度最高），
        但那个时刻早被铺垫段占用了，钳位后峰值根本没进镜内，
        报告里出现 **-10244ms** 这种荒唐的落位偏差，末镜还越了界。
        **顺序必须是"先筛可行域、再谈强度"**，反过来一定出错。

    · **结尾**：优先紧跟最后一个锚点；那里被占住时退到别的空档。
      结尾本来就是"取一段淡出到黑"，从哪一段取不影响观感。

    返回 (新分镜表, 逐镜报告, 统计信息)。
    """
    intro_shots = [i for i in range(N_INTRO) if segs[i][0] == 0]
    fast_shots = [i for i in range(N_INTRO, N_INTRO + N_FAST) if segs[i][0] == 0]
    later_shots = [i for i, s in enumerate(segs)
                   if s[0] == 0 and i >= N_INTRO + N_FAST]

    if scope == "none" or not peaks:
        return segs, [], {"od": 0.0, "max_distinct": 0, "avail_distinct": 0,
                          "reserve": 0.0, "spans": []}

    out = list(segs)
    report = []
    occupied = []                                   # 已被占用的源区间

    if scope != "all":
        # 铺垫段整体平移（它是一整条连续长镜，不能逐镜吸附）
        for i in intro_shots:
            s_src, s_in, s_od, s_kind, s_bw = segs[i]
            out[i] = (s_src, s_in + intro_off, s_od, s_kind, s_bw)
        occupied.append((intro_off, intro_off + intro_len))
        targets = fast_shots
        prev_end = intro_off + intro_len
    else:
        targets = intro_shots + fast_shots
        prev_end = 0.0

    od = max((segs[i][2] for i in targets), default=0.0)
    reserve = sum(segs[i][2] for i in later_shots)

    # ---- 候选空档 = 时间轴减去铺垫段 ----
    spans = (_spans_for_intro(intro_off, intro_len, src_dur, od, reserve)
             if scope != "all" else [(0.0, src_dur)])
    cand = _cands_in_spans(peaks, spans, od, peak_pos)
    chosen = select_anchors(cand, len(targets), od, WEAK_ACTIONS)

    # DP 用来回答"上限到底是多少"，有成本，所以只在**真的出现复用**时才跑。
    # 平时不跑，免得每次渲染都付这个开销。
    n_distinct = len({p["action"] for p in chosen})
    if n_distinct < len(chosen):
        max_dist, avail_dist = max_distinct_actions(peaks, od, peak_pos, spans)
    else:
        max_dist, avail_dist = n_distinct, n_distinct

    # ---- 放置 ----
    for k, i in enumerate(targets):
        src, base, od_i, kind, bw = out[i]
        if k < len(chosen):
            tgt = chosen[k]["t"]
            p = tgt - peak_pos * od_i
            report.append(dict(shot=i, planned=base, in_pt=p, peak=tgt,
                               action=chosen[k]["action"], offset=0.0,
                               covered=True, note=""))
        else:
            # ⚠ 兜底要**避开已用窗口**：空档只剩一个时，直接钳位会把所有
            #   剩余镜头叠到同一帧（成片里就是同一动作连续回放）。
            #   详见 content_aware.fallback_in_pt 的说明。
            p = fallback_in_pt(base, od_i, spans, src_dur, occupied)
            report.append(dict(shot=i, planned=base, in_pt=p, peak=None,
                               action=None, offset=None, covered=False,
                               note="空档内峰值不够，已避开用过的位置顺序取用"))
        out[i] = (src, p, od_i, kind, bw)
        occupied.append((p, p + od_i))
        prev_end = p + od_i

    # ---- 结尾段 ----
    # ⚠ 解包变量不能叫 `od` —— 那会覆盖上面算好的"镜长"（结尾段是 2 拍 = 1.788s），
    #   于是报告里写出"不重叠的 1788ms 镜头"这种自相矛盾的数字，
    #   而 DP 算出的上限却是按 894ms 撑出来的。
    for i in later_shots:
        src, base, od_l, kind, bw = out[i]
        gaps = _free_gaps(src_dur, occupied, od_l)
        if gaps:
            # 优先紧跟上一个锚点（保住"结尾接在最后"的观感），否则退到最靠后的空档
            p = next((a for a, b in gaps if abs(a - prev_end) < 1e-3), gaps[-1][0])
        else:
            p = fallback_in_pt(base, od_l, spans, src_dur, occupied)
            print(f"  ⚠ 结尾段找不到 {od_l:.3f}s 的连续空档，已退到 "
                  f"{p:.3f}s（可能与其它段重叠，请检查素材长度）")
        out[i] = (src, p, od_l, kind, bw)
        occupied.append((p, p + od_l))
        prev_end = p + od_l

    return out, report, {"od": od, "max_distinct": max_dist,
                         "avail_distinct": avail_dist,
                         "reserve": reserve, "spans": spans}

def build_filter(segs, canvas, footage_dur, material_dur, cfg):
    """拼出完整 filter_complex。"""
    parts = [normalize_chain(0, "fw", canvas), normalize_chain(1, "fm", canvas)]

    # ---- 显式 split（必需，否则 concat 报尺寸不匹配）----
    # ⚠⚠ 归一化链的输出被多个 trim 消费。直觉上让 ffmpeg 自动插 split 即可，
    #   但实测报 `Input link in0:v0 parameters (size 1280x720, SAR 1:1)
    #   do not match the corresponding output link (1080x1920, SAR 1:1)`。
    #   原因：overlay 是**多输入滤镜**，输出尺寸由主输入推导（非固定参数），
    #   被自动 split 时尺寸推导链断掉，协商失败退回源尺寸。
    #   已排除：标签复用本身 / overlay 后再 scale / format 钉死 —— 都照样失败。
    #   唯一解：**自己写 split**。
    n_fw = sum(1 for s in segs if s[0] == 0)
    n_fm = sum(1 for s in segs if s[0] == 1)
    if n_fw:
        labs = [f"fwa{i}" for i in range(n_fw)]
        parts.append(f"[fw]split={n_fw}" + "".join(f"[{l}]" for l in labs))
    if n_fm:
        labs = [f"fma{i}" for i in range(n_fm)]
        parts.append(f"[fm]split={n_fm}" + "".join(f"[{l}]" for l in labs))

    fps = float(st.SPEC["fps"])

    # ---- 输出起点吸附到理想网格的最近帧 ----
    # ⚠⚠ 拍周期 0.89392s 在 30fps 下 = 26.8176 帧（不是整数）。
    #   若每段按理想时长切，帧量化误差**逐段累积**：
    #   30 段会漂 0.5s，后半段画面完全偏离拍点，闪白用 mod(t,period) 判定
    #   会有的拍点整帧错过。
    #   解法：每段的输出起点都吸附到"理想网格的最近帧"（不是"上一段结束的
    #   最近帧"）—— 后者会累积，前者每段重新对齐，误差不超过半帧。
    ideal, acc = [0.0], 0.0
    for s in segs:
        acc += s[2]
        ideal.append(acc)
    snapped = [round(v * fps) / fps for v in ideal]

    counters = {0: 0, 1: 0}
    labels = []
    for i, (src, t_in, out_dur, kind, bw) in enumerate(segs):
        k = counters[src]
        counters[src] += 1
        tag = f"fwa{k}" if src == 0 else f"fma{k}"
        avail = footage_dur if src == 0 else material_dur

        src_len = out_dur                       # 原速：消耗 = 输出
        out_len = snapped[i + 1] - snapped[i]
        if out_len <= 1e-6:
            out_len = out_dur

        # 素材不够时循环取用
        t0 = t_in % max(avail - src_len, 0.05) if avail > src_len else 0.0
        lab = f"s{i}"

        fx = shot_chain(bw)

        # ⚠⚠ 这里曾经出一个**静默错误**，值得完整记下（四轮才定位到真因）：
        #
        #   从 edit_slop.py 抄来的写法是「trim 多给 1 帧余量」：
        #       trim=duration=(src_len + 1帧), setpts=(PTS-STARTPTS)/real_speed
        #   理由是"若 src_len 非帧长整数倍，实际取到的帧会少 1 帧，
        #   多给 1 帧后会被 fps 正常丢掉"。
        #
        #   **这个前提在被裁掉变速的脚本里不成立**。实测（真实素材，1080x1920）：
        #       P = 0.89392s = 26.818 帧 @30fps
        #       trim=duration=(P + 1帧)=0.92725 → **输出 28 帧** = ceil(27.818)
        #       trim=duration=P           → **输出 27 帧** ✓
        #       trim=start_frame=0:end_frame=27 → 输出 27 帧 ✓
        #
        #   即 `trim` 的 duration 给多少帧的料，`fps=30` 就圆整出多少帧 ——
        #   那 1 帧余量**不会被丢弃，而是完整进入输出**。
        #   在 edit_slop.py 里它之所以无害，是因为那里的 setpts 有 0.25x/1.35x
        #   这类**强压缩**，压缩后总长明显小于目标，多出的帧确实被丢掉了。
        #   本脚本是原速复刻（无变速），多给的帧原封不动留下来。
        #
        #   症状：硬切点相对理论边界**每镜固定 +1 帧（33.3ms）**，
        #   到快剪段第 4 镜偏移已达 +12 帧（见 verify_ref.py 的对齐表），
        #   判定"3/25 命中"。因为偏差是**稳定匀速**而非随机漂移，
        #   极易误诊成"帧量化漂移"，而吸附逻辑从头到尾都是对的。
        #
        #   ⚠ 教训：**跨脚本抄滤镜链时必须重新验证，不能假设前提成立**。
        #     "多给 1 帧会被丢掉"是 edit_slop.py 的**实现巧合**，
        #     不是 ffmpeg 的通用行为。
        #
        #   另一个已证伪的中间结论（留作记录）：曾以为病根是
        #     `setpts` 改时间戳后接 `fps` 会膨胀 1~2 帧，于是改成
        #     `setpts=PTS-STARTPTS`（只归零不压缩）—— 无效，帧数没变。
        #     原因是本地实验用了 `testsrc` 且 trim 不加余量，
        #     **对照组与被测组的变量不止一个**，属于实验设计错误。
        #     正确做法：单一变量对照，且必须用真实素材。
        #
        #   修法：**trim 精确按输出帧数取料**，不给余量。
        #   用 start_frame/end_frame 按帧取最精确（不受浮点误差影响）。
        n_out_frames = int(round(out_len * fps))
        k0 = int(round(t0 * fps))
        chain = f"trim=start_frame={k0}:end_frame={k0 + n_out_frames}"
        chain += ",setpts=PTS-STARTPTS"
        if fx:
            chain += f",{fx}"
        # ⚠ 每镜必须 setsar=1。hstack（分屏镜像）会改采样宽高比：
        #   两路 1:1 并排后变 SAR 2:1，concat 报 SAR 不匹配。
        #   只补 scale 不够（尺寸对了 SAR 还错），必须显式钉 SAR。
        chain += f",setsar=1,fps={fps:g}"

        parts.append(f"[{tag}]{chain}[{lab}]")
        labels.append(lab)

    # ---- 拼接（硬切：concat 本身就是直接切）----
    cat = "".join(f"[{l}]" for l in labels)
    parts.append(f"{cat}concat=n={len(labels)}:v=1:a=0[cat]")

    # ---- 全局特效链 ----
    chain = []

    # 调色：参考片整体偏冷、略暗、低饱和。轻量方案，不抢戏。
    if cfg["grade"]:
        chain.append("eq=saturation=0.92:contrast=1.06:brightness=-0.02,"
                     "colorbalance=rs=0.03:bs=-0.04")

    # 色差偏移（RGB 分离）：水平偏移，淡淡的彩边
    if cfg["rgb"] > 0:
        chain.append(st.rgb_split(shift=cfg["rgb"]))

    # VHS 胶片颗粒
    if cfg["grain"] > 0:
        chain.append(f"noise=alls={cfg['grain']}:allf=t+u")

    # 结尾淡出到黑（用户要求；参考片是 LUV2 字幕）
    # ⚠ fade 用视频侧，配合音频侧的 afade
    if cfg["fade"] > 0:
        total = snapped[-1]
        chain.append(f"fade=t=out:st={max(0.0, total - cfg['fade']):.4f}"
                     f":d={cfg['fade']:g}")

    chain.append("format=yuv420p")
    parts.append("[cat]" + ",".join(chain) + "[vout]")
    return ";".join(parts)


def build_audio(bgm, dur, cfg):
    """BGM：从 `BGM_DROP - 快剪起点` 处开始截取，**使 Drop 精确落在快剪起点**。

    ⚠⚠ 这里**不能从 0s 开始**。第一版就是从 0s 起，导致画面 8.045s 切快剪、
    音乐 14.35s 才 Drop，错位 6.3 秒（用户一听就发现了）。详见 BGM_DROP 说明。
    """
    fade = cfg["fade"]
    t0 = max(0.0, BGM_DROP - N_INTRO * REF_PERIOD)
    return (f"atrim=start={t0:.4f}:duration={dur:.4f},asetpts=N/SR/TB,"
            f"afade=t=out:st={max(0.0, dur-fade):.4f}:d={fade:g},"
            f"volume={cfg['volume']:g}")


def probe(path):
    """读 元数据。本机没有 ffprobe，只能解析 ffmpeg -i 的 stderr。
    ⚠ ffmpeg -i 的退出码总是 1，判定只能看内容。"""
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    txt = r.stderr or ""
    dur, size, has_a = 0.0, (0, 0), False
    import re
    for line in txt.splitlines():
        if "Duration" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            dur = int(h) * 3600 + int(m) * 60 + float(s)
        if "Stream #" in line and "Video:" in line:
            m2 = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m2:
                size = (int(m2.group(1)), int(m2.group(2)))
        if "Audio:" in line:
            has_a = True
    return dur, size, has_a


def main():
    ap = argparse.ArgumentParser(description="参考片复刻卡点器")
    ap.add_argument("--footage",
                    help="实拍素材 mp4（给了 --clips 时可省略，自动改用内容池的原片）")
    ap.add_argument("--material", required=True,
                    help="交错素材 mp4（如 2022年的剪辑 belike.mp4）")
    ap.add_argument("--clips",
                    help="clip_extract.py 产出的 reel_map.json —— "
                         "接入内容感知取段（顺序游标 → 内容池 + 动作峰值）")
    ap.add_argument("--snap-scope", default="fast",
                    choices=["fast", "all", "none"],
                    help="内容感知的作用范围：fast=铺垫段整体滑窗 + 只快剪段逐镜"
                         "吸附峰值（默认，保住铺垫段的连续长镜观感）；"
                         "all=连铺垫段也逐镜吸附（会变成 9 个跳接）；none=关")
    ap.add_argument("--intro-pick", default="auto", choices=["auto", "start"],
                    help="铺垫段连续窗口的选法：auto=滑窗选动作最密集的那 8 秒"
                         "（默认）；start=固定从 0 秒开始")
    ap.add_argument("--peak-pos", type=float, default=0.35,
                    help="动作峰值落在镜内的相对位置（0=镜首，1=镜尾，默认 0.35）")
    ap.add_argument("--bgm", help="BGM，缺省用 bgm/ref_bgm.m4a")
    ap.add_argument("--out", help="输出路径")
    ap.add_argument("--aspect", default="square",
                    choices=["vertical", "square"],
                    help="square=1080x1080（参考片是 360x360 方）；"
                         "vertical=1080x1920")
    ap.add_argument("--grain", type=int, default=14, help="VHS 颗粒强度")
    ap.add_argument("--rgb", type=int, default=6, help="色差水平偏移 px（0=关）")
    ap.add_argument("--fade", type=float, default=1.5, help="结尾淡出秒数")
    ap.add_argument("--volume", type=float, default=0.9, help="BGM 音量")
    ap.add_argument("--no-grade", action="store_true", help="不调色")
    ap.add_argument("--no-bw", action="store_true",
                    help="关闭黑白镜（不加 hue 去色）")
    ap.add_argument("--face-crop", type=float, nargs="?", const=3.0,
                    metavar="ZOOM",
                    help="先把实拍素材做人脸特写再剪辑（默认放大 3.0 倍）。"
                         "原始录像里脸只占约 16%% 画面高，特写后能占 40%%+。"
                         "⚠ 素材必须**不含烧死的 HUD**（人脸框/文字），"
                         "后期擦不掉 —— 请用 record_auto.py --no-grade 录")
    ap.add_argument("--face-safe-y", metavar="TOP:BOTTOM",
                    help="人脸特写的安全带（配合 --face-crop）。"
                         "手头只有带 HUD 的旧素材时用，例：702:1161")
    ap.add_argument("--face-lock", action="store_true",
                    help="人脸特写开启**锁头运镜**（配合 --face-crop）："
                         "头部固定在画面同一位置，背景绕着头转")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印方案与滤镜链，不渲染")
    ap.add_argument("--dump-plan",
                    help="把分镜表 + 内容感知报告写成 JSON —— "
                         "供独立验收脚本核对（不要让验收脚本自己重算一遍方案，"
                         "那样验的是同一份逻辑、等于没验）")
    a = ap.parse_args()

    clipmap = None
    if a.clips:
        if not os.path.exists(a.clips):
            print(f"映射表不存在: {a.clips}")
            return 1
        clipmap = load_clipmap(a.clips)
        if clipmap is None:
            return 1
        # 内容感知模式下实拍源 = **内容池的原片**（不是素材带）。
        # 若两个都给了，以内容池为准并说明，免得"给了一个却被静默忽略"。
        if a.footage and os.path.abspath(a.footage) != os.path.abspath(clipmap["source"]):
            print(f"提示：已用 --clips，实拍源改用内容池原片，"
                  f"忽略 --footage {os.path.basename(a.footage)}")
        a.footage = clipmap["source"]

    if not a.footage:
        print("需要 --footage 指定实拍素材，或用 --clips 指定内容池映射表")
        return 1
    if not os.path.exists(a.footage):
        print(f"实拍素材不存在: {a.footage}")
        return 1
    if not os.path.exists(a.material):
        print(f"交错素材不存在: {a.material}")
        return 1

    # ---- 可选：先把实拍素材做人脸特写 ----
    # 做法与 edit_slop.py 一致（subprocess 调 face_crop.py 做**前置**，
    # 产出一个中间文件再交给剪辑），但这里**多一道时序校验**，原因是本脚本带内容感知：
    #
    # ⚠⚠ 动作峰值坐标全部是**原片时间轴**上的（clipmap 的 t 来自 action_detect 跑的原片）。
    #   特写片理论上只是换取景、时序不变，可一旦它改了帧数/时长/fps，
    #   所有峰值坐标会**整体错位**，而画面看上去完全正常 ——
    #   典型的"不报错但结果错"。所以特写做完必须比对时长，超一帧就直接退出，
    #   绝不允许带着错位的坐标往下渲染。
    if a.face_crop:
        fc_script = os.path.join(BASE, "face_crop.py")
        if not os.path.exists(fc_script):
            print("找不到 face_crop.py，跳过人脸特写")
        else:
            src_dur = probe(a.footage)[0]
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
            if r.returncode != 0 or not os.path.exists(face_out):
                print("  人脸特写失败，改用原素材")
                for line in (r.stderr or "").strip().splitlines()[-3:]:
                    print("   ", line[:150])
            else:
                new_dur = probe(face_out)[0]
                drift = abs(new_dur - src_dur)
                if clipmap and drift > 1.0 / 30 + 1e-6:
                    print(f"  ✗ 特写片 {new_dur:.3f}s 与原片 {src_dur:.3f}s "
                          f"相差 {drift*1000:.0f}ms（> 1 帧）")
                    print(f"    内容感知的峰值坐标全部基于原片时间轴，"
                          f"带错位渲染等于白做 —— 已终止，不出片。")
                    return 1
                print(f"  特写素材 -> {face_out}")
                print(f"    时长 {new_dur:.3f}s / 原片 {src_dur:.3f}s "
                      f"（偏差 {drift*1000:.0f}ms ✓ 时序一致）")
                a.footage = face_out
                if clipmap:
                    # ⚠ 验收脚本（verify_contentaware.py）会拿 plan["source"]
                    #   去取**源帧**做内容匹配。必须指向特写片，
                    #   否则比的是两种取景，相关系数必然崩，会误报成"选段错了"。
                    clipmap["source"] = os.path.abspath(face_out)

    bgm = a.bgm or os.path.join(BGM_DIR, "ref_bgm.m4a")
    if not os.path.exists(bgm):
        print(f"BGM 不存在: {bgm}")
        print("  提示：跑 analyze_ref_bgm2.py 前需先提取参考片音轨，或指定 --bgm")
        return 1

    canvas = (1080, 1080) if a.aspect == "square" else (1080, 1920)
    cfg = {"grain": a.grain, "rgb": a.rgb, "fade": a.fade,
           "volume": a.volume, "grade": not a.no_grade}

    f_dur, f_sz, f_a = probe(a.footage)
    m_dur, m_sz, m_a = probe(a.material)

    print("=" * 68)
    print("参考片复刻卡点器（edit_ref.py）")
    print("=" * 68)
    print(f"实拍素材 {os.path.basename(a.footage)}  {f_sz[0]}x{f_sz[1]}  "
          f"{f_dur:.2f}s{'  含音轨(会被BGM替换)' if f_a else ''}")
    print(f"         路径 {os.path.abspath(a.footage)}")
    print(f"交错素材 {os.path.basename(a.material)}  {m_sz[0]}x{m_sz[1]}  "
          f"{m_dur:.2f}s{'  含音轨(会被BGM替换)' if m_a else ''}")
    print(f"         路径 {os.path.abspath(a.material)}")
    print(f"BGM      {os.path.basename(bgm)}")
    if clipmap:
        print(f"内容感知 开：{len(clipmap['peaks'])} 个动作峰值 / "
              f"{len(clipmap['regions'])} 段有动作的区间"
              f"（原片 {os.path.basename(clipmap['source'])}）"
              f"   范围 {a.snap_scope}   峰值落位 {a.peak_pos:.0%}")
    print(f"\n拍网格   {REF_BPM:.2f} BPM   拍周期 {REF_PERIOD*1000:.2f}ms   "
          f"相位 {REF_PHASE:.4f}s")

    segs, used_f, used_m = build_timeline(f_dur, m_dur)

    # ---- 内容感知：选铺垫段位置（按"给快剪段留下什么"）+ 快剪位取动作锚点 ----
    snap_report, snap_info = [], {}
    intro_off, intro_sel, intro_info = 0.0, [], {}
    if clipmap:
        intro_len = N_INTRO * REF_PERIOD
        n_shot_fw = sum(1 for i in range(N_INTRO, N_INTRO + N_FAST)
                        if segs[i][0] == 0)
        reserve = sum(s[2] for i, s in enumerate(segs)
                      if s[0] == 0 and i >= N_INTRO + N_FAST)
        if a.intro_pick == "auto":
            # 搜索整条素材，不用 first_run_end —— 那条边界会被几十毫秒的空隙砍掉，
            # 详见 pick_intro_offset 的说明。铺垫段"不冷清"由中位数门槛保证。
            intro_off, intro_sel, intro_info = pick_intro_offset(
                clipmap["peaks"], intro_len, max(0.0, f_dur - intro_len),
                REF_PERIOD, f_dur, a.peak_pos, n_shot_fw, reserve,
                WEAK_ACTIONS)
        else:
            intro_sel = [p for p in clipmap["peaks"]
                         if p["t"] <= intro_len + 1e-6]
        segs, snap_report, snap_info = select_shots(
            segs, clipmap["peaks"], intro_off, intro_len, f_dur,
            a.snap_scope, a.peak_pos)
        # 吸附后"用了多少实拍"不再是顺序游标，而是所有实拍段的**最远结束点**
        used_f = max((s[1] + s[2] for s in segs if s[0] == 0), default=0.0)
    total = sum(s[2] for s in segs)

    intro_end = N_INTRO * REF_PERIOD
    fast_end = intro_end + N_FAST * REF_PERIOD

    print(f"\n时间轴（复刻参考片结构，共 {total:.3f}s / "
          f"{total/REF_PERIOD:.1f} 拍）")
    print(f"  铺垫段   0.000-{intro_end:.3f}s  实拍  {N_INTRO} 拍 × "
          f"{REF_PERIOD*1000:.0f}ms")
    print(f"  快剪段   {intro_end:.3f}-{fast_end:.3f}s  "
          f"{N_FAST} 镜 × 1 拍（实拍 ↔ 素材交替）")
    print(f"  结尾     {fast_end:.3f}-{total:.3f}s  淡出到黑 "
          f"({OUTRO_BEATS} 拍)")

    n_bw = sum(1 for s in segs if s[4])
    print(f"\n黑白镜（照搬参考片镜号 {sorted(REF_BW_SHOTS)}）: 共 {n_bw} 镜")

    # ---- 内容感知取段报告 ----
    # 不复用"命中率"这种词：这里的"偏差"就是**峰值离目标落位的距离**，
    # 它衡量的是"切点有没有压住动作"，和卡点命中率是两件事。
    if clipmap and a.snap_scope != "all":
        from collections import Counter
        intro_len = N_INTRO * REF_PERIOD
        cnt = Counter(p["action"] for p in intro_sel)
        where = ("按「给快剪段留下什么」择优" if a.intro_pick == "auto"
                 else "固定从 0 开始")
        print(f"\n铺垫段选窗（{where}）: {intro_off:.3f} → "
              f"{intro_off + intro_len:.3f}s，窗口内含 {len(intro_sel)} 个动作峰值"
              f"（{'、'.join(f'{k}×{v}' for k, v in cnt.most_common()) or '无'}）")
        if a.intro_pick == "auto" and intro_info:
            print(f"  判据是它**留下**什么，不是它**装进**什么："
                  f"该位置能给快剪段留下 {intro_info['n_chosen']} 个锚点 / "
                  f"{intro_info['n_distinct']} 种不同动作"
                  f"（锚点总强度 {intro_info['strength']:.3f}）")
            print(f"  铺垫段自身丰富度 {intro_info['intro_score']:.3f}"
                  f"（**末位判据**，只用来打破平局）—— "
                  f"**铺垫段**是连续长镜，装进去的事件只被连续播一遍，不做锚点用，"
                  f"所以它不该为了自己不无聊去占用快剪段的素材")
            print(f"  在整条素材上搜了 {intro_info['n_cand']} 个位置；"
                  f"丰富度门槛（中位）{intro_info['rich_floor']:.3f}"
                  f" → 可行 {intro_info['n_feasible']} 个"
                  f"（最高可达 {intro_info['rich_best']:.3f}）。"
                  f"门槛只为保证铺垫段不冷清，不参与择优")
    if snap_report:
        hit = [r for r in snap_report if r["covered"]]
        print(f"\n内容感知取段（实拍位 {len(snap_report)} 个 → "
              f"命中动作峰值 {len(hit)}，退回顺序取用 {len(snap_report)-len(hit)}）")
        print(f"  {'镜':>3}{'计划入点':>10}{'实际入点':>10}{'峰值':>9}"
              f"{'落位偏差':>10}  动作")
        for r in snap_report:
            off = "—" if r["offset"] is None else f"{r['offset']*1000:+.0f}ms"
            pk = "—" if r["peak"] is None else f"{r['peak']:.3f}"
            print(f"  {r['shot']:>3}{r['planned']:>10.3f}{r['in_pt']:>10.3f}"
                  f"{pk:>9}{off:>10}  {r['action'] or r['note']}")
        if hit:
            # ⚠ 这里**不要**把"落位偏差 0ms"当成一项成绩报出来。
            #   它是**按构造恒为 0** 的：入点由"先筛可行域、再反推 p = 峰值 −
            #   peak_pos×镜长"得到，只要候选筛对了，偏差就不可能不是 0。
            #   旧版把它打印成"落位中位偏差 0ms（目标是把峰值放在镜内 35% 处）"，
            #   读起来像一项量出来的验证结果 —— 其实什么都没验证。
            #   真正需要独立核对的（也真的会出错）是"渲染出来那一帧到底是不是
            #   那个动作"，那是**跨模态**的事，必须到成片画面上去量。
            print(f"  落位：{len(hit)} 个锚点的峰值均精确落在镜内 "
                  f"{a.peak_pos:.0%} 处（按构造保证，非量测）")
            print(f"    ⚠ 这是方案自身的算术性质，**不构成验证**。"
                  f"要确认「画面里真的取到了那个动作」，跑 verify_contentaware.py")
        acts = [r["action"] for r in snap_report if r["action"]]
        if acts:
            from collections import Counter
            cnt = Counter(acts)
            rep = [k for k, v in cnt.items() if v > 1]
            print(f"  动作类型 {len(set(acts))} 种 / {len(acts)} 个镜")
            if rep:
                od_ms = snap_info.get("od", REF_PERIOD) * 1000
                print(f"    同类型复用 {sum(cnt[k] - 1 for k in rep)} 次："
                      f"{'、'.join(rep)}")
                print(f"    素材的动作密度凑不满 {len(acts)} 种，这是**当前配置下的上限**：")
                print(f"      约束 = 镜头不重叠 + 峰值固定落在镜内 "
                      f"{a.peak_pos:.0%} + 结尾预留 "
                      f"{snap_info.get('reserve', 0):.2f}s")
                sp = " ∪ ".join("[%.2f, %.2f]" % (lo, hi)
                                for lo, hi in snap_info.get("spans", []))
                print(f"      可放锚点的空档 {sp} 内最多放得下 "
                      f"{snap_info.get('max_distinct', 0)} 种不同动作"
                      f"（精确 DP 算的，不是估值）")
                print(f"      想让 {len(acts)} 个镜全不重复：需要动作更密/种类更多的"
                      f"素材（或缩短铺垫段把时间让给快剪段）")
                for k in rep:
                    ts = sorted(r["peak"] for r in snap_report
                                if r["action"] == k)
                    print(f"    复用挑的是离同类上一次最远的一对："
                          f"{k} {ts[0]:.2f}s 与 {ts[-1]:.2f}s，"
                          f"相隔 {ts[-1]-ts[0]:.2f}s")
            else:
                print(f"    **每类只用一次** ✓")

    # ---- BGM 对齐校验（Drop 必须落在快剪起点）----
    # ⚠ 这一段是"卡点是否真的成立"的自检。第一版就是从 0s 起播 BGM，
    #   错位 6.3 秒却没有任何提示 —— 加了这段就能一眼看出。
    t0_bgm = max(0.0, BGM_DROP - intro_end)
    bgm_dur = probe(bgm)[0]
    print(f"\nBGM 对齐:")
    print(f"  Drop 实测 @ {BGM_DROP:.3f}s（能量突增点，参考片画面切换在 14.333s，吻合）")
    print(f"  BGM 从 {t0_bgm:.3f}s 起截取"
          f" → 视频播到快剪起点 {intro_end:.3f}s 时，"
          f"BGM 正好到达 {t0_bgm + intro_end:.3f}s = Drop ✓")
    need = t0_bgm + total
    if need > bgm_dur + 0.05:
        print(f"  ⚠ BGM 需要 {need:.3f}s / 实有 {bgm_dur:.3f}s"
              f" —— 超出 {need - bgm_dur:.2f}s，结尾会被截断！"
              f"（可减少 OUTRO_BEATS 或 N_FAST）")
    else:
        print(f"  BGM 需要 {need:.3f}s / 实有 {bgm_dur:.3f}s"
              f" ✓ 够用（余 {bgm_dur - need:.2f}s）")

    # ---- 素材消耗核对 ----
    short = []
    if used_f > f_dur + 0.05:
        short.append(f"实拍素材 需要 {used_f:.2f}s，实有 {f_dur:.2f}s"
                     f"（差 {used_f - f_dur:.2f}s）")
    if used_m > m_dur + 0.05:
        short.append(f"交错素材 需要 {used_m:.2f}s，实有 {m_dur:.2f}s"
                     f"（差 {used_m - m_dur:.2f}s）")
    print(f"\n素材消耗:")
    print(f"  实拍 {used_f:.2f}s / 实有 {f_dur:.2f}s")
    print(f"  素材 {used_m:.2f}s / 实有 {m_dur:.2f}s")
    if short:
        print("\n" + "!" * 60)
        print("⚠ 素材长度不足 —— 不足的部分会**循环重复**：")
        for s in short:
            print(f"    {s}")
        print("!" * 60)

    # ---- 成片路径：**在 dump 之前定下来**，好写进方案的 render 段 ----
    #   ⚠ 兜底名也必须能自解释（原来是 `ref_remake_20260925_221530.mp4`）。
    #     与流水线共用同一份命名规则（film_name.py），两条入口才不会不一致。
    #   ⚠ 本模板没有 `--slices` / `--no-fx`，黑白是 `--no-bw`（反向开关、
    #     默认就开），所以这几个词在复刻版里不会出现 —— 不是漏了。
    out = a.out or fn.next_versioned(
        OUT_DIR, fn.film_stem("复刻版", a.aspect,
                              face_crop=a.face_crop, face_lock=a.face_lock,
                              face_safe_y=a.face_safe_y))

    # ---- 方案导出（供独立验收核对）----
    # ⚠ 验收脚本**不应该自己重算一遍方案** —— 那就变成"同一份逻辑验自己"，
    #   逻辑错了照样通过。所以这里把方案落地成 JSON，验收脚本只负责
    #   拿它去和**真实渲染出的画面**对照。
    if a.dump_plan:
        acc, shots_out = 0.0, []
        for i, s in enumerate(segs):
            shots_out.append({"i": i, "src": s[0], "in_pt": round(s[1], 4),
                              "out_dur": round(s[2], 4), "bw": bool(s[4]),
                              "out_start": round(acc, 4)})
            acc += s[2]
        plan = {"source": os.path.abspath(a.footage),
                "material": os.path.abspath(a.material),
                "canvas": list(canvas),
                "period": REF_PERIOD,
                "content_aware": bool(clipmap),
                # 本模板有**逐镜黑白**，验收脚本据此决定要不要跑黑白项。
                # 卡点版（edit_slop）没有逐镜黑白，会写 False 让那一项自动跳过。
                "has_bw": True,
                "peak_pos": a.peak_pos,
                "intro_offset": intro_off,
                # ---- 渲染参数：让工具**不必再解析文件名**去猜模式 ----
                #   ⚠ 没有 `grade` 键：预调色是**流水线层面**的事，本脚本看不到；
                #     它由文件名里的 `调色` 词体现，实际用的源看上面的 `source`。
                "render": {
                    "template": "复刻版",
                    "aspect": a.aspect,
                    "face_crop": a.face_crop,
                    "face_lock": bool(a.face_lock),
                    "face_safe_y": a.face_safe_y,
                    "snap_scope": a.snap_scope,
                    "no_bw": bool(a.no_bw),
                    # ⚠ 必须用 `a.clips`（原始**路径**），不能写 `clipmap`
                    #   —— 那是 load_clipmap 之后的**字典**，abspath 会 TypeError。
                    "clips": os.path.abspath(a.clips) if a.clips else None,
                    "out": os.path.basename(out),
                },
                "snap": snap_report,
                "shots": shots_out}
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_plan)),
                    exist_ok=True)
        with open(a.dump_plan, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print(f"\n方案 -> {a.dump_plan}")

    fc = build_filter(segs, canvas, f_dur, m_dur, cfg)
    af = build_audio(bgm, total, cfg)

    if a.dry_run:
        print(f"\n[filter_complex] {len(fc)} 字符")
        print(fc[:1500] + ("..." if len(fc) > 1500 else ""))
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
            print("   ", line[:170])
        return 1

    od, osz, oa = probe(out)
    print(f"完成 -> {out}")
    print(f"  {canvas[0]}x{canvas[1]}  {od:.2f}s  "
          f"{os.path.getsize(out)/1024/1024:.1f} MB  耗时 {el:.1f}s")
    if abs(od - total) > 0.4:
        print(f"  ⚠ 时长偏差: 期望 {total:.2f}s 实得 {od:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
