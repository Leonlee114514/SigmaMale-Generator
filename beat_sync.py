"""卡点剪辑：把 BGM 节拍网格映射成剪辑锚点，生成变速/闪烁表达式。

核心矛盾：
    **变速会改变画面时长，而 BGM 是固定的。**
    所以不能"先变速再看卡点准不准"，必须反过来：
    以 BGM 的节拍时间轴为基准，去决定画面在第几秒该快、第几秒该慢。

本模块提供两种卡点模式：

  A. 「视频适配音频」（推荐，真正卡点的做法）
     画面总时长跟着音乐走。先算出目标时长（比如正好 8 个强拍 = 15 秒），
     再让变速把素材压缩/拉伸到该时长，且慢放恰好落在强拍上。

  B. 「音频适配视频」
     画面时长不动，BGM 用 atrim 截取 + 淡出，或 aloop 循环补足。
     做法简单但卡点精度差，只适合素材时长已经很接近的情况。

产出的是 ffmpeg setpts 表达式 + 闪烁/抖动的时间点列表。
"""
import json
import os
import sys

import numpy as np

# 复用模板库里的表达式转义入口（减号 + 逗号两个坑都在那里统一处理）。
# 这样"setpts 怎么写才不被 ffmpeg 8.1.1 拒绝"的知识只有一处，不会漂移。
#
# 名字兼容：本项目里叫 slop_templates，抽成通用 skill 后叫 video_templates。
# 两个都找不到时用内置兜底实现 —— 让本模块**能独立运行**，
# 不因为模板库不在就整体崩掉（转义逻辑只有 3 行，重复一份的代价远小于
# 一个模块因为 import 失败而完全不可用）。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "slop_templates")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import slop_templates as st        # noqa: E402
except ImportError:
    try:
        import video_templates as st   # noqa: E402
    except ImportError:
        st = None


def _esc_expr(expr):
    """把裸表达式包成可安全放进 -vf 的完整 setpts 滤镜。

    优先用模板库的实现；没有就本地做同样的两处改写（与模板库保持一致）：
      1. `-` -> `+-`   （ffmpeg 8.1.1 对 setpts 里的裸减号有解析缺陷）
      2. 括号内逗号 -> `\\,`（滤镜图在表达式解析前就按逗号切分）
    """
    if st is not None and hasattr(st, "setpts_expr"):
        return st.setpts_expr(expr)
    out, depth = [], 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        out.append("\\," if (ch == "," and depth > 0) else ch)
    return "setpts=" + "".join(out).replace("-", "+-")


# ==================================================================== 载入
def load_beats(json_path):
    """读取 beat_analyze.py 产出的节拍 json。"""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# ==================================================================== 模式 A
def fit_to_beats(src_duration, beat_info, n_downbeats=4,
                 slow_on_downbeat=True, slow_factor=0.45,
                 fast_factor=2.2):
    """把素材变速到"正好 n 个强拍"的长度，且慢放对齐强拍。

    参数
        src_duration      原始素材时长（秒）
        beat_info         load_beats() 的结果
        n_downbeats       目标长度用几个强拍（4 个强拍 ≈ 15 秒 @128BPM/4拍）
        slow_on_downbeat  是否在每个强拍上做慢放（卡点感来源）
        slow_factor       慢放倍率（<1 减速）
        fast_factor       其余部分的速度（>1 加速）

    返回 dict：
        target_duration  目标时长
        anchors          慢放锚点（秒，相对成片时间轴）
        expr             ffmpeg setpts 表达式
        total_out        变速后实际时长（应≈target_duration）
    """
    bpm = beat_info.get("bpm", 0.0)
    if bpm <= 0:
        raise ValueError("BPM 无效，无法卡点")
    period = 60.0 / bpm
    dbeat = period * 4                      # 一个强拍 = 4 拍
    # ⚠ offset 必须参与运算：
    #   BGM 的第 0 个强拍不一定落在 0.0s（前奏/淡入会让首拍后移）。
    #   若忽略 offset，则所有慢放段整体偏移了 offset 秒，画面"慢在最慢"
    #   的那一刻会错过鼓点 —— 且偏差恰好等于 offset（1:1），很容易被漏掉。
    off = beat_info.get("offset", 0.0) or 0.0
    target = dbeat * n_downbeats

    if not slow_on_downbeat:
        # 不做慢放，只总体压缩/拉伸到目标时长
        rate = src_duration / target
        expr = f"setpts=PTS*{1.0 / rate:g}"
        return {"target_duration": target, "anchors": [],
                "expr": expr, "total_out": target}

    # ---- 段结构：每个"强拍周期"= 一段加速 + 一段慢放 ----
    # ⚠ 每段的**输出时长**必须加起来正好等于 target，否则总时长会跑偏。
    #   正确的不变量：Σ(加速输出) + Σ(慢放输出) == target_duration
    #   素材时长则由 素材 = 输出 × 速度 反推，允许与 src_duration 不同
    #   （这正是"视频适配音频"的含义：素材不够就慢放补，多了就加速压）。
    slow_len = dbeat * 0.45
    fast_out = dbeat - slow_len

    seg_specs = []      # (输出时长, 速度)
    acc = 0.0
    for _ in range(n_downbeats):
        seg_specs.append((fast_out, fast_factor))
        seg_specs.append((slow_len, slow_factor))
        acc += fast_out + slow_len
    # 尾巴：补足到 target（浮点误差或 n_downbeats 不整除时）
    if target - acc > 1e-6:
        seg_specs.append((target - acc, fast_factor))
    # 慢放锚点 = 每个慢放段的**起点**在成片时间轴上的位置。
    #
    # ⚠ 语义必须讲清（否则很容易被误读为"锚点应该等于强拍"）：
    #    锚点 = 慢放段的**起点**，而压住强拍的是慢放段的**中点**。
    #    之所以让中点对齐而不是起点对齐：慢放的视觉冲击在段中部最强，
    #    起点对齐会让"最慢的那一帧"落在强拍之后半个段长。
    #    => 校验对齐时要比对的是 (anchor + slow_len/2)，不是 anchor 本身。
    # 成片时间轴相对 BGM 偏移 0（BGM 从第 0 秒起播），
    # 第 k 个强拍位于 off + (k+1)*dbeat（off 为首拍偏移，常为 0 但不一定）。
    # 丢弃落在负轴的锚点（offset 大于一个强拍周期时，首个锚点会在 0 之前）。
    slow_window = slow_len
    anchors = [round(off + (k + 1) * dbeat - slow_len * 0.5, 4)
               for k in range(n_downbeats)]
    anchors = [a for a in anchors if a >= 0]

    # ---- 反推素材段边界 ----
    src_parts = []      # (素材起点, 素材时长, 速度, 输出起点)
    out_t, src_t = 0.0, 0.0
    for out_len, fact in seg_specs:
        src_len = out_len * fact
        src_parts.append((src_t, src_len, fact, out_t))
        out_t += out_len
        src_t += src_len

    # ---- 生成分段 setpts 表达式 ----
    # ffmpeg 的 PTS/T 都是**素材**时间轴；每段内部线性映射，
    # 段间连续：输出 = 该段输出起点 + (PTS - 段起点) / 速度
    #
    # 算法选择：**从后往前的嵌套 if**，而不是"给每段写一个区间掩码再相加"。
    #
    # 为什么必须换掉掩码相加法（实测踩过）：
    #   掩码法里第 i 段的掩码要写成 (T>=a0)*(T<b0)*(T>=a1)*(T<b1)*...
    #   即"我落在第 0..i 段全都满足"—— 这本身就把长度做成了 O(segs²)。
    #   16 个强拍（32 段）时单条表达式长 13.8KB；更大的 n 会继续平方增长，
    #   既难维护（没法手改）也逼近 ffmpeg 的表达式长度上限。
    #
    # 嵌套 if 法：seg_count 个条件 + seg_count 个表达式，**线性**增长，
    #   且语义更直白："如果 T 在第 i 段，就返回第 i 段的映射，否则看下一段"。
    #   嵌套 if 的括号深度上限由 ffmpeg 的 max_nesting 决定（默认 40），
    #   32 层是安全的。
    #
    # ⚠ 逗号必须转义成 `\,`：ffmpeg 的滤镜图在**进入表达式解析器之前**
    #   就按逗号切分滤镜链，`if(lt(T,1.4),A,B)` 会被切碎报 No such filter。
    # ⚠ 减号必须改写成 `+-`：本机 ffmpeg 8.1.1 的解析器对裸减号有缺陷。
    #   这两个坑都由 slop_templates.setpts_expr() 统一处理。
    # ⚠⚠ 因此这里**绝对不要自己写 `\,`**！setpts_expr 会再转义一次，
    #   变成 `\\,` -> ffmpeg 报 `No such filter: '2.2645)\'`。
    #   正确做法是传**裸表达式**（用普通逗号），把转义权完整交给它。
    parts = []
    for (s_start, s_len, fact, o_start) in src_parts:
        cond = f"lt(T,{s_start + s_len:g})"
        body = f"{o_start:g}+(PTS-{s_start:g})/{fact:g}"
        parts.append((cond, body))
    # 从最后一段往前包：else 分支总是"下一段的 if"，最末段用兜底表达式
    expr_body = parts[-1][1]
    for cond, body in reversed(parts[:-1]):
        expr_body = f"if({cond},{body},{expr_body})"
    expr = _esc_expr(expr_body)
    return {"target_duration": target, "anchors": anchors,
            "slow_window": round(slow_window, 4),
            "expr": expr, "total_out": round(out_t, 3),
            "n_segments": len(src_parts),
            "src_needed": round(src_t, 3)}


def flash_points(beat_info, n_downbeats=4, per_beat=True):
    """给出该在哪些时间点闪白/抖动（相对成片时间轴）。

    per_beat=True  -> 每一拍都闪（密集，适合快剪）
    per_beat=False -> 只在强拍闪（克制，推荐）

    ⚠ 单位要分清：n_downbeats 是"强拍个数"，一个强拍 = 4 拍。
      所以窗口长度 = n_downbeats * 4 拍，与 per_beat 无关。
      早期版本 here 把 per_beat=False 的 end 少乘了 4，只产出 4 个点。
    """
    bpm = beat_info.get("bpm", 0.0)
    if bpm <= 0:
        return []
    period = 60.0 / bpm
    off = beat_info.get("offset", 0.0)
    step = period if per_beat else period * 4
    end = off + period * 4 * n_downbeats      # 总覆盖 = 强拍数 × 4 拍
    pts, t = [], off
    while t < end - 1e-9:
        pts.append(round(t, 4))
        t += step
    return pts


# ==================================================================== 模式 B
def fit_audio_to_video(video_duration, beat_info, fade=1.0):
    """画面时长固定时，BGM 怎么处理（截取 + 淡出 / 循环）。

    返回 ffmpeg 音频滤镜字符串。
    """
    bpm = beat_info.get("bpm", 0.0)
    period = 60.0 / bpm if bpm > 0 else 0.0
    # 尽量让 BGM 结束在一个强拍上，避免"切在半拍中间"的突兀感
    if period > 0:
        dbeat = period * 4
        n = max(1, int(round(video_duration / dbeat)))
        cut = min(beat_info.get("duration", video_duration), dbeat * n)
    else:
        cut = video_duration
    return (f"atrim=0:{cut:g},asetpts=N/SR/TB,"
            f"afade=t=out:st={max(0.0, cut - fade):g}:d={fade:g},"
            f"apad,atrim=0:{video_duration:g}")


# ==================================================================== 预览
def summarize(plan, beat_info):
    """打印人类可读的卡点方案。"""
    print(f"  BGM      {beat_info.get('file','?')}  "
          f"{beat_info.get('bpm',0)} BPM")
    print(f"  目标时长 {plan['target_duration']:.3f}s")
    print(f"  变速后   {plan['total_out']:.3f}s  "
          f"({plan.get('n_segments','-')} 段)")
    print(f"  慢放锚点 {len(plan['anchors'])} 个")
    if plan["anchors"]:
        print(f"  前 6 个  {plan['anchors'][:6]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python beat_sync.py <节拍json> [素材时长]")
        raise SystemExit(1)
    bi = load_beats(sys.argv[1])
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    print(f"素材时长 {dur}s")
    plan = fit_to_beats(dur, bi, n_downbeats=4)
    summarize(plan, bi)
    print(f"\n  setpts 表达式（截断）:\n  {plan['expr'][:180]}...")
    print(f"\n  闪白点（只强拍）: {flash_points(bi, 4, per_beat=False)[:8]}")
    print(f"  闪白点（每拍）  : {flash_points(bi, 4, per_beat=True)[:8]}")
