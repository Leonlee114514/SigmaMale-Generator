"""内容感知取段 —— edit_ref.py 与 edit_slop.py **共用**的助手。

为什么要单独成一个模块
----------------------
这套逻辑两个模板都要用：

  · `edit_ref.py`  —— 参考片复刻版（67.12 BPM，硬切，每镜 1 拍）
  · `edit_slop.py` —— 128BPM 卡点版（快剪段 1.35x 变速，每镜 1 拍）

两者的**段落结构不同**（铺垫段长度、镜长、有无变速都不同），
但「怎么挑动作锚点」「铺垫段该放哪」是**同一套**逻辑。

项目已经吃过一次「两处各写一份」的亏：选择函数一旦分成两份，
报告里写的「最多能凑 N 种」和实际选出来的就不是一回事 ——
那种数字比不报还糟。所以：

  · 这里**全部是纯函数**，镜长 / 变速 / 空档 / 预留全部由调用方传入，
    不含任何某个模板特有的常量（不 import 任何编辑器的常量）
  · `select_anchors()` 被搜索和渲染**共用**，这是硬要求

⚠ 这些函数的正确性靠三条纪律，改动前请先读对应注释：
  1. **先按可行域筛、再按强度挑**（顺序反了会吸到不可用的峰值）
  2. **搜索与渲染必须调用同一个 `select_anchors`**
  3. **判据里不要出现固定阈值**（换素材必失效，同一个坑已踩三次）

通用约定
--------
· `od`       = 一个镜**消耗的源时长**（有变速时 = 输出时长 × 速度）
· `peak_pos` = 动作峰值落在镜内的**相对位置**（0=镜首，1=镜尾）
               → 窗口起点 = 峰值 − peak_pos × od
· `spans`    = 允许放窗口的源区间列表 [(lo, hi), ...]
"""

import bisect
import json
import os


def load_clipmap(path):
    """读 `clip_extract.py` 产出的 reel_map.json，转成**原片坐标**下的内容池。

    ⚠ 为什么用原片坐标而不是素材带坐标：
      素材带是把选中的片段重新编码拼起来的，用它当输入会**多一代编码损失**，
      而且它只有 18.32s（原片 25.01s）—— 快剪段要"跳着挑最好的动作"，
      材料越少越施展不开。原片长度充足，直接在上面 trim 即可。

      素材带本身仍然是有用的产物（可以单独播来看"我的动作片段长什么样"），
      只是剪辑器不必经过它。
    """
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    src = m.get("source")
    if src and not os.path.exists(src):
        print(f"原片不存在: {src}")
        return None
    if not src:
        # 老格式的 reel_map 没记 source，退回用素材带本身
        src = m.get("reel")
    clips = m.get("clips") or []
    # 优先用**全部事件峰值**：同一个区间里往往同时有"举手/手到嘴边/摸头/嘟嘴"
    # 好几个动作，只留"最强那个"（片段主导动作）的话，
    # 剪辑器想"每类动作只用一次"就没得选了 —— 实测可行域内会只剩 6 种，凑不满 7 镜。
    ev = m.get("event_peaks") or []
    if ev:
        peaks = [{"t": float(p["t_src"]), "action": p["action"],
                  "peak": p.get("peak", 0)} for p in ev
                 if p.get("t_src") is not None]
    else:
        peaks = [{"t": float(c["peak_src"]), "action": c.get("action"),
                  "peak": c.get("peak", 0)}
                 for c in clips if c.get("peak_src") is not None]
    # 合并成"有动作"的连续区间 —— 用来限制铺垫段滑窗的范围
    regions = []
    for a, b in sorted((c["src_start"], c["src_end"]) for c in clips):
        if regions and a <= regions[-1][1] + 1e-6:
            regions[-1][1] = max(regions[-1][1], b)
        else:
            regions.append([a, b])
    # 从位置 0 起、第一个连续的覆盖区间有多长 —— 铺垫段必须整段落在里面
    first_run_end = 0.0
    for a, b in regions:
        if a <= 1e-6:
            first_run_end = b
            break
    return {"source": src, "regions": regions, "peaks": peaks,
            "first_run_end": first_run_end,
            "reel": m.get("reel"), "reel_dur": m.get("reel_duration")}


# ---------------------------------------------------------------- 动作权重策略
# 画面上「几乎看不出来」的动作。**不是不能用，而是在挑锚点时排到最后**：
#   眨眼只有 0.03~0.13s，切点压上去观众根本意识不到发生了什么；
#   但它的检测强度（0.4~0.5）并不低，纯按强度排会挤掉真正有戏的动作。
# 排到最后而不是剔除：动作池够用时永远轮不到它，
#   池子不够时还能兜底，不会让某些镜因为「没候选」退回顺序取用。
# ⚠ 这是**两个模板共用的策略**，所以要放在这里而不是各自的文件里。
WEAK_ACTIONS = frozenset({"眨眼"})


INTRO_STEP = 1.0 / 30         # 铺垫段位置的搜索步长（一帧）


def _window_of(p, od, peak_pos):
    """某个动作峰值对应的"锚点窗口" [start, end] —— 峰值落在镜内 peak_pos 处。"""
    s = p["t"] - peak_pos * od
    return s, s + od


def _cands_in_spans(peaks, spans, od, peak_pos):
    """窗口能**整段**落进某个空档的峰值 —— 候选锚点。

    因为候选的可行域是反解出来的（而不是选完再钳位），
    后面放置时 `p = 峰值 − peak_pos×镜长` 必然精确，不需要任何 clamp。
    """
    out = []
    for p in peaks:
        s, e = _window_of(p, od, peak_pos)
        if any(s >= lo - 1e-6 and e <= hi + 1e-6 for lo, hi in spans):
            out.append(p)
    return out


def _spans_for_intro(off, intro_len, src_dur, od, reserve=0.0):
    """铺垫段占住 [off, off+intro_len] 之后，锚点窗口还能落在哪些空档。

    ⚠ 铺垫段**两侧**都能放锚点。旧实现写死了"锚点必须在铺垫段之后"，
      那只是当初顺序放置自然形成的结果，**不是设计约束** ——
      而它有明确的失效场景：素材开头就是最精华的一段时，
      铺垫段必然把它整段吃掉，锚点只能在剩下的次等素材里挑。
      实测本片：铺垫段最初被放在 0.833s（而不是 0），
      就正好吞掉了 8.865s 的「抬头」—— 那是第 7 种动作类型。

    ⚠ 第二段尾部留出 `reserve` 给结尾淡出。这样 [src−reserve, src] 恒为空档，
      结尾段一定放得下（否则锚点可能把尾部切碎到放不下 1.788s 的结尾）。
      只有当留出后仍放不下一个镜头时才不留 —— 那时结尾退到别的空档。
    """
    spans = []
    if off > od:                                  # 铺垫段之前
        spans.append((0.0, off))
    tail = src_dur - (off + intro_len)            # 铺垫段之后
    if tail > od:
        keep = src_dur - reserve if tail - reserve > od else src_dur
        spans.append((off + intro_len, keep))
    return spans


def _free_gaps(src_dur, occupied, min_len):
    """时间轴被 occupied 占掉后剩下的空档（只留放得下 min_len 的）。"""
    gaps, cur = [], 0.0
    for a, b in sorted(occupied):
        if a - cur >= min_len - 1e-9:
            gaps.append((cur, a))
        cur = max(cur, b)
    if src_dur - cur >= min_len - 1e-9:
        gaps.append((cur, src_dur))
    return gaps


def _clamp_free(t, dur, spans, src_dur):
    """把 t 钳进第一个放得下 dur 的空档（用于"没有候选、退回顺序取用"的兜底）。"""
    for lo, hi in spans:
        if hi - lo >= dur - 1e-9:
            return min(max(t, lo), hi - dur)
    return min(max(t, 0.0), max(0.0, src_dur - dur))


def free_gaps(spans, used, min_len):
    """spans 扣掉已占用窗口后，还能放下 min_len 的空档。"""
    out = []
    for lo, hi in spans:
        cuts = [(lo, hi)]
        for a, b in used:
            nxt = []
            for c, d in cuts:
                if b <= c + 1e-9 or a >= d - 1e-9:
                    nxt.append((c, d))
                    continue
                if a > c + 1e-9:
                    nxt.append((c, min(a, d)))
                if b < d - 1e-9:
                    nxt.append((max(b, c), d))
            cuts = nxt
        out += [(c, d) for c, d in cuts if d - c >= min_len - 1e-9]
    return out


def fallback_in_pt(t, dur, spans, src_dur, used):
    """兜底入点：**优先落在没用过的空档**，全占满了才退化成"离已用最远"。

    ⚠⚠ 为什么必须有这个函数（原来的兜底是直接调 `_clamp_free`）：

      `_clamp_free` 是"取第一个放得下的空档，把 t 钳进去"。
      当空档只剩**一个**的时候，**所有**超出范围的 t 都会被钳到
      **同一个位置** —— 于是十几个镜头位叠在同一帧上。

      实测症状：成片里"同一个动作连续回放好几次"（用户原话
      "一模一样的动作持续回放四次"），而报告只写一句"退回顺序取用"，
      看起来像素材动作不够，实际是兜底把画面叠在了一起。
      单段 15s 的素材最典型：铺垫段吃掉 8.09s，只剩 6.9s 空档 →
      最多放 10 个锚点，而快剪段要 22 个位 → 12 个位全被钳到同一处。

    三级降级（越往后越不理想，但都不会"挨着重复"）：
      1. 从**最宽的空档**里顺序取（完全没重叠）
      2. 全占满 → 取**离所有已用窗口中心最远**的位置（隔开重复）
      3. 连 spans 都放不下 → 才退回 `_clamp_free`
    """
    gaps = free_gaps(spans, used, dur)
    if gaps:
        g = max(gaps, key=lambda x: x[1] - x[0])
        return min(max(t, g[0]), g[1] - dur)

    # 全占满：扫描找离已用窗口最远的位置。
    # 宁可"隔几秒再重复一次"，也不要"相邻两镜同一段画面"——
    # 后者观众一眼就看出是复制粘贴，前者几乎察觉不到（同 select_anchors 第二轮的道理）。
    best, best_d = None, -1.0
    for lo, hi in spans:
        if hi - lo < dur - 1e-9:
            continue
        n = int((hi - lo - dur) / INTRO_STEP)
        for i in range(n + 1):
            ct = lo + i * INTRO_STEP
            mid = ct + dur * 0.5
            d = min((abs(mid - (a + b) * 0.5) for a, b in used), default=1e9)
            if d > best_d:
                best_d, best = d, ct
    if best is not None:
        return best
    return _clamp_free(t, dur, spans, src_dur)


def _richness(sel):
    """一段连续窗口自身的丰富度：每种动作取窗口内最强的一个再求和。

    ⚠ 用"按种类取最强再求和"，而不是把所有峰值直接相加。直接相加的坑：
      眨眼每次 0.4~0.5、一段里能有五六个，于是一段"全是眨眼"的窗口
      能靠数量压过"一次举手 + 一次微笑"的窗口 —— 但眨眼画面上几乎看不出来。
    """
    top = {}
    for p in sel:
        a = p["action"]
        if a not in top or abs(p["peak"]) > abs(top[a]["peak"]):
            top[a] = p
    return sum(abs(p["peak"]) for p in top.values())


def parse_action_plan(s):
    """把 `--action-plan` 的字符串解析成 `{槽位下标(0-based): 动作名}`。

    格式：`"3=举手,7=转头"` —— 槽位**从 1 开始**编号（对用户友好），内部转 0-based。
    也接受全角逗号与多余空白：`"3 = 举手 ， 7=转头"`。

    ⚠ 格式错误**必须抛错，不能静默忽略** —— 用户写了 `3:举手` 或
      `3=不存在的动作` 却什么都没发生，正是本项目反复踩的那类"不报错但结果错"。
      所以这里一律 `raise ValueError`，由调用方决定退出还是告警。
      （"动作名不存在"要到拿到 reel_map 才知道，在调用方校验。）
    """
    pins = {}
    if not s:
        return pins
    for part in str(s).replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"--action-plan 格式错误：'{part}'（应为 槽位=动作，如 3=举手）")
        k, _, act = part.partition("=")
        k, act = k.strip(), act.strip()
        if not k.isdigit():
            raise ValueError(
                f"--action-plan 的槽位必须是数字：'{part}'（槽位从 1 开始）")
        if not act:
            raise ValueError(f"--action-plan 缺少动作名：'{part}'")
        idx = int(k) - 1
        if idx < 0:
            raise ValueError(f"--action-plan 槽位从 1 开始，不能是 {k}")
        pins[idx] = act
    return pins


def select_anchors(cand, n, od, weak=frozenset(), pins=None):
    """从候选里挑 n 个锚点：**每类动作只用一次**，类型不够时才放开重复。

    ⚠ 为什么要有"每类一次"这一轮：纯按强度排实测 7 个位里挑到 3 个眨眼、
      2 个转头，同一种动作反复出现，观感上就是"怎么又是这个"。
      第一轮只接受该类型第一次出现的峰值，同类型更弱的直接跳过；
      第二轮才放开重复补齐（不让某些镜因为"没候选"退回顺序取用）。

    ⚠ 第二轮用"**离同类上一次最远**"而不是"强度最高"：同种动作出现在相邻两镜里
      一眼就看出"又是这个"，隔着好几秒再用一次则几乎察觉不到。
      —— 用户对这条的原话是"同类的可以复用一次，这是很好的降级"。

    ⚠ 这个函数**同时被"搜索最优铺垫段位置"和"正式渲染"调用**。
      必须共用：否则报告里写的"最多能凑到 N 种"与实际选出来的不是一回事，
      那种数字比不报还糟。

    `pins`（可选）= `{槽位下标(0-based): 动作名}` —— 用户在 `--action-plan` 里指定的
    "第 k 个实拍位放哪个动作"。语义：

      · **无 pins** → 完全走原路径，返回**按时间排序**的列表（行为与历史逐字段一致）。
      · **有 pins** → 先按槽位序钉住指定动作，再自动补其余槽位，
        返回**按槽位序**排列的列表（下标 k 就是槽位 k）。

      ⚠ 有 pins 时**不能**再按时间排序：`apply_content_aware` 是**按位置**把结果
        映射到槽位的（chosen[k] → 槽位 k）。要保证"槽位 3 = 举手"，下标 k 就必须
        是槽位 k，而不能是"第 k 早的锚点"。

      ⚠ 指定的动作**允许重复**（写两次"举手"就出两次），不受"每类只用一次"限制 ——
        那是**自动阶段**的策略，用户的明确指定优先。

      ⚠ 某个指定槽位在可行域内找不到该动作时**不报错中断**，该槽位退回自动补齐；
        落实与否由调用方在报告里逐条列出（`pin` / `pin_ok`），
        并由 `verify_contentaware.py` 做机器验收 —— 不允许静默失败。
    """
    pool = sorted(range(len(cand)),
                  key=lambda i: (cand[i]["action"] in weak, -abs(cand[i]["peak"])))

    # ------------------------------------------------ 原路径（无指定，行为不变）
    if not pins:
        chosen, used = [], set()

        def fits(i):
            return all(abs(cand[i]["t"] - cand[j]["t"]) >= od - 1e-6 for j in chosen)

        for i in pool:                               # 第一轮：每类动作只用一次
            if len(chosen) == n:
                break
            if cand[i]["action"] in used or not fits(i):
                continue
            chosen.append(i)
            used.add(cand[i]["action"])

        while len(chosen) < n:                       # 第二轮：放开重复补齐
            rest = [i for i in pool if i not in chosen and fits(i)]
            if not rest:
                break

            def farness(i):
                d = [abs(cand[i]["t"] - cand[j]["t"]) for j in chosen
                     if cand[j]["action"] == cand[i]["action"]]
                return min(d) if d else 1e9

            chosen.append(max(rest, key=lambda i: (farness(i), abs(cand[i]["peak"]))))

        return [cand[i] for i in sorted(chosen, key=lambda i: cand[i]["t"])]

    # ------------------------------------------------ 指定模式（下标 k == 槽位 k）
    slots = [None] * n
    used_idx, used_act = set(), set()

    def fits_slot(i):
        return all(abs(cand[i]["t"] - cand[j]["t"]) >= od - 1e-6
                   for j in slots if j is not None)

    for k in sorted(pins):                           # ① 先钉指定槽位（按强度挑）
        if not 0 <= k < n:
            continue                                 # 越界槽位由调用方告警
        want = pins[k]
        for i in pool:
            if i in used_idx or cand[i]["action"] != want or not fits_slot(i):
                continue
            slots[k] = i
            used_idx.add(i)
            used_act.add(want)
            break

    for k in range(n):                               # ② 自动补：每类一次
        if slots[k] is not None:
            continue
        for i in pool:
            if i in used_idx or cand[i]["action"] in used_act or not fits_slot(i):
                continue
            slots[k] = i
            used_idx.add(i)
            used_act.add(cand[i]["action"])
            break

    for k in range(n):                               # ③ 自动补：放开重复
        if slots[k] is not None:
            continue
        rest = [i for i in pool if i not in used_idx and fits_slot(i)]
        if not rest:
            continue

        def farness_pin(i):
            d = [abs(cand[i]["t"] - cand[j]["t"]) for j in slots
                 if j is not None and cand[j]["action"] == cand[i]["action"]]
            return min(d) if d else 1e9

        i = max(rest, key=lambda i: (farness_pin(i), abs(cand[i]["peak"])))
        slots[k] = i
        used_idx.add(i)

    # ④ 非指定槽位按时间升序重排 —— 尽量保住"越靠前的镜取越靠前的源"。
    #    无 pins 时的历史行为就是时间升序（末尾那行 sorted）；只钉住指定槽位、
    #    让其余锚点回到时间序，成片观感才与历史一致。
    #    ⚠ 重排只是**在槽位间置换**同一批锚点，两两时间间隔不变 → 不会破坏不重叠约束。
    #    ⚠ 指定**失败**的槽位算自由槽位（它已经被自动补齐，没有约束要守）。
    pinned_ok = {k for k in pins
                 if 0 <= k < n and slots[k] is not None
                 and cand[slots[k]]["action"] == pins[k]}
    free_slots = [k for k in range(n) if k not in pinned_ok and slots[k] is not None]
    free_idx = sorted((slots[k] for k in free_slots), key=lambda i: cand[i]["t"])
    for k, i in zip(free_slots, free_idx):
        slots[k] = i

    return [cand[i] for i in slots if i is not None]


def pick_intro_offset(peaks, intro_len, hi, od, src_dur, peak_pos, n_shots,
                      reserve=0.0, weak=frozenset(), pins=None):
    """选铺垫段的起始位置 —— 判据是**它给快剪段留下什么**，不是它自己装了什么。

    ⚠⚠ 这里踩过一个"**目标函数本身选错**"的坑，比实现 bug 隐蔽得多，值得记全：

      旧判据是"窗口里动作最丰富的一段"，理由听起来很自然 —— 铺垫段不该无聊。
      但**铺垫段是连续长镜**，窗口里的事件是被连续播出来的，根本不作为锚点使用；
      而窗口每往后挪一点，就多吃掉一点本来能给快剪段当锚点的素材。
      于是这个判据在**优化一个不参与产出的量，代价却是产出本身**。

      实测（25s 素材、7 个快剪实拍位）：
        铺垫段起点 0.000 → 可放 7 种不同动作
        铺垫段起点 0.833 → 只剩 6 种      ← 旧判据选的正是这个
        铺垫段起点 1.667 → 只剩 5 种
      起点 0.833 刚好吞掉 8.865s 的「抬头」，那正是可行域内第 7 种类型。
      症状看起来是"素材动作密度不够、上限就是 6 种"，
      实际是**自己把第 7 种吃掉了**，却被报告解释成素材的问题。

      现在：对每个候选位置先算出**它能留下的锚点集合**，再按
        (能填满几个镜, 能覆盖几种动作, 锚点总强度, 铺垫段自身丰富度, 越靠前越好)
      字典序取最优。前两项是目标；第三、四项同级择优；最后一项只用于打破平局。
      铺垫段自身丰富度降为**末位**而不是主导 —— 它不该为了自己不无聊去偷素材。

    ⚠ 搜索范围也修过一次：旧实现的上限取自 `first_run_end`
      （"从 0 开始第一个连续的动作覆盖区"），本意是"铺垫段要落在一段有内容的素材里"。
      但那个边界是 `clip_extract` 的分区拼出来的，**极其脆弱** ——
      实测本片 4.82→4.88 之间有个 **0.067 秒**的空隙，
      就把上限从 14.425 砍到 4.818，于是"把铺垫段挪到后面、
      把开头的表情让给快剪段"这个选项**优化器根本没被给到**。
      现在改成搜整条素材，改用下面那条**中位数门槛**来保证铺垫段不冷清。

    `pins`（可选）透传给 `select_anchors`，并把「**指定动作落实了几个**」提到判据**首位**。
    为什么必须提到首位：铺垫段放在哪，决定了快剪段还剩哪些空档、哪些动作可行。
    某些位置可能让指定动作**没有候选**（例如那个动作的峰值正好被铺垫段吃掉）。
    不把落实数计入判据的话，优化器会选一个"种类数看起来一样多、但指定动作全落空"的位置 ——
    而用户明明指定了。无 pins 时该项恒为 0，**字典序与历史完全一致**。

    返回 (位置, 窗口内的事件, info)。info 里带上锚点统计与门槛信息，供报告解释"为什么选它"。
    """
    hi = max(0.0, hi)
    cands = []
    k = 0
    while k * INTRO_STEP <= hi + 1e-9:
        off = min(k * INTRO_STEP, hi)
        spans = _spans_for_intro(off, intro_len, src_dur, od, reserve)
        cand = _cands_in_spans(peaks, spans, od, peak_pos)
        chosen = select_anchors(cand, n_shots, od, weak, pins)
        sel = [p for p in peaks
               if off - 1e-6 <= p["t"] <= off + intro_len + 1e-6]
        rich = _richness(sel)
        # 指定动作落实了几个 —— 有 pins 时置首位，无 pins 时恒为 0（不改历史排序）
        n_pin_ok = sum(1 for k2, act in (pins or {}).items()
                       if 0 <= k2 < len(chosen) and chosen[k2]["action"] == act)
        key = (n_pin_ok,
               len(chosen),
               len({p["action"] for p in chosen}),
               round(sum(abs(p["peak"]) for p in chosen), 6),
               round(rich, 6),
               -off)                       # 平局时取更靠前的，成片从素材开头开始更自然
        cands.append((key, off, chosen, sel, rich))
        k += 1
    if not cands:
        return 0.0, [], {"n_chosen": 0, "n_distinct": 0, "strength": 0.0,
                         "intro_score": 0.0, "n_pin_ok": 0, "chosen": []}

    # ---- 可行性门槛：铺垫段至少要"不低于本素材里一个典型的 8 秒" ----
    #
    # ⚠ 为什么需要这条：把判据完全交给"给快剪段留下什么"之后，
    #   优化器理论上可以把铺垫段丢在**完全没人动的一段**上换取锚点。
    #   而铺垫段占成片 1/3 时长，摆一段静止画面是观感事故。
    #
    # ⚠ 门槛为什么取**中位数**而不是某个常数：中位数由**这份素材自己**决定，
    #   不掺入任何与素材无关的调参。含义是"铺垫段不能比本素材的典型 8 秒还冷清"。
    #   实测（25s 素材）：丰富度中位 3.679、最大 8.095，
    #   而锚点最优位置（15.233s）的丰富度是 4.029 —— **高于中位，门槛不生效**。
    #   它只是护栏，不参与择优；素材一变也不会因为门槛过紧而卡死
    #   （万一全部候选都不达标，退回全体里择优）。
    richs = sorted(c[4] for c in cands)
    floor = richs[len(richs) // 2]
    feasible = [c for c in cands if c[4] >= floor - 1e-9] or cands
    floor_relaxed = False
    if pins:
        # ⚠⚠ 护栏与"用户指定"冲突时，**指定优先**。踩过：
        #   指定「举手」时，该动作的峰值（10.51 / 13.00s）只有在铺垫段起点
        #   ≥ 10.93s 时才不被吃掉；而那些位置的丰富度**全部**低于中位数门槛
        #   （2.076 vs 2.196，只低 5%），于是护栏把唯一可行的区间整段砍掉，
        #   指定**永远落不了地** —— 而用户会以为是自己把动作名写错了。
        #   所以：护栏只用于**同等落实数之间**的择优；若落实最多的那一档
        #   整体被护栏挡住，就放开护栏（用户明确指定 > 观感护栏），
        #   并在 info 里置 `floor_relaxed`，由调用方在报告里说清楚。
        best_pin = max(c[0][0] for c in cands)
        top = [c for c in cands if c[0][0] == best_pin]
        top_over = [c for c in top if c[4] >= floor - 1e-9]
        if top_over:
            feasible = top_over
        else:
            feasible = top
            floor_relaxed = True
    best = max(feasible, key=lambda c: c[0])
    key, off, chosen, sel, rich = best
    return off, sel, {"n_pin_ok": key[0],
                      "n_chosen": key[1], "n_distinct": key[2],
                      "strength": key[3], "intro_score": rich,
                      "rich_floor": floor, "rich_best": richs[-1],
                      "floor_relaxed": floor_relaxed,
                      "n_feasible": len(feasible), "n_cand": len(cands),
                      "chosen": chosen}


def max_distinct_actions(peaks, win, peak_pos, spans):
    """在 spans（若干空档）内最多能放进多少**不同种类**的动作。

    用来回答"为什么做不到每类只出现一次" —— 这个数字不该靠猜。

    做法是**精确 DP**（不是贪心）：
      · 候选窗口 = 每个峰值对应的 [peak − peak_pos×win, +win]，须整段落在某个空档里
      · 按结束时间排序，dp[i] = 只考虑前 i 个窗口时可达的"动作集合"（位掩码）
      · dp[i] = dp[i−1]（不选）∪ { mask | bit(动作) | mask ∈ dp[j] }（选，j 为
        最后一个与它不重叠的窗口）
      · 答案 = 所有可达掩码里 popcount 的最大值
    动作种类 ≤ 14 → 掩码 ≤ 16384；窗口几十个 → 状态量完全可接受。

    ⚠ 位掩码必须是 `1 << 类型序号`，**不能直接拿序号当掩码**。
      踩过的坑：调试时写成 `bit = ai[action]`，于是
      `抬头(序号 8) | 眨眼(序号 11) == 11 == 眨眼`，组合被集合去重吃掉。
      症状极隐蔽 —— 小用例（2~3 个窗口）结果正常，窗口一多就崩，
      表现成"DP 说 4 种、贪婪却找出 7 种"这种自相矛盾，
      很容易误判成 DP 逻辑错，其实是位运算错。
    """
    acts = sorted({p["action"] for p in peaks})
    ai = {a: i for i, a in enumerate(acts)}
    wins = []
    for p in peaks:
        s, e = _window_of(p, win, peak_pos)
        if not any(s >= lo - 1e-6 and e <= hi + 1e-6 for lo, hi in spans):
            continue
        wins.append((s, e, 1 << ai[p["action"]]))
    if not wins:
        return 0, 0
    wins.sort(key=lambda w: w[1])
    ends = [w[1] for w in wins]
    import bisect
    # 最后一个"结束 ≤ 本窗口开始"的窗口下标（0-based），-1 表示没有
    prev = [bisect.bisect_right(ends, w[0] - 1e-9) - 1 for w in wins]

    n = len(wins)
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for i in range(1, n + 1):
        dp[i] |= dp[i - 1]                       # 不选第 i 个
        j = prev[i - 1] + 1
        bit = 1 << wins[i - 1][2]
        for mask in dp[j]:
            dp[i].add(mask | bit)                # 选它
    best = max(bin(m).count("1") for m in dp[n]) if dp[n] else 0
    avail = len({w[2] for w in wins})
    return best, avail

