"""动作片段摘取（clip_extract.py）
====================================================================
把 `action_detect.py` 检出的事件，从原片里**切出来**，成为可直接用于剪辑的素材。

为什么单独一步（而不是直接在 edit_slop 里用时间码）：
    切出来的片段可以**复用、打乱、反复拼**，而且能先人工过一遍筛掉误检。
    直接按时间码在滤镜链里 trim，改一次编排就要重算整条链，
    调试成本高得多。

用法：
    # 先检测
    python action_detect.py 素材.mp4 -o 动作.json

    # 再摘片段（默认每个事件前后各留 0.1s，默认做不重叠去重）
    python clip_extract.py 素材.mp4 动作.json -o clips/

    # 只要最强的 12 个、并按动作类型分目录
    python clip_extract.py 素材.mp4 动作.json -o clips/ --top 12 --by-action

    # 额外拼一条"动作集锦"（按时间顺序连起来）
    python clip_extract.py 素材.mp4 动作.json -o clips/ --reel 集锦.mp4

去重模式（`--dedup`，默认 partition）：
    partition  把时间轴切成不重叠分区，每个时刻归属强度最高的动作。
               得到的是"覆盖满、无重复"的素材池 —— 当剪辑素材用选这个。
    greedy     按强度贪心保留、丢弃重叠的（非极大值抑制）。
               宁可留空白也不放弱动作进去，适合"只要最强的几个"。
    none       旧行为：每个事件各切一片，重叠照切。
               ⚠ 会产生大量重复片段（实测 43 片合计 39.87s，
                 去重后其实只有 16.71s）。

输出：
    clips/<动作名>/<序号>_<动作>_<起始秒>.mp4     （--by-action 时）
    clips/<序号>_<动作>_<起始秒>.mp4
    clips/manifest.json                           （清单，供后续编排读取）
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")


# ------------------------------------------------------------------ 去重
def coverage(evs):
    """合并后**不重叠**的总覆盖时长。"""
    iv = sorted((e["start"], e["end"]) for e in evs)
    mg = []
    for a, b in iv:
        if mg and a <= mg[-1][1]:
            mg[-1][1] = max(mg[-1][1], b)
        else:
            mg.append([a, b])
    return sum(b - a for a, b in mg), mg


def partition(evs, pad, src_dur):
    """把重叠事件切分成**不重叠的划分**：每个时刻归属强度最高的动作。

    为什么必须做：
      同一瞬间常被多个检测器同时命中（举手 + 转头 + 微笑 + 摸头…），
      各自切一个片段，于是**同一段画面被切成好几个"不同素材"**。
      实测 5.36 / 5.39 / 5.43 / 5.46 / 5.53 / 5.53 六个片段全落在
      0.17 秒内 —— 43 个片段合计 39.87s，去重后其实只有 16.71s。
      拿这种池子当素材会大量重复，而且"片段数"这个数字完全虚高。

    做法（扫描线）：
      1. 每个事件扩成 [start-pad, end+pad]，权重取 |peak|
      2. 收集全部端点并排序，相邻端点构成"初等区间"
      3. 每个初等区间归属**覆盖它的权重最大**的那个事件
      4. 相邻且同归属的初等区间合并成一段

    结果是一条时间轴的**划分**：无重叠、无遗漏、每个时刻唯一归属。
    """
    ivs = []
    for e in evs:
        t0 = max(0.0, e["start"] - pad)
        t1 = e["end"] + pad
        if src_dur:
            t1 = min(t1, src_dur)
        if t1 - t0 > 1e-6:
            # 第 6 项 peak_t 是关键：它标出"这段里动作最强的那一帧"，
            # 剪辑器要靠它把切点吸附到动作上（见 edit_ref.py --clips）
            ivs.append((t0, t1, e["action"], abs(e.get("peak", 0.0)),
                        e.get("peak", 0.0), e.get("peak_t", e["start"])))
    if not ivs:
        return []

    bounds = sorted({b for iv in ivs for b in iv[:2]})
    runs = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        mid = (a + b) / 2
        cand = [iv for iv in ivs if iv[0] <= mid < iv[1]]
        if not cand:
            continue
        # 权重相同时取更长的（信息量更大），再相同则保持稳定顺序
        best = max(cand, key=lambda iv: (iv[3], iv[1] - iv[0]))
        if (runs and runs[-1]["action"] == best[2]
                and abs(runs[-1]["end"] - a) < 1e-6):
            runs[-1]["end"] = b
            if abs(best[4]) > abs(runs[-1]["peak"]):
                runs[-1]["peak"] = best[4]
                runs[-1]["peak_t"] = best[5]
        else:
            runs.append({"action": best[2], "start": a, "end": b,
                         "peak": best[4], "peak_t": best[5], "padded": True})
    # peak_t 必须落在本段区间内（扩 pad 后区间会变，原来的峰值时刻
    # 有可能被切在外面）
    for r in runs:
        r["peak_t"] = min(max(r["peak_t"], r["start"]), r["end"])
    return runs


def greedy_nms(evs, pad, src_dur):
    """按强度贪心保留、丢弃与已选区间重叠的（非极大值抑制）。

    与 `partition` 的区别：partition 保证**覆盖满整个时间轴**；
    greedy 只保留最强的若干，宁可留空白也不放弱动作进去。
    需要"宁缺毋滥"的短素材池时用这个。
    """
    ranked = sorted(evs, key=lambda e: -abs(e.get("peak", 0.0)))
    kept = []
    for e in ranked:
        t0 = max(0.0, e["start"] - pad)
        t1 = e["end"] + pad
        if src_dur:
            t1 = min(t1, src_dur)
        if t1 - t0 <= 1e-6:
            continue
        if any(t0 < k["end"] - 1e-6 and k["start"] < t1 - 1e-6 for k in kept):
            continue
        kept.append({"action": e["action"], "start": t0, "end": t1,
                     "peak": e.get("peak", 0.0),
                     "peak_t": min(max(e.get("peak_t", e["start"]), t0), t1),
                     "padded": True})
    kept.sort(key=lambda e: e["start"])
    return kept


def split_long(ev, max_dur):
    """把超长区间按 max_dur **等分**成多块。

    ⚠ 不按"以峰值为中心裁一刀"来截短 —— 那会丢掉区间两端的有效内容，
      而等分是可预测的：不丢内容，只是拆成更小的可编排单元。

    ⚠ `peak_t` 只留给**真正包含它的那一块**，其余块不继承。
      若每块都把自己的 peak_t 钳到边界，就会凭空造出一堆"假峰值"，
      剪辑器照着吸附就会吸到一堆同一瞬间的重复锚点上。
    """
    dur = ev["end"] - ev["start"]
    if dur <= max_dur + 1e-6:
        return [ev]
    k = int(dur / max_dur) + 1
    step = dur / k
    pt = ev.get("peak_t")
    out = []
    for i in range(k):
        t0 = ev["start"] + i * step
        t1 = ev["start"] + (i + 1) * step
        piece = {k2: v for k2, v in ev.items() if k2 != "peak_t"}
        piece["start"], piece["end"] = t0, t1
        if pt is not None and t0 <= pt <= t1:
            piece["peak_t"] = pt
        out.append(piece)
    return out


def build_reel(src, clips, out_path, out_dir, crf, evs=None):
    """从**原片**一次性拼出素材带，并写出映射表。

    ⚠ 为什么不用"把切好的片段再 concat 一遍"：
      那是 解码→编码 → 再解码→再编码，白多一代损失。
      这里每个区间直接从原片 trim，concat 后只编码一次。

    映射表（reel_map.json）是给剪辑器 `edit_ref.py --clips` 用的：
      它把"素材带上的**虚拟时间**"映回"原片的**真实时间**"，
      并给出每个动作峰值在素材带上的坐标 —— 剪辑器据此把切点吸到动作上。

    ⚠ 写**两份**峰值：
      · `peaks`       每个片段一个（去重后的**主导动作**）——
                      代表"这段画面最像什么动作"
      · `event_peaks` 全部检测事件（同一个片段里可能有好几个）——
                      剪辑器用它做锚点选择。
        为什么要两份：去重是按强度取主导动作，一个区间里
        "举手 / 手到嘴边 / 摸头 / 嘟嘴"可能同时存在，只留最强那个的话，
        剪辑器想"每类动作只用一次"就没得选了 ——
        实测可行域内只剩 6 种动作，凑不满 7 个快剪位。

    ⚠ 这也是为什么必须重新扫一遍 `peak_t` 的合法性：
      片段可能被 pad 扩过、被 max_dur 拆过，原来记的峰值时刻
      未必还落在片段区间内。落到外面的直接不记（宁可少一个锚点，
      也不要一个假的 —— 剪辑器吸到假锚点会吸到空白画面）。
    """
    ivs = [(m["src_start"], m["src_start"] + m["dur"]) for m in clips]
    n = len(ivs)
    if n == 0:
        return None
    spl = "[0:v]" + f"split={n}" + "".join(f"[s{i}]" for i in range(n))
    trims = "".join(f"[s{i}]trim=start={t0:.4f}:end={t1:.4f},"
                    f"setpts=PTS-STARTPTS[p{i}];"
                    for i, (t0, t1) in enumerate(ivs))
    cat = "".join(f"[p{i}]" for i in range(n))
    fc = f"{spl};{trims}{cat}concat=n={n}:v=1:a=0[vout]"
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", src,
         "-filter_complex", fc, "-map", "[vout]",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-an", out_path],
        capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(out_path):
        print(f"  素材带生成失败: {(r.stderr or '')[:200]}")
        return None

    cur = 0.0
    mapping, peaks = [], []
    for m, (t0, t1) in zip(clips, ivs):
        entry = {"file": m["file"], "action": m["action"],
                 "reel_start": round(cur, 4),
                 "reel_end": round(cur + m["dur"], 4),
                 "src_start": round(t0, 4), "src_end": round(t1, 4),
                 "dur": round(m["dur"], 4), "peak": m.get("peak", 0)}
        pt = m.get("peak_t")
        if pt is not None and t0 - 1e-6 <= pt <= t1 + 1e-6:
            pr = cur + (pt - t0)
            entry["peak_reel"] = round(pr, 4)
            # ⚠ 同时保留**原片坐标**下的峰值时刻。
            #   剪辑器最终是直接在**原片**上 trim 的（不经过素材带，
            #   省掉一代重编码），所以它需要的是原片坐标，不是虚拟坐标。
            entry["peak_src"] = round(float(pt), 4)
            peaks.append({"t": round(pr, 4), "t_src": round(float(pt), 4),
                          "action": m["action"], "peak": m.get("peak", 0)})
        mapping.append(entry)
        cur += m["dur"]

    # 全部检测事件（原片坐标）—— 供剪辑器做"每类动作只用一次"的选择
    ev_peaks = []
    for e in (evs or []):
        pt = e.get("peak_t")
        if pt is None or pt < 0:
            continue
        ev_peaks.append({"t_src": round(float(pt), 4),
                         "action": e["action"], "peak": e.get("peak", 0)})
    ev_peaks.sort(key=lambda x: x["t_src"])

    mp = os.path.join(out_dir, "reel_map.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({"reel": os.path.abspath(out_path),
                   "source": os.path.abspath(src),
                   "reel_duration": round(cur, 4),
                   "clips": mapping,
                   "peaks": sorted(peaks, key=lambda p: p["t"]),
                   "event_peaks": ev_peaks},
                  f, ensure_ascii=False, indent=2)
    print(f"  素材带 -> {out_path}（{n} 段 / {cur:.2f}s）")
    print(f"  映射表 -> {mp}（{len(peaks)} 个片段主导峰值 + "
          f"{len(ev_peaks)} 个事件峰值）")
    return cur


def main():
    ap = argparse.ArgumentParser(description="从检测结果摘取动作片段")
    ap.add_argument("video", help="原素材")
    ap.add_argument("events", help="action_detect.py 产出的 JSON")
    ap.add_argument("-o", "--out-dir", default="clips", help="输出目录")
    ap.add_argument("--pad", type=float, default=0.10,
                    help="事件前后各留多少秒（默认 0.10，避免动作被切头去尾）")
    ap.add_argument("--min-dur", type=float, default=0.20,
                    help="片段最短时长（默认 0.20；太短的动作没有观感价值）")
    ap.add_argument("--max-dur", type=float, default=2.00,
                    help="片段最长时长（默认 2.00；超过就截断，避免拖沓）")
    ap.add_argument("--top", type=int, default=0,
                    help="只取强度最高的 N 个事件（0=全部）")
    ap.add_argument("--dedup", choices=["partition", "greedy", "none"],
                    default="partition",
                    help="去重模式（默认 partition；none = 旧行为，重叠照切）")
    ap.add_argument("--actions", help="只要这些动作（逗号分隔，如 举手,微笑）")
    ap.add_argument("--by-action", action="store_true", help="按动作类型建子目录")
    ap.add_argument("--reel", metavar="PATH", help="额外拼一条集锦（按时间顺序）")
    ap.add_argument("--crf", type=int, default=18, help="输出质量")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        print(f"素材不存在: {a.video}")
        return 1
    if not os.path.exists(a.events):
        print(f"事件文件不存在: {a.events}")
        return 1

    with open(a.events, encoding="utf-8") as f:
        data = json.load(f)
    evs = data.get("events") or []
    # 留一份**未过滤**的副本：下面的去重/筛选会改 evs，
    # 但 reel_map 里要写全部事件峰值（供剪辑器挑锚点）
    evs_raw = list(evs)
    src_dur = float(data.get("duration") or 0) or None
    if not evs:
        print("事件列表为空")
        return 1
    print(f"读入 {len(evs)} 个事件（来自 {os.path.basename(a.events)}）")

    # ---- 先记下改之前的重叠情况，好和去重后对比 ----
    cov_raw, _ = coverage(evs)
    span_raw = sum(e["end"] - e["start"] for e in evs)
    print(f"  原始：覆盖 {cov_raw:.2f}s / 合计 {span_raw:.2f}s"
          f"（重叠率 {(1 - cov_raw / max(span_raw, 1e-6)) * 100:.0f}%）")

    # ---- 过滤 ----
    if a.actions:
        want = {s.strip() for s in a.actions.split(",") if s.strip()}
        evs = [e for e in evs if e["action"] in want]
        print(f"  按动作筛选后 {len(evs)} 个")

    # ---- 去重 ----
    # ⚠ 必须放在 min_dur 过滤**之前**：
    #   partition 要靠**全体**事件来划分时间轴。若先按 min_dur 砍掉一批，
    #   那些时刻就变成"无人认领"的空白，划分不再完整。
    if a.dedup == "partition":
        evs = partition(evs, a.pad, src_dur)
        print(f"  去重 partition 后 {len(evs)} 段")
    elif a.dedup == "greedy":
        evs = greedy_nms(evs, a.pad, src_dur)
        print(f"  去重 greedy 后 {len(evs)} 段")

    cov_dd, _ = coverage(evs)
    span_dd = sum(e["end"] - e["start"] for e in evs)
    if a.dedup != "none":
        print(f"  去重后：覆盖 {cov_dd:.2f}s / 合计 {span_dd:.2f}s"
              f"（重叠率 {(1 - cov_dd / max(span_dd, 1e-6)) * 100:.0f}%）")

    # 太短的丢掉（动作本身都没持续够，切出来没有观感价值）
    evs = [e for e in evs if (e["end"] - e["start"]) >= a.min_dur]
    print(f"  时长 >= {a.min_dur}s 的 {len(evs)} 个")
    if a.top > 0:
        evs = sorted(evs, key=lambda e: -abs(e.get("peak", 0)))[:a.top]
        evs.sort(key=lambda e: e["start"])
        print(f"  取强度最高的 {len(evs)} 个")
    if not evs:
        print("过滤后没有可用的片段")
        return 1

    # 超长区间**拆分**而不是截断（见 split_long 注释）
    evs = [s for e in evs for s in split_long(e, a.max_dur)]
    evs.sort(key=lambda e: e["start"])

    # ---- 切片段 ----
    os.makedirs(a.out_dir, exist_ok=True)
    manifest = []
    for i, e in enumerate(evs):
        if e.get("padded"):
            # 去重模式：pad 已经在 partition/greedy 里加过了，不能再加一次
            t0, t1 = e["start"], e["end"]
        else:
            t0 = max(0.0, e["start"] - a.pad)
            t1 = e["end"] + a.pad
            if src_dur:
                t1 = min(t1, src_dur)
        dur = t1 - t0
        if dur <= 0.05:
            continue
        name = f"{i:03d}_{e['action']}_{t0:.2f}.mp4"
        sub = os.path.join(a.out_dir, e["action"]) if a.by_action else a.out_dir
        os.makedirs(sub, exist_ok=True)
        dst = os.path.join(sub, name)
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{t0:.4f}", "-i", a.video, "-t", f"{dur:.4f}",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
             "-pix_fmt", "yuv420p", "-an", dst],
            capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and os.path.exists(dst):
            manifest.append({"file": os.path.relpath(dst, a.out_dir),
                             "action": e["action"], "src_start": t0,
                             "src_end": t1, "dur": dur,
                             "peak": e.get("peak", 0),
                             "peak_t": e.get("peak_t")})
        else:
            print(f"  切失败 {name}: {(r.stderr or '')[:80]}")

    # ⚠ 最终覆盖要以**真实落地的清单**为准。
    #   前面的 cov_dd 是 min_dur 过滤**之前**的，而 min_dur 会丢掉一批短段，
    #   那些时刻就变成空洞了 —— 报高了会让人以为素材带够长。
    final_cov, _ = coverage([{"start": m["src_start"], "end": m["src_end"]}
                             for m in manifest])
    print(f"\n摘出 {len(manifest)} 个片段 -> {a.out_dir}/"
          f"（覆盖 {final_cov:.2f}s / 原片 {(src_dur or 0):.2f}s）")
    from collections import Counter
    cnt = Counter(m["action"] for m in manifest)
    for k, v in cnt.most_common():
        print(f"  {k:10} {v} 个")

    with open(os.path.join(a.out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"source": os.path.abspath(a.video),
                   "events_file": os.path.abspath(a.events),
                   "src_duration": src_dur,
                   "dedup": a.dedup,
                   "pad": a.pad,
                   # 覆盖信息一起写进去：看清单就知道这批片段能不能当
                   # "顺序素材带"用（覆盖 < 时长 就意味着有空洞）
                   "coverage": round(final_cov, 3),
                   "clips": manifest}, f, ensure_ascii=False, indent=2)
    print(f"  清单 -> {os.path.join(a.out_dir, 'manifest.json')}")

    # ---- 可选：拼一条素材带 + 映射表（供 edit_ref.py --clips 用）----
    if a.reel and manifest:
        build_reel(a.video, manifest, a.reel, a.out_dir, a.crf, evs_raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
