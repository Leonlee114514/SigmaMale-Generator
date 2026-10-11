"""内容感知取段的独立验收（verify_contentaware.py）
====================================================================
验收「切点到底有没有落在动作上」这件事 —— 但**不能只看方案**。

为什么必须独立成一步：
    剪辑器会打印"命中动作峰值 7/7、落位偏差 0ms"，但那是**方案自己的数字**。
    方案说入点是 13.878s，不代表渲染真的取了 13.878s 那一帧 ——
    滤镜链写错、帧号算错、吸附后又被动过，都会让方案与画面脱节，
    而方案里的数字照样漂亮。

    这跟这个项目里反复出现的教训是同一条：
    **单看方案全绿 ≠ 成片合格；必须回到最终画面上量。**

本脚本做两件事：

  [1] 内容匹配（对照式）
      对每个内容感知实拍位，取成片镜中帧，与源片在几个候选时刻的画面做
      归一化互相关：
         · 内容感知入点    —— 应该最高
         · 旧「顺序游标」入点 —— 对照组：这是改造前会用的位置
         · 入点 ±1.2s      —— 对照组：邻域错位
      要求"内容感知入点"同时胜过所有对照组。
      用**对照组**而不是绝对阈值：画面经了模糊底/叠化/调色/颗粒/色差，
      绝对相关系数本来就不会是 1，只有"和别的候选比更高"才说明问题。

  [2] 黑白镜（源-成片色度比）
      绝对饱和度在这批素材上分不开（真黑白镜 7.2~9.8 与低饱和彩色镜
      11~14.7 数值相邻），因为 sigma 调色把彩色素材也压得很灰，
      而 rgb_split 色差又会给黑白画面重新加彩边。
      改成**自归一化**的判据：同一内容，成片色度 ÷ 源色度。
         · 黑白镜：底图去色了，只剩色差彩边 -> 比值明显 < 1
         · 彩色镜：比值 ≈ 1（色差特效反而会 >1）
      阈值取 1.0，理由很直接：**成片没有比源片多出颜色，就说明被去色了**。

用法：
    python verify_contentaware.py 成片.mp4 --plan 成片.plan.json
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
STRIP_W = 64          # 对比用的条纹降采样宽度
STRIP_H = 16

# "近平局"的判定阈值：高分 + 与对照组差距很小 → 视为无法判定，不算失败。
# 依据：真·选段错误会让锚点掉到 0.1~0.5 且对照组明显更高（实测 0.15 vs 0.62）；
# 而细微动作（眨眼 0.03s、抬头）相隔 1.2s 的两帧本就相似，实测 0.916 vs 0.945。
NEAR_TIE_HI = 0.90
NEAR_TIE_GAP = 0.05

# ⚠⚠ 还有第二种必须放行的情形（2026-09-25 实测踩到）：
#   锚点分数**本身已经很高**（≥ ANCHOR_FLOOR），只是对照组略微更高一点点。
#   这不是选段错的形状 —— 真错的形状是锚点塌到 0.1~0.5。
#
#   实测：卡点版镜 31（"嘟嘴"）锚点 0.906、对照 0.976（差 0.070 > 0.05），
#   被旧规则判成"未胜出"；但把三帧抽出来逐张看，成片 / 入点 / 入点−1.2s
#   **姿势几乎完全一样**（用户只是坐着看镜头，动作本来就细微），
#   差别是亚像素头位 + 压缩噪声。那 0.07 是噪声，不是选段错。
#
#   为什么要专门放行：旧规则会让**每一次跑都报红**，久了就没人看验收了
#   —— 比漏报更糟。所以锚点已经很高时，只要不是被**大幅**超过就记"近平局"。
ANCHOR_FLOOR = 0.85
NEAR_TIE_LOSS = 0.15


def probe_size(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    for l in (r.stderr or "").splitlines():
        if "Video:" in l:
            m = re.search(r"(\d{2,5})x(\d{2,5})", l)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def probe_dur(path):
    """读时长。本机没有 ffprobe，只能解析 ffmpeg -i 的 stderr（退出码恒为 1）。"""
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    for l in (r.stderr or "").splitlines():
        if "Duration" in l:
            t = l.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def grab(path, t, vf):
    """取一帧、按 vf 处理后返回 raw 像素。

    ⚠ `max(0.0, t)` 把负时间**悄悄夹到 0**。取"成片"的帧时这没问题，
      但取"源片"的帧时必须先判越界 —— 否则一个超出素材范围的候选
      （比如"入点 − 1.2s"落在负时间）会被换成 t=0 那一帧，
      拿到一个看似合法的分数，可能反而"胜出"。
      实测踩过：卡点版镜 19 的 `入点−shift` 就是靠这一手拿到 0.938。
    """
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{max(0.0, t):.4f}", "-i", path, "-frames:v", "1",
         "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True)
    return r.stdout if r.stdout else None


def ncc(a, b):
    """零均值归一化互相关。"""
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser(description="内容感知取段验收")
    ap.add_argument("video", help="成片 mp4")
    ap.add_argument("--plan", required=True, help="edit_ref.py --dump-plan 的产出")
    ap.add_argument("--shift", type=float, default=1.2,
                    help="对照组的时间偏移（秒，默认 1.2）")
    ap.add_argument("--tl-frac", type=float, default=0.5,
                    help="取样点在**镜内的相对位置**（0=镜首，1=镜尾，默认 0.5）。"
                         "⚠ 必须用比例而不是绝对秒数：卡点版每镜只有 0.469s，"
                         "绝对 0.45s 等于取在镜尾 96%% 处，成片 1 帧的量化漂移"
                         "就能把取样点挤进下一镜（完全是另一个源），"
                         "相关系数会崩到负数，被误读成「选段错了」")
    a = ap.parse_args()

    if not os.path.exists(a.video) or not os.path.exists(a.plan):
        print("成片或方案文件不存在")
        return 1
    plan = json.load(open(a.plan, encoding="utf-8"))
    out_w, out_h = probe_size(a.video)
    src_path = plan["source"]
    mat_path = plan["material"]
    src_size = probe_size(src_path)
    mat_size = probe_size(mat_path)
    print("=" * 72)
    print(f"内容感知验收: {os.path.basename(a.video)}")
    print("=" * 72)
    print(f"成片 {out_w}x{out_h}   原片 {src_size[0]}x{src_size[1]}   "
          f"素材 {mat_size[0]}x{mat_size[1]}")

    ok = True

    # ---------------------------------------------------------------- [1]
    print(f"\n[1] 内容匹配（取样点 = 镜内 {a.tl_frac:.0%} 处）")
    snaps = [r for r in (plan.get("snap") or []) if r.get("peak") is not None]
    shots_by_i = {s["i"]: s for s in plan["shots"]}

    def _pixel_comparable(seg):
        """这一镜的画面有没有被**逐镜变换**过。

        ⚠⚠ 镜像 / 分屏 / 缩放 / 倒放会彻底破坏像素对应关系。
          在这类镜上算相关系数**没有意义**，低分也**不代表选段错了**
          （实测：卡点版 edit_slop 每镜都带变换，直接跑只有 3/11 胜出，
            而那不是选段问题，是判据不适用 —— 又一次"
            把判据的失效误读成产出的错误"）。
        所以：排除它们，并且**明确报出来** ——
          既不能悄悄当成通过，也不能当成失败。
        """
        return (seg.get("kind") in (None, "plain")) and not seg.get("rev")

    skipped = [r for r in snaps
               if not _pixel_comparable(shots_by_i.get(r["shot"], {}))]
    snaps = [r for r in snaps
             if _pixel_comparable(shots_by_i.get(r["shot"], {}))]
    if not snaps:
        if not skipped:
            print("    方案里没有内容感知记录（可能没带 --clips 渲染）")
        else:
            det = {}
            for r in skipped:
                s = shots_by_i.get(r["shot"], {})
                lab = "倒放" if s.get("rev") else (s.get("kind") or "?")
                det[lab] = det.get(lab, 0) + 1
            print(f"    ⚠ 排除 {len(skipped)} 个镜"
                  f"（{'、'.join(f'{k}×{v}' for k, v in det.items())}）")
            print(f"      它们的画面被逐镜变换过，像素对应不成立；")
            print(f"      相关系数低**不代表选段错**。要全覆盖就渲一版验收专用片：")
            print(f"        edit_slop.py ... --no-fx --out output/_verify/nofx.mp4 "
                  f"--dump-plan output/_verify/nofx.plan.json")
            print(f"    可比对的镜为 0 —— **本项不适用**（不计入失败）")
    else:
        print(f"    可比对 {len(snaps)} 个内容感知实拍位"
              f"（另 {len(skipped)} 个带逐镜变换，已排除）")
        print(f"    {'镜':>4}{'内容感知':>10}{'旧顺序':>10}"
              f"{'入点+shift':>12}{'入点-shift':>12}   结论")
        wins = nears = losses = 0
        for r in snaps:
            si = r["shot"]
            seg = next(s for s in plan["shots"] if s["i"] == si)
            spath = src_path if seg["src"] == 0 else mat_path
            sw, sh = src_size if seg["src"] == 0 else mat_size
            # 成片里前景条带的纵向位置：前景按宽度铺满、上下居中
            sc = out_w / sw
            fg_h = int(round(sh * sc / 2)) * 2
            fg_top = (out_h - fg_h) // 2
            strip_o = min(fg_h, 300)
            y_o = fg_top + (fg_h - strip_o) // 2
            strip_s = int(round(strip_o / sc / 2)) * 2
            y_s = int(round((sh - strip_s) / 2))

            # ⚠⚠ 取样点必须用**镜内相对位置**，不能用绝对秒数。
            #
            #   踩过的坑（这条把"判据失效"误读成了"选段错了"）：
            #   原来固定取 `out_start + 0.45s`。复刻版每镜 0.894s，0.45 ≈ 镜内 50%，没问题；
            #   但**卡点版每镜只有 0.469s**（且 1.35x 变速），0.45s 是镜内 **96%** ——
            #   贴着镜尾。实测成片切点与方案边界差 **+25ms（≈1 帧）**，
            #   于是取样点整帧落进**下一镜**（不同源），相关系数直接变成负数，
            #   报告"3/11 胜出""0/3 胜出"，看起来像选段全错 —— 其实是量错了地方。
            #
            #   ⚠ 还有变速：输出 1s 消耗源 `speed` 秒，所以源侧偏移 = 输出偏移 × speed。
            #     倒放镜则要从窗口末端往回数。漏掉任何一项都会取到错的源帧。
            spd = float(seg.get("speed", 1.0) or 1.0)
            od = float(seg["out_dur"])
            t_rel = od * a.tl_frac
            s_rel = (od - t_rel) * spd if seg.get("rev") else t_rel * spd
            t_out = seg["out_start"] + t_rel
            fo = grab(a.video, t_out,
                      f"crop={out_w}:{strip_o}:0:{y_o},scale={STRIP_W}:{STRIP_H}")
            if fo is None:
                print(f"    {si:>4}  成片取帧失败")
                ok = False
                continue
            fo = np.frombuffer(fo, np.uint8)[:STRIP_W * STRIP_H * 3]
            fo = fo.reshape(STRIP_H, STRIP_W, 3).mean(axis=2).astype(float)

            # 位移量也在输出时间轴上，换算到源侧同样要乘 speed
            sh_s = a.shift * spd
            cases = [("内容感知", r["in_pt"] + s_rel),
                     ("旧顺序", r["planned"] + s_rel),
                     (f"入点+{a.shift:g}", r["in_pt"] + s_rel + sh_s),
                     (f"入点-{a.shift:g}", r["in_pt"] + s_rel - sh_s)]
            vals = {}
            sdur = probe_dur(spath)
            for name, t in cases:
                # ⚠ 越界的候选**不是"更差"，是"不存在"**。
                #   不判的话 grab 会把负时间夹到 0，拿 t=0 那一帧去比，
                #   可能拿到高分反超（真踩过）。直接记 nan，让它不参与胜负判定。
                if t < 0.0 or t > sdur - 0.02:
                    vals[name] = float("nan")
                    continue
                fb = grab(spath, t,
                          f"crop={sw}:{strip_s}:0:{y_s},"
                          f"scale={STRIP_W}:{STRIP_H}")
                if fb is None or len(fb) < STRIP_W * STRIP_H * 3:
                    vals[name] = float("nan")
                    continue
                fb = np.frombuffer(fb, np.uint8)[:STRIP_W * STRIP_H * 3]
                fb = fb.reshape(STRIP_H, STRIP_W, 3).mean(axis=2).astype(float)
                vals[name] = ncc(fo, fb)
            base = vals["内容感知"]
            others = {k: v for k, v in vals.items() if k != "内容感知"}
            comp = [v for v in others.values() if v == v]
            win = all(base > v for v in comp)
            # ⚠ "必须严格赢过所有对照组"对**细微动作**太苛刻：
            #   眨眼 0.03s、抬头这类动作，相隔 1.2s 的两帧本来就长得很像，
            #   实测会出现 0.916 vs 0.945 这种 **近平局**（差 0.03）。
            #   真正的选段错误不长这样 —— 它会让锚点位置掉到 0.1~0.5，
            #   而对照组明显更高（实测卡点版错取时是 0.15 vs 0.62）。
            #   所以：**高分近平局单独归类，不计入失败**，但必须打出来，
            #   不能悄悄当成通过。
            best_other = max(comp) if comp else float("nan")
            gap = base - best_other if best_other == best_other else float("nan")
            # 近平局的两种情形（都不算失败，但都要打出来）：
            #   ① 高分 + 差距极小 —— 细微动作，相隔 1.2s 的两帧本就相似
            #   ② 锚点本身已经很高，只是被小幅超过 —— 噪声级差距
            near = (not win) and (gap == gap) and (
                (base >= NEAR_TIE_HI and gap > -NEAR_TIE_GAP)
                or (base >= ANCHOR_FLOOR and gap > -NEAR_TIE_LOSS))
            wins += 1 if win else 0
            nears += 1 if near else 0
            if not win and not near:
                ok = False
                losses += 1
            verdict = ("胜出 ✓" if win else
                       ("近平局 ≈" if near else "未胜出 ✗"))
            print(f"    {si:>4}{base:>10.3f}{vals['旧顺序']:>10.3f}"
                  f"{vals[f'入点+{a.shift:g}']:>12.3f}"
                  f"{vals[f'入点-{a.shift:g}']:>12.3f}   {verdict}")
        print(f"    胜出 {wins}/{len(snaps)}"
              f"（近平局 {nears}，未胜出 {losses}）")
        print(f"    （对照式判据：画面经了模糊底/叠化/调色/颗粒/色差，")
        print(f"      绝对相关系数本来就不会是 1，只有跟别的候选比较才有意义；")
        print(f"      细微动作（眨眼/抬头）相隔 1.2s 本就相似，"
              f"{NEAR_TIE_HI:g}+ 且差距 <{NEAR_TIE_GAP:g} 记为近平局、不算失败；")
        print(f"      锚点已达 {ANCHOR_FLOOR:g}+ 时，差距 <{NEAR_TIE_LOSS:g} 同样算近平局"
              f" —— 真·选段错会让锚点塌到 0.1~0.5，不长这样）")

    # ---------------------------------------------------------------- [2]
    if not plan.get("has_bw", True):
        print("\n[2] 黑白镜 —— **本方案没有逐镜黑白，本项跳过**")
        print("    （卡点版 edit_slop 的 --bw 是**全局降饱和**，"
              "不是逐镜去色；只有复刻版 edit_ref 才有逐镜黑白序列）")
    else:
        print("\n[2] 黑白镜（源-成片色度比，两组**不重叠**即通过 —— 不用阈值）")

        def chroma(buf):
            """色度统计量：8×8 块内先求均值，再取**块间中位数**。

            ⚠ 两个设计点，都是踩出来的：

            1) 为什么要先做块平均：
               rgb_split 色差会给**黑白画面**重新引入细彩边（宽度约等于位移量）。
               直接按像素取 max-min 会把这层彩边算成"颜色"，去色后的画面
               测出来仍有可观色度。块平均能把细彩边抹掉，却保留大块的真实颜色。

            2) 为什么用**中位数**而不是均值：
               彩边只出现在**明暗边缘**所在的块里，而那些块是少数；
               均值会被这少数块拉高，形成约 8 的"彩边底噪"——
               当源片本身色度就低（约 9）时，底噪甚至超过源色度，
               于是"该是黑白"的镜比值反而 >1（实测方形版镜 10 比值 1.124，误判）。
               取块间中位数，绝大多数平坦块的色度接近 0，彩边块动不了中位。
               换完之后：黑白组 0.55~0.95 / 彩色组 1.35~2.14，间隙 +0.43。
            """
            x = np.frombuffer(buf, np.uint8)
            x = x[: (x.size // (240 * 3)) * 240 * 3]
            x = x.reshape(-1, 240, 3).astype(np.int16)
            h = (x.shape[0] // 8) * 8
            y = x[:h].reshape(h // 8, 8, 30, 8, 3).mean(axis=(1, 3))
            ch = (y.max(axis=2) - y.min(axis=2)).reshape(-1)
            return float(np.median(ch))

        def fg_box(sw, sh):
            """成片里"前景"所在的矩形（前景按宽度铺满、上下居中）。

            ⚠ 必须只比**前景区域**，不能比整帧。
              踩过的坑：一开始拿整帧比，竖屏版和方形版给出不同结论 ——
              方形画布前景正好铺满整帧，而竖屏画布上下各留 400+px 的模糊底；
              模糊底是去色的、色度近零，把整帧均值**稀释**了。
              于是同一个素材镜，方形版比值 1.124（误判彩色）、竖屏版 0.881（正确黑白）。
              构图一变判据就漂 —— 那不是画面错了，是量错了地方。
            """
            sc = out_w / sw
            fg_h = min(int(round(sh * sc / 2)) * 2, out_h)
            return (out_w - out_w, (out_h - fg_h) // 2, out_w, fg_h)

        bw_shots = {s["i"] - 9 for s in plan["shots"] if 9 <= s["i"] < 24 and s["bw"]}
        print(f"    理论黑白镜号 {sorted(bw_shots)}")

        rows = []
        for s in plan["shots"]:
            if not (9 <= s["i"] < 24):
                continue
            i = s["i"] - 9
            sw, sh = src_size if s["src"] == 0 else mat_size
            _, y_o, w_o, h_o = fg_box(sw, sh)
            # 与 [1] 用同一套取样口径（镜内相对位置 + 变速换算），
            # 免得两块判据对"镜内哪一帧"的理解不一致
            _spd = float(s.get("speed", 1.0) or 1.0)
            _od = float(s["out_dur"])
            _rel = _od * a.tl_frac
            _soff = ((_od - _rel) if s.get("rev") else _rel) * _spd
            fo = grab(a.video, s["out_start"] + _rel,
                      f"crop={w_o}:{h_o}:0:{y_o},scale=240:-2")
            fb = grab(src_path if s["src"] == 0 else mat_path,
                      s["in_pt"] + _soff, "scale=240:-2")
            if not fo or not fb:
                print(f"    {i:>4}  取帧失败")
                ok = False
                continue
            cs, co = chroma(fb), chroma(fo)
            rows.append(dict(i=i, should=i in bw_shots, cs=cs, co=co,
                             ratio=co / max(cs, 1e-6)))

        # ⚠⚠ 判据**不用任何阈值**，改成"两组是否完全不重叠"。
        #
        #   踩过的坑（同一个"阈值必须随素材走"问题的又一变体）：
        #   原先判 `成片色度/源色度 < 1.0` 即黑白。本批素材上失效 ——
        #   有两镜的**源色度只有 4.8 / 4.7**，比 `--rgb` 色差在去色画面上留下的
        #   彩边底噪（约 6）还低。于是"去过色的成片"色度反而**高于**源，
        #   比值 1.200 / 1.277 > 1.0，被判成彩色。
        #   抽帧核对：那两镜确实是灰度的 —— **渲染没错，错的是判据**。
        #
        #   规律很明确：**源色度一旦低于彩边底噪，比值必然虚高**，
        #   任何固定阈值都躲不开（试过 1.0、Otsu 二分割，都会误判）。
        #   但两组其实分得开：黑白组最大 1.277 < 彩色组最小 1.379。
        #
        #   所以只用一句话判定：**每一个黑白镜的比值，都必须低于每一个彩色镜**。
        #   判定线取"彩色组的最小值"——它不是人为常数，且这个判据恰好抓得住
        #   真正要防的回归：一旦去色滤镜失效，两组比值必然混在一起、必然重叠。
        #
        #   ⚠ 盲区（写出来，免得以后把误判当成渲染错误）：
        #     判据对"彩色组里出现一个本身就接近灰的画面"敏感 —— 它会把判定线拉低。
        #     真出现 ✗ 时**先抽帧肉眼确认**，不要直接改代码。
        bw = [r for r in rows if r["should"]]
        co = [r for r in rows if not r["should"]]
        print(f"    {'镜':>4}{'源色度':>9}{'成片色度':>11}{'比值':>8}   期望   判定")
        if bw and co:
            cut = min(r["ratio"] for r in co)
            for r in rows:
                got_bw = r["ratio"] < cut
                good = got_bw == r["should"]
                if not good:
                    ok = False
                print(f"    {r['i']:>4}{r['cs']:>9.1f}{r['co']:>11.1f}{r['ratio']:>8.3f}   "
                      f"{'黑白' if r['should'] else '彩色'}   "
                      f"{'黑白' if got_bw else '彩色'}  {'✓' if good else '✗'}")
            bmin, bmax = min(r["ratio"] for r in bw), max(r["ratio"] for r in bw)
            cmin, cmax = min(r["ratio"] for r in co), max(r["ratio"] for r in co)
            print(f"    黑白组 {bmin:.3f}~{bmax:.3f}   彩色组 {cmin:.3f}~{cmax:.3f}   "
                  f"间隙 {cmin - bmax:+.3f}")
            if cmin - bmax > 0:
                print(f"    ✓ 两组不重叠（判定线 = 彩色组最小值 {cut:.3f}，"
                      f"不含任何人为阈值）")
            else:
                over = sorted(r["i"] for r in bw if r["ratio"] >= cut)
                print(f"    ✗ 两组重叠，黑白镜 {over} 高于判定线")
                print(f"      可能是去色失效，也可能是判据被内容混淆"
                      f"（源色度低于彩边底噪时比值虚高）—— **请抽帧肉眼确认这几个镜**")
        else:
            for r in rows:
                print(f"    {r['i']:>4}{r['cs']:>9.1f}{r['co']:>11.1f}{r['ratio']:>8.3f}   "
                      f"{'黑白' if r['should'] else '彩色'}")

    # ---------------------------------------------------------------- [3]
    # 指定动作（--action-plan）：**指定的槽位必须真的拿到那个动作**。
    #
    # ⚠ 为什么要机器查：挑选逻辑在可行域里可能找不到指定的动作，
    #   此时会**退回自动**（不是报错中断）。若只靠肉眼看成片，
    #   "指定了却没落实"极容易漏掉 —— 正是本项目反复强调的那类静默降级。
    #   判据只看 plan 自报（snap.action）与实际是否一致，不重算一遍方案
    #   （重算等于验的是同一份逻辑）。
    print("\n[3] 指定动作（--action-plan）")
    ap_plan = plan.get("action_plan") or {}
    if not ap_plan:
        print("    本方案没有指定动作，本项跳过")
    else:
        print(f"    指定 {len(ap_plan)} 个槽位：" + "、".join(
            f"槽位{k}={v}" for k, v in sorted(ap_plan.items(), key=lambda kv: int(kv[0]))))
        by_slot = {}
        for r in (plan.get("snap") or []):
            if r.get("slot") is not None:
                by_slot.setdefault(r["slot"], r)
        if not by_slot:
            print("    ⚠ 方案的 snap 里没有 `slot` 字段，无法核对 ——")
            print("      plan 是不是用**旧版**编辑器产出的？（新版才有槽位概念）")
            ok = False
        else:
            print(f"    {'槽位':>4}  {'指定':<9}{'实际':<9}{'入点':>9}   判定")
            n_ok = 0
            for k, want in sorted(ap_plan.items(), key=lambda kv: int(kv[0])):
                r = by_slot.get(int(k) - 1)
                if r is None:
                    print(f"    {k:>4}  {want:<9}{'—':<9}{'—':>9}   ✗ 槽位不存在")
                    ok = False
                    continue
                got = r.get("action")
                good = (got == want) and (r.get("pin_ok") is True)
                n_ok += 1 if good else 0
                if not good:
                    ok = False
                print(f"    {k:>4}  {want:<9}{(got or '—'):<9}"
                      f"{r.get('in_pt', 0):>9.3f}   {'✓' if good else '✗ 未落实'}")
            print(f"    落实 {n_ok}/{len(ap_plan)}"
                  + (" ✓" if n_ok == len(ap_plan) else "  ✗ 有未落实项"))
            rec = plan.get("action_plan_ok")
            if rec is not None and rec != n_ok:
                print(f"    ✗ 方案自记的 action_plan_ok={rec} 与实际核对数 {n_ok} 不符"
                      f"（报告与产出一致性检查）")
                ok = False

    print("\n" + "=" * 72)
    print("验收结果:", "通过" if ok else "有未通过项")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
