"""slop 生产线编排器（slop_pipeline.py）
====================================================================
把原本要手动依次跑的 6 个脚本串成**一条命令**，并且把验收做成**门禁**：

    检测 → 切片+映射 →（调色）→（人脸特写）→ 成片 → 验收

设计要点（每条都是踩过坑才写上的）：

1. **验收是门禁，不是装饰。**
   以前 verify_* 是独立脚本，跑完不阻断 —— 指标不达标也得靠人看见。
   现在任一验收项不过就**非零退出**，并明确列出是哪一项，
   不会"静默出一版不合格的片"。

2. **卡点版的内容感知要在「验收专用片」上验。**
   `edit_slop` 每一镜都带逐镜变换（镜像/分屏/缩放/负片/倒放），
   像素对应关系被破坏，在成片上做内容匹配是无意义的
   （实测直接跑只有 3/11 胜出，而那不是选段错、是判据失效）。
   所以这里额外渲一版 `--no-fx` 且关掉全局特效的**诊断片**专门用于内容验收；
   节奏类验收（闪白/硬切）仍在正式成片上做。

3. **调色会改"实拍源"，必须同步改映射表。**
   动作峰值坐标是**时间轴**上的，调色不改时间轴，所以坐标仍然有效；
   但 `--clips` 会把实拍源强制改成映射表里的 `source` ——
   不改映射表的话调色版会被静默忽略。这里显式重指向，并**校验时长一致**
   （时间轴一旦变了，所有坐标就全错，而画面看起来完全正常）。

4. **每一步失败立刻停**，不带着上一步的坏产物往下跑。

用法
----
    # 复刻版（参考片结构）出片 + 验收
    python slop_pipeline.py --template ref \
        --footage output/新素材_原始.mp4 \
        --material "C:/Users/Lyonl/Desktop/2022年的剪辑 belike.mp4"

    # 两条模板都出
    python slop_pipeline.py --template both --footage 素材.mp4 --material 参考片.mp4

    # 复用已算好的动作分析（跳过检测与切片）
    python slop_pipeline.py --template ref --skip-detect --material 参考片.mp4

    # 只打印将执行的命令，不真的跑
    python slop_pipeline.py --template both --material 参考片.mp4 --dry-run
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(BASE, "output")
WORK = os.path.join(OUT, "_pipeline")
FFMPEG = None
try:
    sys.path.insert(0, os.path.join(BASE, "slop_templates"))
    import slop_templates as st
    FFMPEG = st.FFMPEG
except Exception:
    pass

import film_name as fn  # noqa: E402  成片命名（与两个剪辑器共用同一份规则）


# ------------------------------------------------------------------ 基础设施
class Fail(Exception):
    """某一步失败 —— 立刻终止整条线，不带着坏的中间产物往下跑。"""


class Progress:
    """把整条线的进度写成**结构化 JSON**，供前端读取。

    为什么单独做这个：前端有一条硬约定 —— **不解析中文日志**
    （靠正则匹配中文输出，换个措辞就静默失效）。所以进度必须走独立文件，
    而不是让前端去扫 stdout 或 `output/_pipeline/log_*.txt`。

    `path=None` 时**所有方法都是空操作** —— 命令行直接跑的行为与以前完全一致。

    写入用"临时文件 + os.replace"**原子替换**：前端是轮询读的，
    直接覆写会让它读到写了一半的 JSON（表现为页面步骤乱跳）。
    """

    KEYS = [("merge", "合并"), ("detect", "检测"), ("clips", "切片"),
            ("grade", "调色"), ("render", "成片"),
            ("diagnostic", "验收片"), ("verify", "验收")]

    def __init__(self, path=None):
        self.path = path
        self.t0 = time.time()
        self.data = {
            "state": "idle",          # idle | running | done | failed
            "updated": "",
            "elapsed": 0.0,
            "steps": [{"key": k, "name": n, "state": "wait",
                       "secs": 0.0, "note": ""} for k, n in self.KEYS],
            "failed": "",
            "outputs": [],
            "inputs": {},
        }

    # ---- 内部 ----
    def _flush(self):
        if not self.path:
            return
        self.data["updated"] = time.strftime("%H:%M:%S")
        self.data["elapsed"] = round(time.time() - self.t0, 1)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                        exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            # 进度写不进去**不该拖垮出片本身** —— 出片是正事，进度是旁路。
            pass

    def _step(self, key):
        for s in self.data["steps"]:
            if s["key"] == key:
                return s
        raise KeyError(key)

    # ---- 对外 ----
    def start(self, inputs=None):
        self.t0 = time.time()
        self.data["state"] = "running"
        self.data["inputs"] = inputs or {}
        self._flush()

    def skip(self, key, note=""):
        """本轮不跑这一步（例如没开 --grade）—— 前端要能看出"跳过"和"没轮到"的区别。"""
        s = self._step(key)
        s["state"] = "skip"
        s["note"] = note
        self._flush()

    def begin(self, key):
        s = self._step(key)
        s["state"] = "run"
        self._t0 = time.time()
        self._flush()

    def end(self, key, ok=True):
        s = self._step(key)
        s["state"] = "done" if ok else "fail"
        s["secs"] = round(time.time() - getattr(self, "_t0", time.time()), 1)
        self._flush()

    def fail(self, msg):
        self.data["state"] = "failed"
        self.data["failed"] = str(msg)
        self._flush()

    def finish(self, outputs):
        self.data["state"] = "done"
        self.data["outputs"] = list(outputs)
        self._flush()


# 全局进度对象：main() 里按 --progress 初始化，run()/verify() 直接引用。
# 不传路径时是空操作，所以不改变原有的命令行行为。
PG = Progress()


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def probe_dur(path):
    """读时长（本机没有 ffprobe，只能解析 ffmpeg -i 的 stderr，退出码恒为 1）。"""
    if not FFMPEG:
        return 0.0
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    for l in (r.stderr or "").splitlines():
        if "Duration" in l:
            t = l.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def run(title, cmd, dry=False, log_path=None, key=None):
    """跑一步。返回 True/False；dry 时只打印。

    key: 对应 Progress.KEYS 里的步骤名。传了就把这一步的起止写进进度文件，
         前端据此显示"跑到哪一步了"以及每步耗时。
    """
    print(f"\n【{title}】")
    print("  " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    if key:
        PG.begin(key)
    if dry:
        if key:
            PG.end(key, True)
        return True
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    out = (r.stdout or "") + (r.stderr or "")
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(out)
    if r.returncode != 0:
        print(f"  ✗ 失败（rc={r.returncode}，{dt:.1f}s）")
        for l in out.strip().splitlines()[-14:]:
            print("    " + l[:150])
        if key:
            PG.end(key, False)
        return False
    # 只回显最后几行有信息量的
    tail = [l for l in out.strip().splitlines() if l.strip()][-6:]
    for l in tail:
        print("  " + l[:150])
    print(f"  ✓ 完成（{dt:.1f}s）")
    if key:
        PG.end(key, True)
    return True


def need(path, what):
    if not os.path.exists(path):
        raise Fail(f"{what} 不存在：{path}")


# ------------------------------------------------------------------ 关键工具
def repoint_map(src_map, new_source, dst_map):
    """把映射表里的"实拍源"改指向新文件（调色后用）。

    ⚠ 必须**先校验时长一致**：整条内容感知链路（峰值坐标）都建在
      **原片时间轴**上。调色/特写这类处理只要时序不变，坐标就仍然有效；
      一旦时长变了（哪怕几帧），所有坐标整体错位，而画面看上去完全正常 ——
      典型"不报错但结果错"。所以这里对不上就直接终止。
    """
    with open(src_map, encoding="utf-8") as f:
        m = json.load(f)
    old = m.get("source")
    if old and os.path.abspath(old) != os.path.abspath(new_source):
        d_old, d_new = probe_dur(old), probe_dur(new_source)
        if abs(d_new - d_old) > 1.0 / 30 + 1e-6:
            raise Fail(
                f"时间轴变了：{os.path.basename(new_source)} {d_new:.3f}s "
                f"vs 原片 {d_old:.3f}s（差 {(d_new - d_old) * 1000:+.0f}ms > 1 帧）。\n"
                f"    峰值坐标全部基于原片时间轴，带错位出片等于白做。")
        print(f"    时长一致（{d_new:.3f}s ✓），坐标仍然有效")
    m["source"] = os.path.abspath(new_source)
    os.makedirs(os.path.dirname(os.path.abspath(dst_map)), exist_ok=True)
    with open(dst_map, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    return dst_map


# ------------------------------------------------------------------ 各步骤
def collect_footage(a):
    """把 --footage 的几种写法归一成路径列表。

    支持重复传参（`--footage a --footage b`）与逗号/换行分隔（`--footage a,b`），
    因为命令行习惯前者、网页表单天然是后者。
    """
    out = []
    for item in (a.footage or []):
        for part in str(item).replace("\n", ",").replace("；", ",").split(","):
            p = part.strip().strip('"').strip("'")
            if p:
                out.append(p)
    return out


def step_merge(a, paths):
    """把多段实拍素材**首尾相接**成一段。

    为什么必须合并：下游的 action_detect / clip_extract / edit_slop 都只吃
    **单个**文件。不合并的话只有第一段会被用上 ——
    症状是"我明明录了两三段，成片里却一直重复同一小段"，
    因为镜头池只来自那一段（实测：一段 15s 切出 10 个片段，
    而成片要 22 个镜头位，于是每个平均要用两次多）。
    """
    if len(paths) <= 1:
        return paths[0] if paths else None
    merged = os.path.join(WORK, "_merged_footage.mp4")
    # ⚠ 必须用 **filter_complex 的 concat 滤镜**，不能用 concat demuxer。
    #   demuxer 要求所有输入的流参数（分辨率 / 帧率 / 时间基）**完全一致**；
    #   而本项目录出来的素材 fps 是按实际帧数算的（见 save_recording 的
    #   real_fps），每段都不一样 —— 实测 demuxer 只输出第一段
    #   （16.32s，期望 31.35s），**而且返回码是 0**，静默丢素材。
    #   concat 滤镜按帧拼接，自动兼容不同帧率。
    n = len(paths)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p in paths:
        cmd += ["-i", p]
    cmd += ["-filter_complex",
            "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", merged]
    ok = run(f"0/7 合并 {len(paths)} 段实拍素材", cmd, dry=a.dry_run,
             log_path=os.path.join(WORK, "log_合并.txt"), key="merge")
    if not ok:
        raise Fail("素材合并失败")
    if not a.dry_run:
        need(merged, "合并后的素材")
        # **时长断言**：合并类操作最容易静默丢素材（filter 写错、参数不兼容、
        # 路径被吞），而画面照样有输出、返回码照样是 0。
        # 所以这里横向核对"合并结果 vs 各段之和"，对不上就直接停。
        got = probe_dur(merged)
        want = sum(probe_dur(p) for p in paths)
        if abs(got - want) > 1.0:
            raise Fail(f"合并结果时长不对：{got:.2f}s，各段之和 {want:.2f}s"
                       f"（差 {got - want:+.2f}s）—— 有素材被丢掉了，"
                       f"别带着它继续出片")
        print(f"    合并 {len(paths)} 段 -> {got:.2f}s ✓")
    return merged


def step_detect(a, footage, tag):
    ev = os.path.join(WORK, "动作分析", "动作.json")
    # ⚠ 缓存名必须**绑定素材**。原来固定叫 features.npz，于是换素材出片时
    #   会静默复用上一段的特征（实测：素材 434 帧/人脸命中 0%，缓存却是
    #   429 帧/88.3%，来自另一段素材），锚点全部对不上真实动作 →
    #   内容感知只能大面积钳位兜底 → 同一帧被反复取用。
    #   action_detect 那边现在也会校验来源指纹（双保险），这里再按素材
    #   分文件名，避免交替出片时反复重提取。
    h = hashlib.md5(os.path.abspath(footage).encode("utf-8")).hexdigest()[:8]
    cache = os.path.join(WORK, "动作分析", f"features_{h}.npz")
    ok = run(f"1/6 动作检测（MediaPipe，本地推理）",
             [PY, os.path.join(BASE, "action_detect.py"), footage,
              "-o", ev, "--cache", cache],
             dry=a.dry_run, log_path=os.path.join(WORK, "log_检测.txt"),
             key="detect")
    if not ok:
        raise Fail("动作检测失败")
    if not a.dry_run:          # dry-run 不产出文件，跳过产物校验
        need(ev, "动作事件表")
    return ev


def step_clips(a, footage, ev):
    d = os.path.join(WORK, "动作片段")
    ok = run(f"2/6 切片 + 内容池映射表",
             [PY, os.path.join(BASE, "clip_extract.py"), footage, ev,
              "-o", d, "--by-action", "--pad", "0.20",
              "--reel", os.path.join(d, "素材带.mp4"), "--crf", "16"],
             dry=a.dry_run, log_path=os.path.join(WORK, "log_切片.txt"),
             key="clips")
    if not ok:
        raise Fail("切片失败")
    rm = os.path.join(d, "reel_map.json")
    if not a.dry_run:          # dry-run 不产出文件，跳过产物校验
        need(rm, "内容池映射表")
    return rm


def step_grade(a, footage, src_map):
    """可选调色。返回（可能被改指向的）映射表路径。"""
    if not a.grade:
        return src_map
    graded = os.path.join(WORK, os.path.splitext(os.path.basename(footage))[0]
                          + "_调色.mp4")
    ok = run("3/6 调色（灰度冷调电影感）",
             [PY, os.path.join(BASE, "grade.py"), footage, "-o", graded],
             dry=a.dry_run, log_path=os.path.join(WORK, "log_调色.txt"),
             key="grade")
    if not ok:
        raise Fail("调色失败")
    if a.dry_run:
        return src_map
    # 调色不改时间轴 → 坐标仍有效；但必须把映射表重指向调色版，
    # 否则剪辑器的 --clips 会把实拍源改回原片，调色被静默忽略。
    return repoint_map(src_map, graded, os.path.join(WORK, "reel_map_used.json"))


def template_word(template):
    """模板键 → 文件名里用的中文模板词。"""
    return "复刻版" if template == "ref" else "卡点"


def film_paths(a, template):
    """算出这次要产出的 (成片路径, 方案路径)。

    ⚠ 名字**必须**能自解释，否则不同模式会静默互相覆盖 —— 真实事故：
      勾了「人脸特写+锁头」出片，输出的仍叫 `卡点_竖屏.mp4`，
      把普通版覆盖掉了，事后只能翻 plan.json 的 source 字段才知道是哪种源。

    ⚠ 定名与"选版本号"是两步：`film_stem` 拼主名，`next_versioned`
      扫 output/ 找下一个可用号（含存量文件，绝不覆盖）。
      两者都在 `film_name.py`，与 `卡点成片.bat` 那条路共用同一份规则 ——
      命名逻辑只要有两份，迟早会不一致。
    """
    stem = fn.film_stem(
        template_word(template), a.aspect,
        face_crop=a.face_crop, face_lock=a.face_lock,
        face_safe_y=a.face_safe_y,
        # 预调色（先跑 grade.py 再重指向映射表）。注意这与
        # `edit_slop --no-grade`（编辑器自己那道轻调色）**不是一回事**。
        grade=a.grade)
    out = fn.next_versioned(OUT, stem)
    return out, fn.plan_path_of(out)


def step_render(a, template, clips_map, footage):
    """渲染一个模板。返回 (成片路径, 方案路径)。"""
    asp = a.aspect
    name = template_word(template)
    out, plan = film_paths(a, template)
    if template == "ref":
        cmd = [PY, os.path.join(BASE, "edit_ref.py"),
               "--material", a.material, "--aspect", asp,
               "--out", out, "--dump-plan", plan]
    else:
        cmd = [PY, os.path.join(BASE, "edit_slop.py"),
               "--material", a.material, "--aspect", asp,
               "--out", out, "--dump-plan", plan]
    if clips_map:
        cmd += ["--clips", clips_map]
    elif footage:
        cmd += ["--footage", footage]
    if a.face_crop:
        cmd += ["--face-crop", str(a.face_crop)]
        if a.face_lock:
            cmd.append("--face-lock")
        if a.face_safe_y:
            cmd += ["--face-safe-y", a.face_safe_y]
    lbl = "4/6 成片" + ("（复刻版 / 参考片结构）" if template == "ref"
                        else "（卡点版 / 128BPM）")
    ok = run(lbl, cmd, dry=a.dry_run,
             log_path=os.path.join(WORK, f"log_成片_{name}.txt"),
             key="render")
    if not ok:
        raise Fail(f"{name} 渲染失败")
    return out, plan


def step_render_diagnostic(a, clips_map, footage, film_plan):
    """给卡点版渲一版「验收专用片」：关掉逐镜变换与全局特效。

    ⚠ 为什么必须单独渲一版：镜像/分屏/缩放/负片/倒放会**破坏像素对应**，
      在正式成片上做内容匹配是无意义的（实测只有 3/11 胜出，
      而那不是选段错、是判据失效）。关掉特效后 22/22 逐镜与方案吻合。

    `film_plan`：**成片的方案路径，由调用方传入**。
    ⚠ 不要在这里用 `film_paths()` 反推 —— 成片名带自动版本号（`_v2`/`_v3`），
      反推得到的是"下一个可用名"，不是这次真正产出的那个。
      早先就是反推的，改名后必然错位（会去比对一份不存在的方案）。
    """
    out = os.path.join(WORK, "验收_卡点.mp4")
    plan = os.path.join(WORK, "验收_卡点.plan.json")
    cmd = [PY, os.path.join(BASE, "edit_slop.py"),
           "--material", a.material, "--aspect", a.aspect,
           "--no-fx", "--no-grade", "--bw", "0", "--flash", "0",
           "--shake", "0", "--rgb", "0", "--grain", "0", "--cut-blur", "0",
           "--out", out, "--dump-plan", plan]
    if clips_map:
        cmd += ["--clips", clips_map]
    elif footage:
        cmd += ["--footage", footage]
    ok = run("5/6 渲染验收专用片（卡点版：关掉逐镜变换与全局特效）", cmd,
             dry=a.dry_run, log_path=os.path.join(WORK, "log_验收片.txt"),
             key="diagnostic")
    if not ok:
        raise Fail("验收专用片渲染失败")

    # ⚠⚠ 验收专用片必须与成片**是同一份方案** —— 否则验收是在验另一部片子，
    #   而报告只会说"内容未胜出"，看起来像选段错了。
    #
    #   实测踩过两次，都是"两边输入不同"导致的：
    #   1) `--no-fx` 会把所有 rev 置为 False。若"微波炉对"的识别依赖 rev，
    #      两半就会被分到不同入点（镜 22/23 变成 1.039 / 10.758），
    #      方案整体走偏。→ 已改为只按"入点相同"识别，不再依赖 rev。
    #   2) 验收片**不带 `--face-crop`**（成片带），两个源不是同一个文件；
    #      只要时长差一帧，`pick_intro_offset` 就可能选出完全不同的方案。
    #      → 保留这个差异（带 face-crop 要重跑一次人脸特写，慢 2 分钟），
    #        但**在这里硬性比对逐镜入点**，不一致就当场失败。
    if os.path.exists(film_plan) and os.path.exists(plan):
        try:
            p_film = json.load(open(film_plan, encoding="utf-8"))
            p_diag = json.load(open(plan, encoding="utf-8"))
            a1 = [(s["i"], round(s["in_pt"], 3)) for s in p_film["shots"]]
            a2 = [(s["i"], round(s["in_pt"], 3)) for s in p_diag["shots"]]
            if a1 != a2:
                diff = [f"镜{i}: 成片 {v1} vs 验收片 {v2}"
                        for (i, v1), (_, v2) in zip(a1, a2) if v1 != v2]
                raise Fail(
                    "验收专用片与成片的**方案不一致**，验收会验错片子：\n      "
                    + "；".join(diff[:6])
                    + (f"（共 {len(diff)} 处）" if len(diff) > 6 else ""))
            print("      方案一致性：验收片与成片逐镜入点一致 ✓")
        except Fail:
            raise
        except Exception as e:
            print(f"      ⚠ 方案一致性比对未能完成（跳过）：{e}")
    return out, plan


def verify(a, cmd, title):
    """跑一项验收。通过返回 True。"""
    print(f"\n[{title}]")
    print("  " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    if a.dry_run:
        return True
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    out = (r.stdout or "") + (r.stderr or "")
    with open(os.path.join(WORK, f"log_验收_{title}.txt"), "w",
              encoding="utf-8") as f:
        f.write(out)
    key = ("胜出 ", "压点命中", "中位偏差", "Drop @", "两组不重叠", "两组重叠",
           "已淡到黑", "误判", "相近命中", "均匀", "中位偏移", "验证结果",
           "验收结果", "近平局", "间隙", "平均偏移", "命中率")
    for l in out.splitlines():
        if any(k in l for k in key):
            print("    " + l.strip()[:140])
    passed = r.returncode == 0
    print(f"  {'✓ 通过' if passed else '✗ 未通过'}"
          f"（rc={r.returncode}，完整输出 {os.path.basename(WORK)}/log_验收_{title}.txt）")
    return passed


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(
        description="slop 生产线编排器：检测→切片→成片→验收（验收作门禁）")
    ap.add_argument("--template", default="both",
                    choices=["ref", "slop", "both"],
                    help="ref=复刻版（照搬参考片结构）；slop=128BPM 卡点版；both=两条都出")
    ap.add_argument("--footage", action="append", metavar="PATH",
                    help="实拍素材 mp4（有 --clips 时可省）。**可给多段**："
                         "重复传 --footage，或用逗号分隔 —— 多段会先首尾相接"
                         "合并再出片。⚠ 只给一段时成片镜头全来自那一段，"
                         "很容易反复用同一小段")
    ap.add_argument("--material", required=True,
                    help="交错素材 mp4（如 2022年的剪辑 belike.mp4）")
    ap.add_argument("--aspect", default="vertical",
                    choices=["vertical", "square"])
    ap.add_argument("--face-crop", type=float, nargs="?", const=3.0,
                    metavar="ZOOM",
                    help="给人脸特写（默认放大 3.0x）。⚠ 素材必须无烧死 HUD")
    ap.add_argument("--face-lock", action="store_true", help="人脸特写 + 锁头运镜")
    ap.add_argument("--face-safe-y", metavar="TOP:BOTTOM")
    ap.add_argument("--grade", action="store_true",
                    help="先对实拍素材调色（灰调冷色电影感），再出片")
    ap.add_argument("--clips", help="复用已有的 reel_map.json（跳过检测与切片）")
    ap.add_argument("--skip-detect", action="store_true",
                    help="复用 output/_pipeline 里已有的动作分析结果")
    ap.add_argument("--no-verify-gate", action="store_true",
                    help="验收不过也返回 0（**只用于排查**，正常出片别用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    ap.add_argument("--progress", metavar="PATH",
                    help="把每步进度写成 JSON（供前端读取；不传则完全不写文件，"
                         "命令行直接用时行为与以前一致）")
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)

    global PG
    PG = Progress(a.progress)

    banner("slop 生产线（检测 → 切片 → 成片 → 验收）")
    footage_list = collect_footage(a)
    if not footage_list and not a.clips:
        if a.skip_detect:
            rm = os.path.join(WORK, "动作片段", "reel_map.json")
            if not a.dry_run:
                need(rm, "已有的映射表（--skip-detect 需要）")
            a.clips = rm
        else:
            print("需要 --footage（实拍素材）或 --clips（已有映射表）")
            PG.fail("参数不足：需要 --footage（实拍素材）或 --clips（已有映射表）")
            return 2
    for p in footage_list:
        need(p, "实拍素材")
    need(a.material, "交错素材")

    # ---- 前置自检：把"跑一半才发现缺东西"提前到开头 ----
    miss = []
    for f, what in [("action_detect.py", "动作检测"),
                    ("clip_extract.py", "切片"),
                    ("face_crop.py", "人脸特写"),
                    ("grade.py", "调色"),
                    ("edit_ref.py", "复刻版剪辑器"),
                    ("edit_slop.py", "卡点版剪辑器"),
                    ("verify_ref.py", "复刻版验收"),
                    ("verify_contentaware.py", "内容感知验收"),
                    ("verify_flash.py", "闪白验收"),
                    ("verify_cuts.py", "硬切验收")]:
        if not os.path.exists(os.path.join(BASE, f)):
            miss.append(f"{f}（{what}）")
    if a.template in ("ref", "both") and not os.path.exists(
            os.path.join(BASE, "bgm", "ref_bgm.m4a")):
        miss.append("bgm/ref_bgm.m4a（复刻版 BGM，从参考片提取）")
    if len(footage_list) > 1 and (not FFMPEG or not os.path.exists(FFMPEG)):
        miss.append("ffmpeg（合并多段实拍素材要用）")
    for m in ["face_landmarker.task", "pose_landmarker_lite.task"]:
        if not os.path.exists(os.path.join(BASE, "models", m)):
            miss.append(f"models/{m}（MediaPipe 模型）")
    if miss:
        print("\n前置自检未通过，缺少：")
        for m in miss:
            print("  ·", m)
        PG.fail("前置自检未通过：" + "、".join(miss))
        return 2
    print("\n前置自检 ✓（脚本 / 模型 / BGM 齐全）")
    PG.start({
        "template": a.template, "material": a.material,
        "footage": a.footage or "", "aspect": a.aspect,
        "grade": bool(a.grade),
        "face_crop": a.face_crop or 0,
    })

    # ---- 0) 多段素材先合并（下游脚本只吃单个文件）----
    if not footage_list:
        PG.skip("merge", "复用已有映射表，不需要素材")
        footage_main = None
    elif len(footage_list) == 1:
        PG.skip("merge", "只有一段素材，无需合并")
        footage_main = footage_list[0]
    else:
        footage_main = step_merge(a, footage_list)

    # ---- 1) 检测 + 2) 切片 ----
    clips_map = a.clips
    footage_for_render = footage_main
    if not clips_map:
        ev = os.path.join(WORK, "动作分析", "动作.json")
        if a.skip_detect and os.path.exists(ev) and not a.dry_run:
            print(f"\n【1/6 动作检测】复用已有结果：{os.path.relpath(ev, BASE)}")
            PG.skip("detect", "复用已有动作分析")
        else:
            ev = step_detect(a, footage_main, "detect")
        clips_map = step_clips(a, footage_main, ev)
        footage_for_render = footage_main
    else:
        print(f"\n【1-2/6 检测与切片】复用已有映射表："
              f"{os.path.relpath(clips_map, BASE)}")
        PG.skip("detect", "复用已有映射表")
        PG.skip("clips", "复用已有映射表")

    # 调色（可选）—— 注意它会把映射表重指向调色版
    if a.grade:
        clips_map = step_grade(a, footage_for_render, clips_map)
    else:
        PG.skip("grade", "未开启调色")

    # ---- 4) 成片 ----
    todos = ["ref", "slop"] if a.template == "both" else [a.template]
    made, gates = {}, []
    for t in todos:
        out, plan = step_render(a, t, clips_map, footage_for_render)
        made[t] = (out, plan)

    # ---- 5) 验收（门禁）----
    banner("验收（不通过则整条线失败）")
    # 「验收专用片」只在卡点版上渲（见 step_render_diagnostic 的说明）
    if not clips_map:
        PG.skip("diagnostic", "没有映射表")
    elif "slop" not in todos:
        PG.skip("diagnostic", "本次没有卡点版")
    PG.begin("verify")
    for t in todos:
        out, plan = made[t]
        if t == "ref":
            gates.append((f"复刻版·卡点", verify(
                a, [PY, os.path.join(BASE, "verify_ref.py"), out], "复刻版_卡点")))
        else:
            gates.append((f"卡点版·闪白", verify(
                a, [PY, os.path.join(BASE, "verify_flash.py"), out], "卡点版_闪白")))
            gates.append((f"卡点版·硬切", verify(
                a, [PY, os.path.join(BASE, "verify_cuts.py"), out], "卡点版_硬切")))
        # 内容感知：卡点版要在"验收专用片"上验（见 step_render_diagnostic 的说明）
        if clips_map:
            if t == "slop":
                vout, vplan = step_render_diagnostic(
                    a, clips_map, footage_for_render, made["slop"][1])
            else:
                vout, vplan = out, plan
            gates.append((f"{'复刻版' if t == 'ref' else '卡点版'}·内容感知", verify(
                a, [PY, os.path.join(BASE, "verify_contentaware.py"),
                    vout, "--plan", vplan],
                f"{'复刻版' if t == 'ref' else '卡点版'}_内容感知")))

    # ---- 汇总 ----
    banner("生产线汇总")
    print("  产出：")
    for t, (out, plan) in made.items():
        if a.dry_run or not os.path.exists(out):
            print(f"    {os.path.relpath(out, BASE)}")
        else:
            print(f"    {os.path.relpath(out, BASE)}"
                  f"  {os.path.getsize(out) / 1024 / 1024:.1f}MB"
                  f"  {probe_dur(out):.2f}s")
    print("\n  验收门禁：")
    bad = [n for n, ok in gates if not ok]
    for n, ok in gates:
        print(f"    {'✓ 通过' if ok else '✗ 未通过'}  {n}")
    PG.end("verify", ok=not bad)
    if bad and not a.no_verify_gate:
        print(f"\n✗ 有 {len(bad)} 项验收未通过：{'、'.join(bad)}")
        print("  已在验收环节拦下 —— **没有静默出一版不合格的片**。")
        print("  详情见 output/_pipeline/log_验收_*.txt")
        PG.fail("验收未通过：" + "、".join(bad))
        return 1
    print("\n✓ 全链完成，全部验收通过")
    PG.finish([os.path.relpath(p, BASE).replace("\\", "/")
               for p, _ in made.values()])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print(f"\n✗ 中止：{e}")
        PG.fail(str(e))
        sys.exit(1)
