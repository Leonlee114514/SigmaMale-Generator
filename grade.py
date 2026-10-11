"""素材调色（grade.py）
====================================================================
把剪映风格的调色参数翻译成 ffmpeg 滤镜链，作用于**录制素材**。

设计定位：录制用 `record_auto.py --no-grade` 产出**干净原画**（不带 sigma 的
暖金调与人脸矩形），再由本脚本统一调色 —— 这样调色是"后期一层"，
可随时改参数重跑，而不用重录。

剪映参数 -> ffmpeg 映射
--------------------------------------------------------------------
| 剪映参数 | 值 | ffmpeg 实现 | 说明 |
|---|---|---|---|
| 亮度 | -12 | `eq=brightness=-0.12` | 剪映 -100~100 线性映射到 -1~1 |
| 对比度 | +18 | `eq=contrast=1.18` | 1 + 值/100 |
| 饱和度 | -22 | `eq=saturation=0.78` | 1 + 值/100（**Sigma 标志性灰调**） |
| 高光 | -16 | `curves` 亮部控制点 | 压亮部，白点降到 0.94 |
| 阴影 | -10 | `curves` 暗部控制点 | 压暗部，黑点保持 0（不变灰） |
| 色温 | -8 | `colorbalance` 红减蓝增 | 偏冷青；只动 midtones 最自然 |
| 锐化 | +30 | `unsharp=5:5:0.9` | 值/100 × 3.0 = 强度 |
| 暗角 | +25 | `vignette=angle=PI/4` | angle = PI × 值/100 |
| 滤镜叠加 | 暗银灰调 35% | `split` + `blend all_opacity` | 强度可调 30~40 |

⚠ 顺序按剪映的逻辑：**先基础调节，再叠滤镜**（滤镜是叠在调好的画面上的）。

⚠ 关于 `curves`：控制点必须用单引号包住（内含空格），
  写成 `curves=all='0/0 0.25/0.225 ...'`。这与 `setpts`/`eq`/`crop`
  的规则都不同 —— 本机 ffmpeg 每个滤镜的表达式规则要单独试（见 README）。

⚠ 关于 `vignette`：加 `eval=init` 让暗角遮罩只算一次。
  默认 `eval=frame` 会逐帧重算，白白浪费算力（遮罩是静止的）。

用法：
    python grade.py 素材.mp4                       # 默认参数
    python grade.py 素材.mp4 -o 调色后.mp4
    python grade.py 素材.mp4 --saturation -30      # 覆盖单项
    python grade.py 素材.mp4 --no-filter           # 不叠暗银滤镜
    python grade.py 素材.mp4 --dry-run             # 只打印滤镜链
"""
import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
OUT_DIR = os.path.join(BASE, "output")


def build_chain(a):
    """按剪映参数拼出滤镜链。返回 (基础链, 灰调滤镜链, 混合强度)。"""
    # ---- 1) 对比度 / 饱和度（eq，这两个是乘性的，映射直接）----
    #
    # ⚠⚠ **亮度不能用 eq 的 brightness** —— 这是实测踩出来的关键坑。
    #
    #   剪映的"亮度"是**乘性**的（按百分比缩放整体亮度）；
    #   而 ffmpeg `eq` 的 `brightness` 是**加性偏移**（在 0~255 上加减常量）。
    #
    #   我们直接按 `值/100` 映射（-12 → -0.12），实测把画面平均亮度
    #   从 94.7 砍到 51.5（**-43.3，即 -46%**），而用户要的只是 -12%。
    #   逐环节归因（在干净原画上单跑每个滤镜）：
    #
    #       原始(无处理)       94.7
    #       仅 eq           51.5   -43.3   ← 元凶
    #       eq+curves       47.0   -47.7
    #       +colorbalance   47.3   -47.4
    #       +unsharp        47.3   -47.5
    #       +vignette       30.6   -64.1
    #       仅 vignette     60.5   -34.2   ← 次因
    #       仅 GREY 滤镜    103.7   +9.0
    #
    #   修法：亮度改用**乘性**实现，合进下面的 `curves` 基准
    #   （等价于 `输出 = 输入 × (1 + 亮度/100)`）。
    eq_parts = []
    if a.contrast:
        eq_parts.append(f"contrast={1 + a.contrast/100:.4f}")
    if a.saturation:
        eq_parts.append(f"saturation={1 + a.saturation/100:.4f}")

    # ---- 2) 亮度（乘性）+ 高光 / 阴影（区域压缩）----
    # 合成一条 curves：基准是"整体乘 f"，再按区间叠高光/阴影系数。
    #   f  = 1 + 亮度/100            （乘性亮度）
    #   暗部(x=0.25) 受阴影主导；亮部(x=0.75~1.0) 受高光主导
    #   ⚠ 黑点 0/0 保持不动，否则画面整体发灰（"黑不下去"）
    f = 1.0 + a.brightness / 100.0
    sh = a.shadows / 100.0            # 负数 = 压暗
    hi = a.highlights / 100.0

    def pt(x, gain):
        return min(1.0, max(0.0, x * f * (1 + gain)))

    p1 = pt(0.25, sh * 0.60)                       # 暗部
    p2 = pt(0.50, (sh + hi) * 0.15)                # 中间调
    p3 = pt(0.75, hi * 0.50)                       # 亮部
    p4 = min(1.0, f * (1 + hi * 0.35))             # 白点
    curves = (f"all='0/0 {p1:.4f}/{p1:.4f} {p2:.4f}/{p2:.4f} "
              f"{p3:.4f}/{p3:.4f} 1/{p4:.4f}'")
    # ⚠ curves 的控制点是 "输入/输出" 对。上面 pt() 已算好输出，
    #   写成 `x/x` 会被误解 —— 这里显式用 `输入点/输出值`：
    curves = (f"all='0/0 0.2500/{p1:.4f} 0.5000/{p2:.4f} "
              f"0.7500/{p3:.4f} 1/{p4:.4f}'")

    # ---- 3) 色温（colorbalance，偏冷 = 红减蓝增）----
    t = a.temperature / 100.0          # 负数 = 偏冷
    cb = None
    if a.temperature:
        cb = (f"colorbalance=rm={t:.4f}:bm={-t:.4f}"
              f":rs={t*0.5:.4f}:bs={-t*0.5:.4f}")

    # ---- 4) 锐化 ----
    unsharp = None
    if a.sharpen:
        amt = a.sharpen / 100.0 * 3.0
        unsharp = f"unsharp=5:5:{amt:.2f}:5:5:0"

    # ---- 5) 暗角 ----
    # ⚠ 映射要**温和**。`vignette` 的 angle 默认是 PI/5(0.628)，
    #   一开始按 `PI × 值/100` 映射（25 → PI/4 = 0.785，**比默认还强**），
    #   实测单独就砍掉 34 亮度 —— 用户要的只是"轻压四周"。
    #   改成 `PI/12 × 值/25`（25 → 0.262），实测亮度损失降到可接受范围。
    vig = None
    if a.vignette:
        ang = min(3.14159 / 12.0 * (a.vignette / 25.0), 1.4)
        vig = f"vignette=angle={ang:.4f}:eval=init"

    base = []
    if eq_parts:
        base.append("eq=" + ":".join(eq_parts))
    base.append(f"curves={curves}")
    if cb:
        base.append(cb)
    if unsharp:
        base.append(unsharp)
    if vig:
        base.append(vig)

    # ---- 6) 暗银/灰调电影滤镜（独立一层，按强度混合）----
    # 银盐电影感：大幅降饱和 + 轻微青 + 抬黑位（胶片 fad）+ 略降对比
    grey = ("eq=saturation=0.55:contrast=0.94:brightness=0.030,"
            "colorbalance=rs=-0.030:bs=0.055:bm=0.040,"
            "curves=all='0/0.055 0.5/0.5 1/0.955'")
    return base, grey, a.filter_strength / 100.0


def build_filter(a):
    """完整 filter_complex（含灰调层的 split + blend）。"""
    base, grey, k = build_chain(a)
    if a.no_filter or k <= 0:
        return "[0:v]" + ",".join(base) + "[vout]"
    return (
        f"[0:v]{','.join(base)}[bg];"
        f"[bg]split=2[b0][b1];"
        f"[b1]{grey}[gv];"
        f"[b0][gv]blend=all_mode=normal:all_opacity={k:.3f}[vout]"
    )


def probe(path):
    """读元数据。本机无 ffprobe，走 ffmpeg -i 的 stderr；
    ⚠ ffmpeg -i 的退出码总是 1，判定只能看内容。"""
    import re
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    txt = r.stderr or ""
    dur, size, fps = 0.0, (0, 0), 0.0
    for line in txt.splitlines():
        if "Duration" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            dur = int(h) * 3600 + int(m) * 60 + float(s)
        if "Stream #" in line and "Video:" in line:
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                size = (int(m.group(1)), int(m.group(2)))
            f = re.search(r"([\d.]+)\s*fps", line)
            if f:
                fps = float(f.group(1))
    return dur, size, fps


def main():
    ap = argparse.ArgumentParser(
        description="素材调色（剪映风格参数 -> ffmpeg）")
    ap.add_argument("video")
    ap.add_argument("-o", "--out")
    ap.add_argument("--brightness", type=float, default=-12,
                    help="亮度 -100~100（默认 -12）")
    ap.add_argument("--contrast", type=float, default=18,
                    help="对比度 -100~100（默认 +18）")
    ap.add_argument("--saturation", type=float, default=-22,
                    help="饱和度 -100~100（默认 -22，Sigma 灰调）")
    ap.add_argument("--highlights", type=float, default=-16,
                    help="高光 -100~100（默认 -16）")
    ap.add_argument("--shadows", type=float, default=-10,
                    help="阴影 -100~100（默认 -10）")
    ap.add_argument("--temperature", type=float, default=-8,
                    help="色温 -100~100，负数偏冷青（默认 -8）")
    ap.add_argument("--sharpen", type=float, default=30,
                    help="锐化 0~100（默认 +30）")
    ap.add_argument("--vignette", type=float, default=25,
                    help="暗角 0~100（默认 +25）")
    ap.add_argument("--filter-strength", type=float, default=35,
                    help="暗银/灰调滤镜强度 0~100（默认 35）")
    ap.add_argument("--no-filter", action="store_true",
                    help="不叠暗银滤镜，只做基础调节")
    ap.add_argument("--crf", type=int, default=18, help="输出质量")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印滤镜链，不渲染")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        print(f"文件不存在: {a.video}")
        return 1

    dur, size, fps = probe(a.video)
    print("=" * 68)
    print("素材调色（grade.py）")
    print("=" * 68)
    print(f"输入 {os.path.basename(a.video)}  {size[0]}x{size[1]}  "
          f"{fps:.2f}fps  {dur:.2f}s")

    print(f"\n调色参数（剪映口径）:")
    print(f"  亮度 {a.brightness:+g}   对比度 {a.contrast:+g}   "
          f"饱和度 {a.saturation:+g}  色温 {a.temperature:+g}")
    print(f"  高光 {a.highlights:+g}   阴影 {a.shadows:+g}")
    print(f"  锐化 {a.sharpen:+g}   暗角 {a.vignette:+g}")
    if a.no_filter:
        print(f"  滤镜叠加: 关闭（--no-filter）")
    else:
        print(f"  滤镜叠加: 暗银/灰调，强度 {a.filter_strength:g}%")

    fc = build_filter(a)
    if a.dry_run:
        print(f"\n[filter_complex] {len(fc)} 字符")
        for part in fc.split(";"):
            print("   ", part)
        return 0

    out = a.out or os.path.join(
        OUT_DIR, os.path.splitext(os.path.basename(a.video))[0] + "_调色.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", a.video, "-filter_complex", fc, "-map", "[vout]",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
           "-pix_fmt", "yuv420p", "-an", out]

    print(f"\n渲染中...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    el = time.time() - t0

    if r.returncode != 0 or not os.path.exists(out) \
            or os.path.getsize(out) < 1024:
        print(f"失败 (rc={r.returncode}):")
        for line in (r.stderr or "").strip().splitlines()[-8:]:
            print("   ", line[:170])
        return 1

    od, osz, _ = probe(out)
    print(f"完成 -> {out}")
    print(f"  {osz[0]}x{osz[1]}  {od:.2f}s  "
          f"{os.path.getsize(out)/1024/1024:.2f}MB  耗时 {el:.1f}s")
    if abs(od - dur) > 0.2:
        print(f"  ⚠ 时长偏差: 输入 {dur:.2f}s 输出 {od:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
