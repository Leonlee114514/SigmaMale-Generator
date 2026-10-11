"""一键给录像套上完整 slop 剪辑模板（slop_apply.py）
=====================================================
把 `sigma_realtime.py` 录出来的原始 mp4，按 slop 标准流水线加工成
"可以直接发平台"的成片。

默认流水线（顺序很重要）：
    倒放(可选) -> 曲线变速 -> 调色 -> 抖动(可选) -> RGB故障(可选) -> 竖屏
理由：
  - 倒放必须最早做，否则它要处理已经被放大/调速的帧，内存和时间翻倍。
  - 变速放在调色前，可以让调色后的画面少处理多余的帧。
  - 竖屏放最后，避免前序滤镜在大画布上跑。

用法：
    python slop_apply.py                          # 处理 output 里最新的 mp4
    python slop_apply.py 输入.mp4                  # 指定文件
    python slop_apply.py 输入.mp4 --preset sigma    # 选预设
    python slop_apply.py 输入.mp4 --reverse --rgb   # 开关单个特效
"""
import argparse
import glob
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "slop_templates"))

import slop_templates as st   # noqa: E402

OUT_DIR = os.path.join(BASE, "output")

# ---------------------------------------------------------------- 预设
# 每个预设 = 一组 (开关, 滤镜) 的有序列表
PRESETS = {
    # 最标准的"卡点变速 sigma"：快入慢放快出 + 金调色 + 竖屏
    "sigma": dict(reverse=False, ramp=True, grade="sigma", vertical=True,
                  shake=False, rgb=False, label="sigma 卡点"),
    # gigachad：黑白高对比 + 正方形 + 倒放
    "chad": dict(reverse=True, ramp=False, grade="chad", square=True,
                 shake=False, rgb=False, label="chad 黑白正方倒放"),
    # 重故障：所有效果都上（仅适合做示范，实际别这么堆）
    "chaos": dict(reverse=False, ramp=True, grade="vhs", vertical=True,
                  shake=True, rgb=True, label="全特效 chaos"),
    # 便宜的快剪：只变速 + 竖屏，不加调色
    "raw": dict(reverse=False, ramp=True, grade=None, vertical=True,
                shake=False, rgb=False, label="纯变速"),
}


def probe(path):
    """读回真实时长/分辨率/帧率，用于算变速的绝对时间锚点。"""
    r = subprocess.run([st.FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    dur, w, h, fps = 0.0, 0, 0, 30.0
    for line in r.stderr.splitlines():
        if "Duration" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            hh, mm, ss = t.split(":")
            dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
        if "Stream #" in line and "Video" in line:
            for tok in line.replace(",", " ").split():
                if "x" in tok and tok.replace("x", "").isdigit():
                    w, h = (int(v) for v in tok.split("x"))
                    break
            if "fps" in line:
                for i, tok in enumerate(line.split()):
                    if tok == "fps":
                        try:
                            fps = float(line.split()[i - 1])
                        except (ValueError, IndexError):
                            pass
    return dur, w, h, fps


def build_filter(cfg, duration, src_w, src_h):
    """按预设顺序拼出滤镜链。"""
    parts = []

    # 1) 倒放（最早）
    if cfg.get("reverse"):
        parts.append(st.reverse_only())

    # 2) 曲线变速：慢放窗口放在素材的中段（通常是"转折/高潮"）
    if cfg.get("ramp"):
        dur_after = duration  # 粗略用原始时长当锚点
        parts.append(st.speed_ramp_piecewise(
            slow_at=0.15, slow_span=0.40, fast=1.6, slow=0.35,
            duration=dur_after))

    # 3) 抖动（变速后做，否则抖动会被变速拉长/压缩）
    if cfg.get("shake"):
        parts.append(st.shake(intensity=max(6, src_w // 90), freq=0.9))

    # 4) 调色
    g = cfg.get("grade")
    if g == "sigma":
        parts.append(st.sigma_grade())
    elif g == "chad":
        parts.append(st.chad_grade())
    elif g == "vhs":
        parts.append(st.vhs_grade())

    # 5) RGB 故障（全片一次的位置由表达式控制，这里简化成整段轻微色散）
    if cfg.get("rgb"):
        parts.append(st.rgb_split(shift=max(4, src_w // 160)))

    # 6) 构图
    if cfg.get("square"):
        parts.append(st.square_canvas(1080, True))
    elif cfg.get("vertical"):
        parts.append(st.to_vertical())

    return ",".join(parts)


def process(src, preset, out_path, overrides):
    cfg = dict(PRESETS[preset])
    cfg.update(overrides)
    duration, w, h, fps = probe(src)
    if not w:
        print(f"读不出视频信息: {src}")
        return None
    print(f"输入: {os.path.basename(src)}  {w}x{h}  {duration:.2f}s  {fps:.1f}fps")
    print(f"预设: {preset}  ({cfg['label']})")

    vf = build_filter(cfg, duration, w, h)
    if not vf:
        print("没有可用的滤镜")
        return None

    cmd = [st.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-vf", vf, "-an",
           "-c:v", st.SPEC["vcodec"], "-preset", st.SPEC["preset"],
           "-crf", st.SPEC["crf"], "-pix_fmt", st.SPEC["pix_fmt"],
           "-r", str(st.SPEC["fps"]), out_path]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    el = time.time() - t0

    if r.returncode != 0 or not os.path.exists(out_path) \
            or os.path.getsize(out_path) < 1024:
        print(f"失败 (rc={r.returncode}):")
        for line in (r.stderr or "").strip().splitlines()[-5:]:
            print("   ", line[:140])
        return None

    od, ow, oh, ofps = probe(out_path)
    print(f"输出: {os.path.basename(out_path)}  {ow}x{oh}  {od:.2f}s  "
          f"{os.path.getsize(out_path)/1024:.0f} KB  耗时 {el:.1f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="给录像套 slop 剪辑模板")
    ap.add_argument("src", nargs="?", help="输入 mp4，缺省用 output 里最新的")
    ap.add_argument("--preset", default="sigma", choices=list(PRESETS))
    ap.add_argument("--out", help="输出路径")
    ap.add_argument("--reverse", action="store_true", help="加倒放")
    ap.add_argument("--no-ramp", action="store_true", help="关掉曲线变速")
    ap.add_argument("--shake", action="store_true", help="加抖动")
    ap.add_argument("--rgb", action="store_true", help="加 RGB 故障")
    ap.add_argument("--grade", default=None,
                    choices=["sigma", "chad", "vhs", "none"],
                    help="指定调色（none = 不调色）")
    a = ap.parse_args()

    src = a.src
    if not src:
        cands = sorted(glob.glob(os.path.join(OUT_DIR, "*.mp4")),
                       key=os.path.getmtime)
        if not cands:
            print(f"{OUT_DIR} 里没有 mp4")
            return 1
        src = cands[-1]
        print(f"未指定输入，用最新的: {os.path.basename(src)}\n")
    if not os.path.exists(src):
        print(f"文件不存在: {src}")
        return 1

    ov = {}
    if a.reverse:
        ov["reverse"] = True
    if a.no_ramp:
        ov["ramp"] = False
    if a.shake:
        ov["shake"] = True
    if a.rgb:
        ov["rgb"] = True
    if a.grade:
        ov["grade"] = None if a.grade == "none" else a.grade

    out = a.out or os.path.join(
        OUT_DIR, f"slop_{a.preset}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    res = process(src, a.preset, out, ov)
    if res:
        print(f"\n完成 -> {res}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
