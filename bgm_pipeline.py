"""固定 BGM 卡点流水线：素材 + gigachad BGM -> 卡点成片（bgm_pipeline.py）
=====================================================================
与 `slop_apply.py` 的分工：
    slop_apply.py     只管**画面**（套滤镜、变速、调色、竖屏），不碰音频
    bgm_pipeline.py   只管**卡点**（分析 BGM -> 算锚点 -> 配乐 -> 对齐）

为什么必须分成两步而不是一步到底：
    画面滤镜（尤其 reverse / speed ramp）会**改变时长**，而 BGM 是固定的。
    如果一次性写完，滤镜顺序稍有变化，算锚点时依赖的"输出时间轴"就失效了，
    卡点会整体漂移 —— 而且漂移量恰好等于时长变化，肉眼极难发现。
    分成两步后，锚点计算只依赖**已知的、已量化的**变速方案，可测可证。

卡点的实现思路（关键）：
    不是"把画面变速去追音乐"，而是：
      1. 先定**目标时长** = N 个强拍（比如 8 个强拍 @128BPM = 15s）
      2. 让变速把素材压缩/拉伸到恰好这个时长（数学上保证，见 beat_sync）
      3. 慢放段的**中点**压在强拍上（不是起点，见 beat_sync 的语义说明）
      4. BGM 用 atrim 截到相同时长并淡出，保证音画同收

    于是"卡点"由构造保证，而不依赖渲染后的主观听感 —— 后者无法评审。

用法：
    python bgm_pipeline.py --video output/sigma_xxx.mp4 --bgm bgm/gigachad.mp3
    python bgm_pipeline.py --video in.mp4 --bgm bgm.mp3 --downbeats 8 --analyze-only
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))

import beat_analyze as ba          # noqa: E402
import beat_sync as bs             # noqa: E402
import slop_templates as st        # noqa: E402

OUT_DIR = os.path.join(BASE, "output")
BGM_DIR = os.path.join(BASE, "bgm")


# ---------------------------------------------------------------- 元数据
def probe(path):
    """读回真实时长/分辨率/帧率。本机无 ffprobe，统一用 ffmpeg -i 的 stderr。

    ⚠ 不要指望 ffmpeg -i 的退出码：读元数据时它**总是返回 1**（因为没给
      输出文件），所以判定必须看**是否解析到内容**，而不是 returncode。
      这类"退出码不可靠"的地方最容易写出静默失败。
    """
    r = subprocess.run([st.FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    dur, w, h, fps, has_audio = 0.0, 0, 0, 30.0, False
    for line in (r.stderr or "").splitlines():
        if "Duration" in line and dur == 0.0:
            t = line.split("Duration:")[1].split(",")[0].strip()
            hh, mm, ss = t.split(":")
            dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
        if "Stream #" in line:
            if "Video" in line and not w:
                # 形如 "1280x720 [SAR 1:1 ...]" —— 用正则在 ',' 切开的段里找
                for tok in line.replace(",", " ").split():
                    core = tok.split("[")[0]
                    if "x" in core and core.replace("x", "").isdigit():
                        w, h = (int(v) for v in core.split("x"))
                        break
                for i, tok in enumerate(line.split()):
                    if tok == "fps":
                        try:
                            fps = float(line.split()[i - 1])
                        except (ValueError, IndexError):
                            pass
            if "Audio" in line:
                has_audio = True
    return dur, w, h, fps, has_audio


def load_or_analyze_bgm(bgm_path, force=False):
    """BGM 只分析一次，结果缓存成 json（分析是 CPU 密集的）。

    缓存失效条件：json 比音频文件旧。这个判断比"文件是否存在"可靠 ——
    换了一首 BGM 但文件名相同（覆盖）时，只看存在性会拿到旧数据。
    """
    stem = os.path.splitext(bgm_path)[0]
    cache = stem + "_beats.json"
    if not force and os.path.exists(cache) \
            and os.path.getmtime(cache) >= os.path.getmtime(bgm_path):
        with open(cache, encoding="utf-8") as f:
            info = json.load(f)
        print(f"  用缓存: {os.path.basename(cache)}")
        return info, cache
    print(f"  分析 BGM（首次较慢）: {os.path.basename(bgm_path)}")
    t0 = time.time()
    info = ba.analyze(bgm_path)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"  分析完成 {time.time() - t0:.1f}s -> {os.path.basename(cache)}")
    return info, cache


# ---------------------------------------------------------------- 滤镜拼装
def build_beat_filter(plan, grade=None, vertical=True, motion_interp=False):
    """把 beat_sync 的 setpts 表达式接进画面滤镜链。

    ⚠ 顺序是有讲究的（和 slop_apply.py 保持一致的理由）：
      变速必须在调色/构图**之前**，否则会在大画布上处理多余帧。
      光流插帧（minterpolate）必须**紧接** setpts 之后，且非常昂贵，
      默认关闭 —— 只在慢放段很短时才值得开。
    """
    parts = [plan["expr"]]                       # 1) 卡点变速
    if motion_interp:
        parts.append(st.with_motion_interpolation())   # 2) 慢放丝滑（可选）
    if grade == "sigma":
        parts.append(st.sigma_grade())           # 3) 调色
    elif grade == "chad":
        parts.append(st.chad_grade())
    elif grade == "vhs":
        parts.append(st.vhs_grade())
    if vertical:
        parts.append(st.to_vertical())           # 4) 竖屏
    return ",".join(parts)


def build_audio_filter(video_duration, beat_info, fade=1.0):
    """BGM 侧：截到目标时长 + 淡出 + 补齐，保证与画面同收。

    复用 beat_sync.fit_audio_to_video（它已把截取点对齐到强拍边界）。
    """
    return bs.fit_audio_to_video(video_duration, beat_info, fade=fade)


# ---------------------------------------------------------------- 主流程
def render(video, bgm, out_path, n_downbeats=4, grade="sigma",
           vertical=True, slow_factor=0.45, fast_factor=2.2,
           fade=1.0, volume=0.85, motion_interp=False, verbose=True):
    """卡点渲染主函数。返回 (成片路径 or None, 卡点方案 dict or None)。"""
    vdur, vw, vh, vfps, v_has_audio = probe(video)
    if not vw:
        print(f"读不出视频信息: {video}")
        return None, None
    if v_has_audio:
        print(f"  ⚠ 素材自带音轨，将被 BGM 替换（卡点与素材原声本就无关）")

    beat_info, _ = load_or_analyze_bgm(bgm)
    bpm = beat_info.get("bpm", 0.0)
    if bpm <= 0:
        print(f"  BGM 分析不出 BPM，无法卡点：{bgm}")
        return None, None

    src_dur = min(vdur, 600.0)   # 夹一道上限，防止异常长素材吃满内存
    plan = bs.fit_to_beats(src_dur, beat_info, n_downbeats=n_downbeats,
                           slow_factor=slow_factor, fast_factor=fast_factor)

    if verbose:
        bs.summarize(plan, beat_info)
        print(f"  素材需用 {plan['src_needed']:.2f}s / 实有 {src_dur:.2f}s")
        if plan["src_needed"] > src_dur + 0.05:
            print(f"  ⚠ 素材不够长，末尾会定格（ffmpeg 默认行为）")

    vf = build_beat_filter(plan, grade=grade, vertical=vertical,
                           motion_interp=motion_interp)
    target = plan["target_duration"]
    af = build_audio_filter(target, beat_info, fade=fade)

    # 音量：slop 规范里 BGM 只占 15-25%，但那是"有人声解说"的情况。
    # 这里成片没有人声口播（BGM 就是主角），所以用 0.85 更合适。
    af = f"volume={volume:g}," + af

    cmd = [st.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", video, "-i", bgm,
           "-filter_complex", f"[0:v]{vf}[v];[1:a]{af}[a]",
           "-map", "[v]", "-map", "[a]",
           "-c:v", st.SPEC["vcodec"], "-preset", st.SPEC["preset"],
           "-crf", st.SPEC["crf"], "-pix_fmt", st.SPEC["pix_fmt"],
           "-r", str(st.SPEC["fps"]),
           "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
           "-shortest", out_path]

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    el = time.time() - t0

    ok = r.returncode == 0 and os.path.exists(out_path) \
        and os.path.getsize(out_path) > 1024
    if not ok:
        print(f"  失败 (rc={r.returncode}):")
        for line in (r.stderr or "").strip().splitlines()[-6:]:
            print("    ", line[:150])
        return None, plan

    od, ow, oh, _, o_has_audio = probe(out_path)
    print(f"  成片 {os.path.basename(out_path)}  {ow}x{oh}  {od:.2f}s  "
          f"音轨={'有' if o_has_audio else '无'}  "
          f"{os.path.getsize(out_path) / 1024:.0f} KB  耗时 {el:.1f}s")
    # 自检：成片时长必须接近目标，音轨必须存在。这两条任一不满足都说明
    # 滤镜表达式的时长推导错了（静默失败的重灾区）。
    if abs(od - target) > max(0.35, target * 0.04):
        print(f"  ⚠ 时长偏差过大: 实得 {od:.2f}s vs 目标 {target:.2f}s")
    if not o_has_audio:
        print("  ⚠ 成片没有音轨 —— 检查 BGM 文件是否可解码")
    return out_path, plan


def write_report(out_path, video, bgm, beat_info, plan):
    """把卡点方案落成 json，方便核对"第几秒该慢"（可复算、可评审）。"""
    rep = {
        "source_video": os.path.basename(video),
        "bgm": os.path.basename(bgm),
        "bpm": beat_info.get("bpm"),
        "bgm_offset": beat_info.get("offset"),
        "target_duration": plan["target_duration"],
        "slow_window": plan["slow_window"],
        "anchors": plan["anchors"],
        "downbeats_used": beat_info.get("downbeats", [])[:len(plan["anchors"]) + 1],
        "output": os.path.basename(out_path),
    }
    rp = os.path.splitext(out_path)[0] + "_beat.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"  卡点方案 -> {os.path.basename(rp)}")
    return rp


def main():
    ap = argparse.ArgumentParser(description="固定 BGM 卡点流水线")
    ap.add_argument("--video", help="输入 mp4，缺省用 output 里最新的")
    ap.add_argument("--bgm", help="BGM 文件，缺省用 bgm/ 里第一个音频")
    ap.add_argument("--out", help="输出路径")
    ap.add_argument("--downbeats", type=int, default=4,
                    help="成片长度用几个强拍（4 强拍≈7.5s@128BPM）")
    ap.add_argument("--grade", default="sigma",
                    choices=["sigma", "chad", "vhs", "none"])
    ap.add_argument("--no-vertical", action="store_true", help="不转竖屏")
    ap.add_argument("--slow", type=float, default=0.45, help="慢放倍率")
    ap.add_argument("--fast", type=float, default=2.2, help="快段倍率")
    ap.add_argument("--volume", type=float, default=0.85, help="BGM 音量")
    ap.add_argument("--interp", action="store_true", help="慢放光流插帧（很慢）")
    ap.add_argument("--analyze-only", action="store_true",
                    help="只分析 BGM 并打印，不渲染")
    a = ap.parse_args()

    # ---- 选 BGM（放在最前：--analyze-only 只需要 BGM，不该要求有素材）----
    # ⚠ 早期版本先选视频再选 BGM，导致 `--analyze-only` 在还没录素材时
    #   直接报"没有可用 mp4"退出 —— 而"只想看看这首 BGM 的 BPM"本来
    #   就不需要任何素材。顺序错了会让一个独立功能被无关的前置条件卡住。
    bgm = a.bgm
    if not bgm:
        os.makedirs(BGM_DIR, exist_ok=True)
        exts = ("*.mp3", "*.m4a", "*.wav", "*.aac", "*.opus", "*.flac", "*.mp4")
        cands = []
        for e in exts:
            cands += glob.glob(os.path.join(BGM_DIR, e))
        if not cands:
            print(f"还没放 BGM。请把 gigachad theme (phonk house version) "
                  f"放到:\n  {BGM_DIR}\\")
            return 1
        bgm = sorted(cands)[0]
        print(f"未指定 BGM，用: {os.path.basename(bgm)}")
    if not os.path.exists(bgm):
        print(f"BGM 不存在: {bgm}")
        return 1

    if a.analyze_only:
        info, cache = load_or_analyze_bgm(bgm, force=True)
        print(f"\n  文件     {info['file']}")
        print(f"  时长     {info['duration']}s")
        print(f"  BPM      {info['bpm']}  (置信度 {info['bpm_confidence']})")
        print(f"  相位偏移 {info['offset']}s")
        print(f"  起点数   {info['n_onsets']}")
        print(f"  拍点总数 {len(info['beats'])}")
        print(f"  强拍总数 {len(info['downbeats'])}")
        print(f"  前 8 强拍 {info['downbeats'][:8]}")
        print(f"\n  缓存 -> {cache}")
        print(f"  下一步：python bgm_pipeline.py --video 素材.mp4 --start 22 --end 41")
        return 0

    # ---- 选输入素材 ----
    video = a.video
    if not video:
        cands = [p for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.mp4")),
                                   key=os.path.getmtime)
                 if not os.path.basename(p).startswith("slop_")
                 and not os.path.basename(p).startswith("beat_")]
        if not cands:
            print(f"{OUT_DIR} 里没有可用 mp4（先录制一段素材）")
            return 1
        video = cands[-1]
        print(f"未指定素材，用最新的: {os.path.basename(video)}")
    if not os.path.exists(video):
        print(f"素材不存在: {video}")
        return 1

    out = a.out or os.path.join(
        OUT_DIR, f"beat_{a.grade}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    print(f"\n卡点渲染")
    print(f"  素材 {os.path.basename(video)}")
    print(f"  BGM  {os.path.basename(bgm)}")
    res, plan = render(video, bgm, out, n_downbeats=a.downbeats,
                       grade=None if a.grade == "none" else a.grade,
                       vertical=not a.no_vertical, slow_factor=a.slow,
                       fast_factor=a.fast, volume=a.volume,
                       motion_interp=a.interp)
    if not res:
        return 1
    bi, _ = load_or_analyze_bgm(bgm)
    write_report(res, video, bgm, bi, plan)
    print(f"\n完成 -> {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
