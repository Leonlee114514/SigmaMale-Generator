"""验证 edit_slop 里的「trim 多给 1 帧余量」到底需不需要。

背景：用户在他的复刻脚本里实测发现——trim 给多少帧料，fps 就圆整出多少帧，
那多给的 1 帧根本不会被丢弃，所以该 padding 是实现巧合而非通用规则。
但我在 edit_slop 里的实测记录是"不加就丢 16 帧、成片短 0.53s"。
两边结论冲突，必须按"一次只改一个变量"重验。

本脚本**只改 padding 这一个变量**，其余（素材/BGM/画布/分镜表）全部相同，
各自渲染一遍比对时长与帧数。
"""
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))
import edit_slop as es       # noqa: E402

FOOT = os.path.join(BASE, "output", "my_footage.mp4")
MAT = r"C:/Users/Lyonl/Desktop/2022年的剪辑 belike.mp4"
BGM = os.path.join(BASE, "bgm", "gigachad_phonk.aac")
OUTDIR = os.path.join(BASE, "output", "_pad")
os.makedirs(OUTDIR, exist_ok=True)

f_dur, m_dur = es.probe_dur(FOOT), es.probe_dur(MAT)
SL = int(sys.argv[1]) if len(sys.argv)>1 else 1
segs, _, _ = es.build_timeline(f_dur, m_dur, slices=SL)
total = sum(s[2] for s in segs)
ideal_frames = round(total * 30)
print(f"slices={SL}  素材 {f_dur:.2f}s / {m_dur:.2f}s   分镜 {len(segs)} 段")
print(f"理想总时长 {total:.4f}s = {ideal_frames} 帧\n")


def build(pad_frames):
    """构造滤镜链。pad_frames=0 表示不给余量，1 表示多给 1 帧。

    只改这一个变量 —— 其余逻辑与 edit_slop.build_filter 完全一致。
    """
    fps = float(es.st.SPEC["fps"])
    parts = [es.normalize_chain(0, "fw", (1080, 1920)),
             es.normalize_chain(1, "fm", (1080, 1920))]
    n_fw = sum(1 for s in segs if s[0] == 0)
    n_fm = sum(1 for s in segs if s[0] == 1)
    parts.append("[fw]split=%d%s" % (n_fw, "".join(f"[fwa{i}]" for i in range(n_fw))))
    parts.append("[fm]split=%d%s" % (n_fm, "".join(f"[fma{i}]" for i in range(n_fm))))

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
        avail = f_dur if src == 0 else m_dur
        src_len = out_dur * speed
        out_len = snapped[i + 1] - snapped[i]
        if out_len <= 1e-6:
            out_len = out_dur
        real_speed = src_len / out_len
        t0 = t_in % max(avail - src_len, 0.05) if avail > src_len else 0.0
        lab = f"s{i}"
        dur = src_len + pad_frames / fps          # ← 唯一变量
        fx = es.shot_chain(kind, rev, (1080, 1920))
        chain = f"trim=start={t0:.4f}:duration={dur:.4f}"
        if rev:
            chain += ",reverse"
        chain += f",setpts=(PTS-STARTPTS)/{real_speed:g}"
        if fx:
            chain += f",{fx}"
        chain += f",setsar=1,fps={fps:g}"
        parts.append(f"[{tag}]{chain}[{lab}]")
        labels.append(lab)

    cat = "".join(f"[{l}]" for l in labels)
    parts.append(f"{cat}concat=n={len(labels)}:v=1:a=0[cat]")
    parts.append("[cat]format=yuv420p[vout]")
    return ";".join(parts)


def frames_of(path):
    r = subprocess.run([es.FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    d = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr or "")
    if not d:
        return None, None
    sec = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
    return sec, round(sec * 30)


print(f"{'padding':>10} {'时长':>10} {'帧数':>7} {'相对理想':>10}  判定")
print("-" * 62)
for pad in (0, 1):
    out = os.path.join(OUTDIR, f"pad{pad}.mp4")
    fc = build(pad)
    p = os.path.join(OUTDIR, f"graph{pad}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"{fc};[2:a]{es.build_audio(BGM, total, {'fade':1.2,'volume':0.9})}[aout]")
    t0 = time.time()
    r = subprocess.run(
        [es.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", FOOT, "-i", MAT, "-i", BGM,
         "-filter_complex_script", p,
         "-map", "[vout]", "-map", "[aout]",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
         "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-shortest", out],
        capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"pad={pad} 渲染失败: {(r.stderr or '')[:120]}")
        continue
    sec, nf = frames_of(out)
    dev = nf - ideal_frames if nf else 0
    verdict = "完全一致" if dev == 0 else f"差 {dev:+d} 帧"
    print(f"{pad:>10} {sec:>9.3f}s {nf:>7} {dev:>+10d}  {verdict}  ({time.time()-t0:.0f}s)")

print()
print("结论：若两行帧数相同 → 那 1 帧余量**确实不需要**（用户结论正确）")
print("      若 pad=0 少帧    → edit_slop 里**确实需要**（与用户场景不同）")
