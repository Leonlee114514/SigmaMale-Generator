"""端到端验证固定 BGM 卡点流水线 —— 用**已知 BPM** 的合成 BGM。

为什么要端到端实测而不是只跑单元测试：
    test_beat.py 验的是算法，test_beat_sync.py 验的是数学。
    但这两者都不保证"ffmpeg 真能把滤镜链吃下去并产出正确时长的成片"。
    实测中已经遇到过：表达式数学正确、但 ffmpeg 因逗号/减号/尺寸问题
    直接输出 0 字节 —— 单元测试全绿也没用。

本测试的判定标准（都是硬指标，不看观感）：
    1. ffmpeg 退出码 0 且产物 > 1KB（挡住静默失败）
    2. 成片时长 ≈ 目标时长（误差 < 5%）—— 验证变速的时长推导没跑偏
    3. 成片**必须有音轨**（BGM 已混入）
    4. 产出的卡点 json 里，每个锚点+半窗都要压在强拍上
    5. 换成不同 BPM 的 BGM 后，目标时长应随之变化（不是写死的）
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))

import bgm_pipeline as bp       # noqa: E402
import beat_analyze as ba       # noqa: E402
import slop_templates as st     # noqa: E402

WORK = os.path.join(BASE, "output", "_e2e")
os.makedirs(WORK, exist_ok=True)


def make_bgm(path, bpm, duration=30.0, offset=0.0, sr=44100):
    """合成 phonk 风格的 BGM：每拍一个 kick（低频衰减）+ 每 4 拍一个军鼓。

    ⚠ 这里刻意做成**有低频 kick + 高频军鼓**的复合信号，而不是纯脉冲：
      真实的 phonk 就是"kick 定拍、军鼓定强拍"，如果只用单一脉冲测，
      测不出 onset 检测在宽频谱下是否还稳。
    """
    n = int(duration * sr)
    x = np.random.randn(n).astype(np.float32) * 0.002
    period = 60.0 / bpm
    t, bi = offset, 0
    while t < duration:
        i = int(t * sr)
        L = int(0.05 * sr)
        if i + L < n:
            e = np.exp(-np.arange(L) / (0.010 * sr))
            kick = np.sin(2 * np.pi * 55 * np.arange(L) / sr) * e * 0.75
            if bi % 4 == 0:      # 强拍：叠一层军鼓（噪声 + 高次谐波）
                e2 = np.exp(-np.arange(L) / (0.005 * sr))
                snare = (np.random.randn(L) * e2 * 0.35
                         + np.sin(2 * np.pi * 2200 * np.arange(L) / sr)
                         * e2 * 0.20)
                kick = kick + snare
            x[i:i + L] += kick.astype(np.float32)
        t += period
        bi += 1
    # 写 16bit wav，ffmpeg 直接可读，无需额外编码器
    pcm = np.clip(x * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def make_video(path, duration=12.0, w=640, h=480, fps=30):
    """合成一段有运动画面的测试素材（用 ffmpeg 的 testsrc，不依赖摄像头）。"""
    cmd = [st.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate={fps}",
           "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
           "-crf", "28", "-pix_fmt", "yuv420p", path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    ok = r.returncode == 0 and os.path.exists(path) \
        and os.path.getsize(path) > 1024
    if not ok:
        print(f"  合成素材失败: {(r.stderr or '')[-200:]}")
    return path if ok else None


results = []
print("=" * 74)
print("固定 BGM 卡点流水线 —— 端到端实测")
print("=" * 74)

# ---- 准备素材 ----
print("\n[0] 准备测试素材")
vid = make_video(os.path.join(WORK, "src.mp4"), duration=12.0)
print(f"  [{'OK  ' if vid else 'FAIL'}] 合成画面素材 12.0s 640x480 "
      f"({os.path.getsize(vid)//1024 if vid else 0} KB)")
results.append(bool(vid))

# ---- 案例 1：128 BPM（phonk 典型），4 个强拍 ----
print("\n[1] 128BPM BGM + 4 强拍 -> 目标 7.5s")
bgm128 = make_bgm(os.path.join(WORK, "bgm128.wav"), bpm=128.0, offset=0.0)
info128, _ = bp.load_or_analyze_bgm(bgm128, force=True)
print(f"  测得 BPM={info128['bpm']} 偏移={info128['offset']} "
      f"置信度={info128['bpm_confidence']}")
bpm_ok = abs(info128["bpm"] - 128.0) < 2.0
print(f"  [{'OK  ' if bpm_ok else 'FAIL'}] BPM 识别（真实 128.0）")
results.append(bpm_ok)

out1 = os.path.join(WORK, "beat128.mp4")
res1, plan1 = bp.render(vid, bgm128, out1, n_downbeats=4, grade="sigma")
if res1:
    bp.write_report(res1, vid, bgm128, info128, plan1)
    od, ow, oh, _, has_a = bp.probe(res1)
    err = abs(od - plan1["target_duration"]) / plan1["target_duration"] * 100
    print(f"  [{'OK  ' if ow == 1080 and oh == 1920 else 'FAIL'}] 输出规格 "
          f"{ow}x{oh}（期望 1080x1920）")
    results.append(ow == 1080 and oh == 1920)
    print(f"  [{'OK  ' if err < 5.0 else 'FAIL'}] 时长对齐 目标="
          f"{plan1['target_duration']:.3f}s 实得={od:.3f}s 误差={err:.2f}%")
    results.append(err < 5.0)
    print(f"  [{'OK  ' if has_a else 'FAIL'}] 成片含音轨（BGM 已混入）")
    results.append(has_a)
else:
    print("  [FAIL] 渲染失败")
    results += [False, False, False]

# ---- 案例 2：锚点是否真的压在强拍上（用成片时长对应的那段 BGM） ----
print("\n[2] 卡点锚点是否压在 BGM 强拍上")
if res1 and plan1:
    dbs = np.asarray(info128["downbeats"], dtype=float)
    # 用成片时间轴上落在窗口内的强拍做比对
    n_want = len(plan1["anchors"])
    use = dbs[:n_want + 1]
    errs = []
    for a in plan1["anchors"]:
        mid = a + plan1["slow_window"] / 2.0
        errs.append(float(np.min(np.abs(use - mid))))
    mx = max(errs) if errs else 9.9
    ok = bool(errs) and mx < 0.02
    print(f"  [{'OK  ' if ok else 'FAIL'}] {len(plan1['anchors'])} 个锚点 "
          f"最大偏差 {mx*1000:.2f}ms（容差 20ms）")
    results.append(ok)

# ---- 案例 3：换 BPM 应改变目标时长（证明不是写死的） ----
print("\n[3] 换不同 BPM 的 BGM，目标时长应随之变化")
durs = []
for bpm in (100, 140):
    p = make_bgm(os.path.join(WORK, f"bgm{bpm}.wav"), bpm=float(bpm))
    inf, _ = bp.load_or_analyze_bgm(p, force=True)
    pl = __import__("beat_sync").fit_to_beats(12.0, inf, n_downbeats=4)
    got_bpm_ok = abs(inf["bpm"] - bpm) < 2.5
    durs.append((bpm, inf["bpm"], pl["target_duration"]))
    print(f"  [{'OK  ' if got_bpm_ok else 'FAIL'}] BPM 真实={bpm:3} "
          f"测得={inf['bpm']:6.1f} -> 目标时长 {pl['target_duration']:.3f}s")
    results.append(got_bpm_ok)
# 100BPM 目标应明显长于 140BPM（周期更长）
ok_ratio = durs[0][2] > durs[1][2] * 1.3
print(f"  [{'OK  ' if ok_ratio else 'FAIL'}] 100BPM 目标({durs[0][2]:.2f}s) "
      f"> 140BPM 目标({durs[1][2]:.2f}s) 且比值>1.3")
results.append(ok_ratio)

# ---- 案例 4：无 BGM 时应明确失败而不是产出无声视频 ----
print("\n[4] BGM 分析不出 BPM 时应明确拒绝")
silent = os.path.join(WORK, "silent.wav")
pcm = (np.random.randn(44100 * 5) * 0.002 * 32767).astype("<i2")
with wave.open(silent, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100)
    wf.writeframes(pcm.tobytes())
out4 = os.path.join(WORK, "should_fail.mp4")
res4, _ = bp.render(vid, silent, out4, n_downbeats=4, grade=None, verbose=False)
ok4 = res4 is None and not os.path.exists(out4)
print(f"  [{'OK  ' if ok4 else 'FAIL'}] 纯底噪 BGM -> 正确拒绝（未产出垃圾文件）")
results.append(ok4)

print()
print("=" * 74)
n_ok = sum(1 for r in results if r)
print(f"通过 {n_ok}/{len(results)}")
if n_ok < len(results):
    print("存在失败项 —— 卡点流水线还不能用")
print("=" * 74)
sys.exit(0 if n_ok == len(results) else 1)
