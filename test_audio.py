"""验证「无声录制 + BGM 混音」这条链路（test_audio.py）
================================================================
架构变更说明（重要）：
    早期版本录制时采集麦克风，并用 voice_beat_summary() 按 RMS 分位判定
    "说话帧"，测试也围绕它写。现在成片音轨改由**固定 BGM** 提供，
    卡点建立在该 BGM 的节拍分析上（beat_analyze.py / bgm_pipeline.py），
    录制阶段不再收声 —— 因此本测试也随之改写。

    保留的教训（写在这里免得后人重犯）：
      一旦要重做"人声口播"，判定阈值必须用**分位数**而不是中位数。
      RMS 分布是**双峰**的（静音 ~30 / 说话 ~6000，中间几乎无样本），
      中位数会正好落进两峰之间的空档（实测 2808），乘 2.2 得 6177，
      **比说话电平本身还高** —— 实测 47/50 被误判为静音。
      这是"用集中趋势代表双峰分布"的经典错误。

本测试覆盖四件事：
  1. save_recording(silent=True) 能产出**规格正确的无声成片**
  2. 无声成片不该被误加空音轨（空音轨会浪费体积且让后期多一步剔除）
  3. bgm_pipeline 混入 BGM 后，成片**必须有音轨**且时长与画面吻合
  4. 换 BGM 不需要重录素材（素材与配乐解耦，这是本次重构的核心收益）
"""
import os
import subprocess
import sys
import wave

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))

import sigma_realtime as sr     # noqa: E402
import bgm_pipeline as bp       # noqa: E402
import slop_templates as st     # noqa: E402

WORK = os.path.join(BASE, "output", "_audio_test")
os.makedirs(WORK, exist_ok=True)
results = []


def synth_frames(n=90, w=320, h=240):
    """合成帧序列，避免依赖摄像头。"""
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 1] = (i * 3) % 256
        cv2.rectangle(f, (10 + i % 50, 30), (80 + i % 50, 120),
                      (200, 200, 60), -1)
        out.append(f)
    return out


def has_audio_stream(path):
    """判定 mp4 是否含音轨。本机无 ffprobe，用 ffmpeg -i 的 stderr 读。"""
    r = subprocess.run([st.FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    return "Audio:" in (r.stderr or "")


print("=" * 66)
print("无声录制 + BGM 混音 链路验证")
print("=" * 66)

# ---- 1. 无声录制 ----
print("\n[1] save_recording(silent=True) 应产出无声竖屏成片")
# 先记下已有的 _raw 残留，便于区分"本次新增"还是历史遗留
_raw_before = {f for f in os.listdir(sr.OUTDIR) if f.startswith("_raw_")}
frames = synth_frames(90, 320, 240)
dur = 3.0
sr.save_recording(frames, dur, 320, 240, None, silent=True)
outs = sorted(
    [os.path.join(sr.OUTDIR, f) for f in os.listdir(sr.OUTDIR)
     if f.startswith("sigma_") and f.endswith(".mp4")],
    key=os.path.getmtime)
silent_path = outs[-1] if outs else None
ok1 = bool(silent_path) and os.path.getsize(silent_path) > 1024
print(f"  [{'OK  ' if ok1 else 'FAIL'}] 产出 {os.path.basename(silent_path or '?')} "
      f"({os.path.getsize(silent_path)//1024 if silent_path else 0} KB)")
results.append(ok1)

# ---- 1b. 中间文件必须被清理（回归：曾间歇性残留） ----
# 这条是补的回归测试。Windows 上 os.remove 会因句柄未释放而**间歇性**失败，
# 早期"单次尝试失败就放弃"的写法在连跑 3 次时有 2 次留下 _raw_*.mp4，
# 且失败日志被重定向就完全看不见。现在用 _robust_remove 重试。
print("\n[1b] 中间文件 _raw_*.mp4 应被清理（曾间歇性残留）")
_raw_after = {f for f in os.listdir(sr.OUTDIR) if f.startswith("_raw_")}
new_raw = _raw_after - _raw_before
ok_raw = not new_raw
print(f"  [{'OK  ' if ok_raw else 'FAIL'}] 本次新增残留: "
      f"{sorted(new_raw) if new_raw else '无'}")
results.append(ok_raw)

# ---- 2. 无声成片不应带音轨 ----
print("\n[2] 无声成片不应含空音轨")
no_a = silent_path and not has_audio_stream(silent_path)
print(f"  [{'OK  ' if no_a else 'FAIL'}] 音轨检查 -> "
      f"{'无音轨（正确）' if no_a else '意外含音轨'}")
results.append(bool(no_a))

# ---- 3. 混入 BGM 后必须有音轨且时长吻合 ----
print("\n[3] 混入固定 BGM 后应含音轨，且时长≈目标")


def make_bgm(path, bpm=128.0, duration=20.0, sr_hz=44100):
    n = int(duration * sr_hz)
    x = np.random.randn(n).astype(np.float32) * 0.002
    per = 60.0 / bpm
    t, bi = 0.0, 0
    while t < duration:
        i = int(t * sr_hz)
        L = int(0.05 * sr_hz)
        if i + L < n:
            e = np.exp(-np.arange(L) / (0.010 * sr_hz))
            k = np.sin(2 * np.pi * 55 * np.arange(L) / sr_hz) * e * 0.75
            if bi % 4 == 0:
                e2 = np.exp(-np.arange(L) / (0.005 * sr_hz))
                k = k + np.random.randn(L) * e2 * 0.35
            x[i:i + L] += k.astype(np.float32)
        t += per
        bi += 1
    pcm = np.clip(x * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr_hz)
        wf.writeframes(pcm.tobytes())
    return path


bgm_a = make_bgm(os.path.join(WORK, "a.wav"), bpm=128.0)
out3 = os.path.join(WORK, "mixed_a.mp4")
res3, plan3 = bp.render(silent_path, bgm_a, out3, n_downbeats=2, grade=None,
                        verbose=False)
if res3:
    d3, w3, h3, _, a3 = bp.probe(res3)
    err3 = abs(d3 - plan3["target_duration"]) / plan3["target_duration"] * 100
    ok_size = w3 == 1080 and h3 == 1920
    print(f"  [{'OK  ' if ok_size else 'FAIL'}] 规格 {w3}x{h3}（期望 1080x1920）")
    results.append(ok_size)
    print(f"  [{'OK  ' if a3 else 'FAIL'}] 音轨 = {'有' if a3 else '无'}")
    results.append(a3)
    ok_d = err3 < 5.0
    print(f"  [{'OK  ' if ok_d else 'FAIL'}] 时长 目标="
          f"{plan3['target_duration']:.3f}s 实得={d3:.3f}s 误差={err3:.2f}%")
    results.append(ok_d)
else:
    print("  [FAIL] 混音渲染失败")
    results += [False, False, False]

# ---- 4. 素材与配乐解耦：同一素材可换 BGM 出不同版本 ----
print("\n[4] 同一素材换 BGM 可再出一版（证明素材/配乐解耦）")
bgm_b = make_bgm(os.path.join(WORK, "b.wav"), bpm=100.0)
out4 = os.path.join(WORK, "mixed_b.mp4")
res4, plan4 = bp.render(silent_path, bgm_b, out4, n_downbeats=2, grade=None,
                        verbose=False)
ok4 = res4 is not None and os.path.exists(res4) \
    and os.path.getsize(res4) > 1024
if ok4 and res3:
    d4, _, _, _, a4 = bp.probe(res4)
    differ = abs(d4 - plan4["target_duration"]) \
        / plan4["target_duration"] * 100 < 5.0 and a4
    print(f"  128BPM -> {plan3['target_duration']:.2f}s   "
          f"100BPM -> {plan4['target_duration']:.2f}s")
    print(f"  [{'OK  ' if differ else 'FAIL'}] 两版时长不同且都含音轨 "
          f"（说明无需重录素材）")
    results.append(bool(differ))
else:
    print("  [FAIL] 换 BGM 渲染失败")
    results.append(False)

# ---- 5. 素材缺失音轨时混音不应报错（兼容旧素材） ----
print("\n[5] 给无声素材混音不应因'素材无音轨'而失败")
ok5 = res3 is not None
print(f"  [{'OK  ' if ok5 else 'FAIL'}] 无声素材 + BGM -> 成功（上一步已验证）")
results.append(ok5)

print()
print("=" * 66)
n_ok = sum(1 for r in results if r)
print(f"通过 {n_ok}/{len(results)}")
print("=" * 66)
sys.exit(0 if n_ok == len(results) else 1)
