"""启动自检：模型 / 摄像头 / HUD / 卡点链路 / BGM 五条链路。

不进入交互窗口，纯初始化校验，可反复跑。

架构变更（2026-09-22）：
    录制阶段不再收麦克风，成片音轨固定由 BGM 提供。因此自检里
    "麦克风采集 + 说话帧判定"两项已替换为"卡点链路 + BGM 就位情况"。
    这才是当前真正会出问题的地方 —— 自检要跟着架构走，
    否则会出现"自检全绿但功能是坏的"这种最坏情况。
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "slop_templates"))
import sigma_realtime as sr   # noqa: E402
import bgm_pipeline as bp     # noqa: E402

ok = True

# ---- 1. 模型文件 ----
# 只判断"存在 + 大小合理"：空文件/半截下载会让 MediaPipe 静默降级，
# 看起来程序在跑，实际检测永远返回空 —— 这是典型的静默失败。
print("[1] MediaPipe 模型")
for m, floor in ((sr.MODEL_FACE, 1_000_000), (sr.MODEL_POSE, 1_000_000)):
    if os.path.exists(m) and os.path.getsize(m) > floor:
        print(f"  OK   {os.path.basename(m)} "
              f"({os.path.getsize(m) / 1024 / 1024:.2f} MB)")
    else:
        print(f"  缺失或过小: {m}")
        ok = False

# ---- 2. 摄像头 ----
print("\n[2] 摄像头")
try:
    cap = sr.open_camera(0)
    if cap is not None:
        good, fr = cap.read()
        if good and fr is not None:
            print(f"  OK   {fr.shape}")
        else:
            print("  WARN 打开成功但读不到帧（可能被其他程序占用）")
        cap.release()
    else:
        print("  WARN 打不开（不影响自检；录制时才会用到）")
except Exception as e:
    print(f"  WARN 摄像头异常: {e}")

# ---- 3. HUD 对 state 的容错 ----
print("\n[3] HUD 渲染")
img = np.full((720, 1280, 3), 90, np.uint8)
try:
    # state 允许缺字段（含已移除的 mic_rms）以及直接传 None
    for st in ({"fps": 15.0}, {}, None):
        sr.hud(img.copy(), "sigma", 15.0, True, True, True, "Cam 0", state=st)
    print("  OK   兼容缺字段 / None 的 state")
except Exception as e:
    print(f"  异常: {e}")
    ok = False

# ---- 4. 卡点链路（用合成 BGM，标准答案已知） ----
print("\n[4] 卡点链路（合成 BGM 回测）")
try:
    import wave
    import beat_analyze as ba
    import beat_sync as bs

    SR_HZ, BPM_TRUE = 44100, 128.0
    dur = 15.0
    n = int(dur * SR_HZ)
    x = np.random.randn(n).astype(np.float32) * 0.002
    per, t = 60.0 / BPM_TRUE, 0.0
    while t < dur:
        i = int(t * SR_HZ)
        L = int(0.05 * SR_HZ)
        if i + L < n:
            e = np.exp(-np.arange(L) / (0.010 * SR_HZ))
            x[i:i + L] += (np.sin(2 * np.pi * 55 * np.arange(L) / SR_HZ)
                           * e * 0.75).astype(np.float32)
        t += per
    pcm = np.clip(x * 32767, -32768, 32767).astype("<i2")
    tmp = os.path.join(sr.OUTDIR, "_selfcheck_bgm.wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR_HZ)
        wf.writeframes(pcm.tobytes())

    info = ba.analyze(tmp)
    plan = bs.fit_to_beats(10.0, info, n_downbeats=4)
    bpm_ok = abs(info["bpm"] - BPM_TRUE) < 2.0
    dur_ok = abs(plan["total_out"] - plan["target_duration"]) < 0.05
    print(f"  {'OK  ' if bpm_ok else 'FAIL'} BPM 识别 真实={BPM_TRUE} "
          f"测得={info['bpm']}（误差 {abs(info['bpm'] - BPM_TRUE):.2f}）")
    print(f"  {'OK  ' if dur_ok else 'FAIL'} 时长守恒 目标="
          f"{plan['target_duration']:.3f}s 实得={plan['total_out']:.3f}s")
    if not (bpm_ok and dur_ok):
        ok = False
    os.remove(tmp)
    for f in (tmp.replace(".wav", "_beats.json"),):
        if os.path.exists(f):
            os.remove(f)
except Exception as e:
    print(f"  异常: {type(e).__name__}: {e}")
    ok = False

# ---- 5. BGM 就位情况 ----
print("\n[5] 固定 BGM")
cands = []
for e in ("*.mp3", "*.m4a", "*.wav", "*.aac", "*.opus", "*.flac"):
    import glob
    cands += glob.glob(os.path.join(bp.BGM_DIR, e))
if not cands:
    print(f"  WARN 还没放 BGM。请把 gigachad theme (phonk house version) 放到:")
    print(f"         {bp.BGM_DIR}\\")
    print(f"       本机 YouTube 不可达，无法自动下载，需要手动放入。")
else:
    for c in sorted(cands):
        sz = os.path.getsize(c) / 1024
        print(f"  OK   {os.path.basename(c)}  ({sz:.0f} KB)")
    # 有 BGM 就顺手报一下 BPM，确认能被识别
    try:
        target = sorted(cands)[0]
        cache = os.path.splitext(target)[0] + "_beats.json"
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                inf = json.load(f)
            print(f"       已分析缓存: BPM={inf['bpm']} "
                  f"偏移={inf['offset']}s 强拍={len(inf['downbeats'])}")
        else:
            print(f"       尚未分析，跑一次: python bgm_pipeline.py --analyze-only")
    except Exception as e:
        print(f"       读取缓存失败（不影响）: {e}")

print()
print(">>> 结果:", "全部链路正常" if ok else "存在失败项")
sys.exit(0 if ok else 1)
