"""BGM 结构分析：找出段落边界、Drop 位置、以及最佳剪辑切点。
====================================================================
为什么需要这个（beat_analyze.py 不够用）：
    beat_analyze.py 只给**全局** BPM + 均匀拍网格。但剪辑需要的不是
    "整首歌平均多少 BPM"，而是：
      - 主歌→副歌（build→drop）的**切换时刻**在哪
      - Drop 段落从哪开始、到哪结束
      - 哪些是重鼓点（用于 0.04s 闪白）
    这些是**结构信息**，靠均匀网格推不出来 —— phonk 的 Drop 处往往
    有明显能量跳变，必须从频谱能量的时间演化里找。

方法（纯 numpy，不依赖 librosa/scipy）：
    1. 解码成 22.05kHz 单声道
    2. STFT（n_fft=2048 / hop=256）→ 幅度谱
    3. 分频段能量：sub(30-120Hz 底鼓) / bass / mid / high(4k-11k 踩镲军鼓)
    4. 谱通量（spectral flux）作为 novelty 曲线
    5. 用拍网格把上述特征聚合到**小节**粒度（每 4 拍 = 1 小节）
    6. 对"小节级 sub 能量"做**递归二分变点检测** → 段落边界
    7. 按相对能量给段落打标签（intro/build/drop/break/outro）

自检（很重要）：
    会算出"有多少起点落在拍网格 ±50ms 内"。若这个比例低，说明 BPM 估错了，
    后面所有结论都不可信 —— 必须先看这个数再信其它输出。

用法：
    python analyze_sections.py                       # 用 bgm/ 里第一个音频
    python analyze_sections.py --bgm x.aac --zoom 18 45
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import beat_analyze as ba          # noqa: E402

SR = 22050
N_FFT = 2048
HOP = 256                          # 11.6ms
BGM_DIR = os.path.join(BASE, "bgm")

# 频段定义（Hz）——针对 phonk/EDM 的乐器分布
BANDS = {
    "sub":  (30, 120),      # 底鼓基频：Drop 的判据
    "bass": (120, 300),     # 贝斯
    "mid":  (300, 2000),    # 人声/主旋律
    "high": (4000, 11000),  # 踩镲/军鼓噪声：密度判据
}


# ==================================================================== 解码
def decode(path, sr=SR):
    """ffmpeg 解码成单声道 float32。"""
    cmd = [ba.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
           "-f", "s16le", "-acodec", "pcm_s16le",
           "-ac", "1", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"解码失败: {(r.stderr or b'')[-300:]}")
    return np.frombuffer(r.stdout, dtype="<i2").astype(np.float32) / 32768.0


# ==================================================================== STFT
def stft_mag(x, n_fft=N_FFT, hop=HOP):
    """返回 (幅度谱, 频率轴)。用 numpy 的 rfft，加 Hann 窗。"""
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + max(0, (len(x) - n_fft) // hop)
    if n <= 0:
        return np.zeros((n_fft // 2 + 1, 0), np.float32), np.zeros(0, np.float32)
    # 用 stride_tricks 做无拷贝分帧（比循环快一个量级）
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * win
    S = np.abs(np.fft.rfft(frames, axis=1)).T.astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR).astype(np.float32)
    return S, freqs


def band_energy_db(S, freqs, lo, hi):
    """某频段的能量（dB），逐帧。"""
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return np.zeros(S.shape[1], np.float32)
    e = (S[m] ** 2).mean(axis=0)
    return 10.0 * np.log10(np.maximum(e, 1e-12)).astype(np.float32)


def novelty(S):
    """谱通量：只取上升沿，求和。用于找"变化剧烈"的时刻。"""
    if S.shape[1] < 2:
        return np.zeros(S.shape[1], np.float32)
    d = np.diff(S, axis=1, prepend=S[:, :1])
    return np.maximum(d, 0).sum(axis=0).astype(np.float32)


# ==================================================================== 小节聚合
def aggregate_to_bars(feat, times, bar_times):
    """把逐帧特征按小节边界求均值。bar_times 是每小节起点（秒）。"""
    out = []
    bt = np.asarray(bar_times, dtype=float)
    for i in range(len(bt) - 1):
        m = (times >= bt[i]) & (times < bt[i + 1])
        out.append(float(feat[m].mean()) if m.any() else 0.0)
    return np.asarray(out, dtype=np.float32)


# ==================================================================== 变点检测
def binary_segment(feat, n_splits=7, min_len=3):
    """递归二分变点检测：每次找"左右均值差最大"的切点。

    比"能量过阈值"稳健：阈值法需要人为定阈值，而变点检测是**相对**的，
    在整首歌响度整体漂移时依然能找到结构转折。
    """
    bounds = [0, len(feat)]
    for _ in range(n_splits):
        best_gain, best_pos = -1.0, None
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            if b - a < 2 * min_len:
                continue
            seg = feat[a:b]
            # 候选切点：保证左右都不短于 min_len
            for p in range(a + min_len, b - min_len + 1):
                left, right = feat[a:p], feat[p:b]
                # 用"均值差 × 段长"作为增益：既看差异，也看重要性
                gain = abs(left.mean() - right.mean()) * min(p - a, b - p)
                if gain > best_gain:
                    best_gain, best_pos = gain, p
        if best_pos is None:
            break
        bounds.insert(bounds.index(best_pos) if best_pos in bounds else
                      next(i for i in range(len(bounds)) if bounds[i] > best_pos),
                      best_pos)
        bounds = sorted(set(bounds))
    return bounds


def label_section(sub_db, high_db, all_sub, all_high, is_first, is_last):
    """按相对能量给小段打标签。"""
    s = (sub_db - all_sub.mean()) / (all_sub.std() + 1e-6)
    h = (high_db - all_high.mean()) / (all_high.std() + 1e-6)
    if is_last and s < 0:
        return "outro"
    if s > 0.6 and h > 0.2:
        return "DROP/副歌"
    if s > 0.3:
        return "chorus/主歌偏强"
    if is_first and s < -0.4:
        return "intro/前奏"
    if s < -0.5:
        return "break/间奏"
    if s > 0:
        return "build/推进"
    return "verse/主歌"


# ==================================================================== 主流程
def analyze(path, zoom=None):
    x = decode(path)
    dur = len(x) / SR
    S, freqs = stft_mag(x)
    times = (np.arange(S.shape[1]) * HOP + N_FFT / 2) / SR

    be = {k: band_energy_db(S, freqs, *v) for k, v in BANDS.items()}
    nov = novelty(S)

    # ---- 拍网格 ----
    info = ba.analyze(path)
    bpm = info["bpm"]
    period = 60.0 / bpm if bpm > 0 else 0.0
    downbeats = info["downbeats"]

    print("=" * 78)
    print(f"BGM 结构分析: {os.path.basename(path)}")
    print("=" * 78)
    print(f"\n[基础] 时长 {dur:.2f}s   BPM {bpm} (置信度 "
          f"{info['bpm_confidence']})   相位 {info['offset']}s")
    print(f"       拍周期 {period*1000:.1f}ms   小节(4拍) {period*4*1000:.1f}ms   "
          f"强拍数 {len(downbeats)}")

    # ---- 自检：起点是否贴合拍网格 ----
    onsets = np.asarray(ba.detect_onsets(
        ba.onset_strength(ba.energy_envelope(x, ba.SR)[0]), HOP, ba.SR))
    # 上面这行用 beat_analyze 自己的包络，仅用于自检起点数
    if len(downbeats) > 1:
        db = np.asarray(downbeats, float)
        # 把所有起点映射到最近的强拍，看偏差分布
        ot = np.asarray(ba.detect_onsets(
            ba.onset_strength(ba.energy_envelope(x, ba.SR)[0]),
            max(1, int(ba.SR * ba.HOP_MS / 1000)), ba.SR)) * \
            (ba.SR * ba.HOP_MS / 1000) / ba.SR
        if ot.size:
            d = np.min(np.abs(ot[:, None] - db[None, :]), axis=1)
            hit = float((d < 0.05).mean())
            print(f"\n[自检] 起点贴合拍网格(±50ms)比例: {hit*100:.1f}%  "
                  f"({int((d<0.05).sum())}/{ot.size})")
            if hit < 0.6:
                print("       ⚠ 偏低 —— BPM 可能估错，后续结论需谨慎")
            else:
                print("       ✓ 拍网格可信")

    if period <= 0:
        print("\nBPM 无效，无法做小节聚合")
        return None

    # ---- 小节级聚合 ----
    bar_times = list(downbeats) + [dur]
    bar_sub = aggregate_to_bars(be["sub"], times, bar_times)
    bar_high = aggregate_to_bars(be["high"], times, bar_times)
    bar_mid = aggregate_to_bars(be["mid"], times, bar_times)
    bar_nov = aggregate_to_bars(nov, times, bar_times)

    # ---- 变点检测 ----
    bounds = binary_segment(bar_sub, n_splits=8, min_len=3)
    print(f"\n[段落结构]（以小节为单位，1 小节 = 4 拍）")
    print(f"  {'起':>7} {'止':>7} {'时长':>6} {'sub':>7} {'high':>7}  标签")
    sections = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        t0 = bar_times[min(a, len(bar_times) - 1)]
        t1 = bar_times[min(b, len(bar_times) - 1)]
        s_db = float(bar_sub[a:b].mean()) if b > a else 0.0
        h_db = float(bar_high[a:b].mean()) if b > a else 0.0
        lab = label_section(s_db, h_db, bar_sub, bar_high,
                            i == 0, i == len(bounds) - 2)
        sections.append({"start": round(t0, 2), "end": round(t1, 2),
                         "dur": round(t1 - t0, 2), "sub_db": round(s_db, 1),
                         "high_db": round(h_db, 1), "label": lab})
        print(f"  {t0:7.2f} {t1:7.2f} {t1-t0:6.2f} {s_db:7.1f} {h_db:7.1f}  {lab}")

    # ---- 找 Drop：sub 能量从低到高的最大跃升 ----
    drop_t, drop_gain = None, -1e9
    for i in range(2, len(bar_sub) - 1):
        # 前 2 小节均值 vs 后 2 小节均值
        left = bar_sub[max(0, i - 2):i].mean()
        right = bar_sub[i:min(len(bar_sub), i + 2)].mean()
        gain = right - left
        if gain > drop_gain:
            drop_gain, drop_t = gain, bar_times[i]
    print(f"\n[Drop 检测] 最大 sub 能量跃升: +{drop_gain:.1f} dB  @ {drop_t:.2f}s")

    # ---- 局部放大 ----
    if zoom:
        z0, z1 = zoom
        print(f"\n[局部放大 {z0}-{z1}s] 小节级明细")
        print(f"  {'小节起':>8} {'sub':>7} {'mid':>7} {'high':>7} {'novelty':>8}")
        for i in range(len(bar_times) - 1):
            t = bar_times[i]
            if z0 <= t < z1:
                print(f"  {t:8.2f} {bar_sub[i]:7.1f} {bar_mid[i]:7.1f} "
                      f"{bar_high[i]:7.1f} {bar_nov[i]:8.1f}")

    # ---- 重鼓点（用于 0.04s 闪白）----
    # 判据：sub 频段的上升沿最陡的那些帧
    sub_flux = np.maximum(np.diff(be["sub"], prepend=be["sub"][:1]), 0)
    if zoom:
        m = (times >= zoom[0]) & (times < zoom[1])
        st, sf = times[m], sub_flux[m]
        thr = np.percentile(sf, 92)          # 取最陡的 8% 当"重鼓点"
        picks, last = [], -1.0
        for i in np.argsort(sf)[::-1]:
            t = float(st[i])
            if sf[i] < thr:
                break
            if all(abs(t - p) > 0.12 for p in picks):   # 去重
                picks.append(t)
        picks.sort()
        print(f"\n[重鼓点 {z0}-{z1}s] 共 {len(picks)} 个（用于闪白）")
        print("  " + ", ".join(f"{p:.2f}" for p in picks[:40]))

    return {"duration": dur, "bpm": bpm, "sections": sections,
            "drop": drop_t, "bar_times": bar_times,
            "bar_sub": [round(float(v), 1) for v in bar_sub]}


def main():
    ap = argparse.ArgumentParser(description="BGM 结构分析")
    ap.add_argument("--bgm", help="音频文件，缺省用 bgm/ 里第一个")
    ap.add_argument("--zoom", nargs=2, type=float, metavar=("A", "B"),
                    help="局部放大区间（秒）")
    ap.add_argument("--json", help="把结果另存为 json")
    a = ap.parse_args()

    bgm = a.bgm
    if not bgm:
        import glob
        cands = []
        for e in ("*.mp3", "*.m4a", "*.wav", "*.aac", "*.opus", "*.flac"):
            cands += glob.glob(os.path.join(BGM_DIR, e))
        if not cands:
            print(f"{BGM_DIR} 里没有音频")
            return 1
        bgm = sorted(cands)[0]
    if not os.path.exists(bgm):
        print(f"文件不存在: {bgm}")
        return 1

    res = analyze(bgm, zoom=a.zoom)
    if res and a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"\n已保存: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
