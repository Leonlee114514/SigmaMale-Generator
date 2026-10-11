"""节拍分析：对**固定 BGM** 做一次性离线分析，产出节拍网格。

设计前提（与"用麦克风实时判定"的关键区别）：
    BGM 是固定不变的，所以**只需要分析一次**，结果可以缓存成 json。
    这带来三个重要好处：
      1. 不必在录制/导出时重复分析，导出速度不受影响
      2. 可以人工校正（分析结果是人可读的 json，能改）
      3. 结果是**绝对时间点**（秒），可以直接拿去对齐剪辑锚点

产出结构：
    {
      "file": "gigachad_phonk.mp3",
      "duration": 142.5,
      "bpm": 128.0,
      "offset": 0.234,        # 第一个强拍相对 0 秒的偏移
      "beats": [0.234, 0.703, ...],       # 全部拍点（秒）
      "downbeats": [0.234, 2.579, ...],   # 每 4 拍取 1（强拍）
      "energy": [ ... ]       # 每 10ms 一格的能量包络，供画图/调试
    }

方法（不依赖任何第三方音频分析库，只用 numpy）：
    1. 解码成 16kHz 单声道 PCM
    2. 算短时能量包络（10ms 帧 / 5ms 跳）
    3. 对包络做**频谱通量**式的起点检测：能量的一阶正差分 + 自适应阈值
    4. 对起点序列做**自相关**求 BPM（在 60-200 BPM 范围内搜）
    5. 用 BPM 反推拍点网格，并用起点做相位对齐（求 offset）
"""
import json
import os
import subprocess

import numpy as np

FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
FFPROBE = FFMPEG  # 本机无独立 ffprobe，用 ffmpeg -i 读元数据

SR = 16000          # 分析用采样率，够抓鼓点低频
HOP_MS = 5          # 包络跳距
WIN_MS = 20         # 包络窗长
BPM_MIN, BPM_MAX = 60.0, 200.0


def decode_pcm(path, sr=SR):
    """把音频解码成 float32 单声道 numpy 数组（范围 -1~1）。"""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
           "-f", "s16le", "-acodec", "pcm_s16le",
           "-ac", "1", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"解码失败: {(r.stderr or b'')[-300:]}")
    a = np.frombuffer(r.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return a


def energy_envelope(x, sr=SR, hop_ms=HOP_MS, win_ms=WIN_MS):
    """短时 RMS 能量包络，单位 dB。"""
    hop = max(1, int(sr * hop_ms / 1000))
    win = max(hop, int(sr * win_ms / 1000))
    n = 1 + max(0, (len(x) - win) // hop)
    if n <= 0:
        return np.zeros(0, dtype=np.float32), hop
    # 用平方和滑窗（cumsum 差）算 RMS，比逐窗循环快得多
    sq = np.concatenate(([0.0], np.cumsum(x * x, dtype=np.float64)))
    idx = np.arange(n) * hop
    tot = sq[idx + win] - sq[idx]
    rms = np.sqrt(np.maximum(tot, 0.0) / win)
    db = 20.0 * np.log10(np.maximum(rms, 1e-7))
    return db.astype(np.float32), hop


def onset_strength(db):
    """起点强度：能量的一阶正差分（只保留上升沿）。

    为什么用正差分而不是"能量超过阈值"：
      能量高只说明"这一刻很响"，但鼓点要的是"突然变响"。
      只有上升沿才对应击打瞬间 —— 这是 onset detection 的标准做法。
    """
    d = np.diff(db, prepend=db[:1])
    return np.maximum(d, 0.0)


def detect_onsets(strength, hop, sr=SR, delta=3.0, look=25):
    """自适应阈值找起点峰值。

    delta 是"要比局部均值高多少 dB"才认；look 是局部均值窗口（帧）。
    用局部均值而非全局：BGM 有前奏/主歌/副歌，响度差异很大，
    全局阈值会把安静的段落全部漏掉。

    ⚠ 边界必须防：若整段包络**完全平坦**（真静音，或削波后的方波），
      则 strength 恒为 0，而 `0 > mean + delta` 恒为假 —— 不会误报，是安全的。
      真正的风险在**除零**：局部均值窗口若被裁成 0 长会变成 NaN，
      而 `x > NaN` 恒为 False（静默漏检，不报错）。
      这里用显式长度检查挡掉，而不是靠 min/max 恰好算对。
    """
    if strength.size == 0:
        return np.zeros(0, dtype=int)
    n = len(strength)
    if n < 3:
        # 少于 3 帧无法判断"局部极大"（需要左右邻居）
        return np.zeros(0, dtype=int)
    onsets = []
    for i in range(1, n - 1):
        lo = max(0, i - look)
        hi = min(n, i + look)
        if hi <= lo:                    # 窗口退化 -> 跳过而不是算出 NaN
            continue
        thr = strength[lo:hi].mean() + delta
        # 局部极大 + 超阈值
        if strength[i] > thr and strength[i] >= strength[i - 1] \
                and strength[i] >= strength[i + 1]:
            onsets.append(i)
    return np.asarray(onsets, dtype=int)


def estimate_bpm(onsets, hop, sr=SR):
    """用起点间隔的自相关/直方图估 BPM。"""
    if onsets.size < 4:
        return 0.0, 0.0
    t = onsets * hop / sr
    ioi = np.diff(t)
    ioi = ioi[(ioi > 60.0 / BPM_MAX) & (ioi < 60.0 / BPM_MIN)]
    if ioi.size < 3:
        return 0.0, 0.0
    # 直方图归票：每个间隔投给它可能代表的 1/2、1/3、1 拍
    votes = {}
    for v in ioi:
        for mult, w in ((1, 1.0), (0.5, 0.7), (1 / 3.0, 0.5)):
            bpm = 60.0 / (v / mult)
            if BPM_MIN <= bpm <= BPM_MAX:
                key = round(bpm)
                votes[key] = votes.get(key, 0.0) + w
    if not votes:
        return 0.0, 0.0
    best = max(votes, key=votes.get)
    # 在最优值附近做加权平均，提升精度
    near = [(k, w) for k, w in votes.items() if abs(k - best) <= 2]
    tot = sum(w for _, w in near)
    refined = sum(k * w for k, w in near) / max(tot, 1e-9)
    conf = votes[best] / max(sum(votes.values()), 1e-9)
    return float(refined), float(conf)


def align_phase(onsets, bpm, hop, sr=SR, dur=None):
    """用起点做相位搜索，找第一个强拍相对 0 秒的偏移。"""
    if bpm <= 0 or onsets.size == 0:
        return 0.0
    period = 60.0 / bpm
    t = onsets * hop / sr
    best_off, best_score = 0.0, -1.0
    # 在 0~period 内扫 200 个候选相位，取与起点最贴合的那个
    for off in np.linspace(0, period, 200, endpoint=False):
        ph = np.mod(t - off, period) / period
        # 相位越接近 0 得分越高（用 cos 做平滑核）
        score = np.cos(2 * np.pi * ph).sum()
        if score > best_score:
            best_score, best_off = score, float(off)
    return best_off


def analyze(path, cache=True):
    """完整分析流程，返回节拍字典。"""
    x = decode_pcm(path)
    sr = SR
    duration = len(x) / sr
    db, hop = energy_envelope(x, sr)
    strength = onset_strength(db)
    onsets = detect_onsets(strength, hop, sr)
    bpm, conf = estimate_bpm(onsets, hop, sr)
    off = align_phase(onsets, bpm, hop, sr)

    period = 60.0 / bpm if bpm > 0 else 0.0
    beats = []
    if period > 0:
        t = off
        while t < duration:
            beats.append(round(float(t), 4))
            t += period
    # 每 4 拍取一个强拍（4/4 拍的常见假设）
    downbeats = beats[::4]

    return {
        "file": os.path.basename(path),
        "duration": round(duration, 3),
        "bpm": round(bpm, 2),
        "bpm_confidence": round(conf, 3),
        "offset": round(off, 4),
        "n_onsets": int(onsets.size),
        "beats": beats,
        "downbeats": downbeats,
        "energy_db": [round(float(v), 2) for v in db[::4]],  # 抽稀存储
        "hop_ms": HOP_MS * 4,
    }


def main():
    import sys
    if len(sys.argv) < 2:
        print("用法: python beat_analyze.py <bgm 文件> [输出.json]")
        return 1
    src = sys.argv[1]
    if not os.path.exists(src):
        print(f"文件不存在: {src}")
        return 1
    out = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.splitext(src)[0] + "_beats.json"

    print(f"分析: {src}")
    res = analyze(src)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print(f"  时长     {res['duration']}s")
    print(f"  BPM      {res['bpm']}  (置信度 {res['bpm_confidence']})")
    print(f"  相位偏移 {res['offset']}s")
    print(f"  起点数   {res['n_onsets']}")
    print(f"  拍点总数 {len(res['beats'])}")
    print(f"  强拍总数 {len(res['downbeats'])}")
    print(f"  前 8 个强拍: {res['downbeats'][:8]}")
    print(f"\n已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
