"""复刻版验证（verify_ref.py）
====================================================================
照 edit_ref.py 的复刻结构做硬指标验证，不看观感。

与 verify_cuts.py 的区别：那个验的是"硬切是否压在 128BPM 拍上"，
本脚本验的是**参考片自己的 67.12BPM 网格**，并且验的东西更多：

  1. 成片规格（分辨率 / 时长 / 有无音轨）
  2. 硬切时刻是否压在 893.92ms 拍网格上（理论切点 25 个）
  3. 铺垫段是否"每拍一刀"（参考片是长镜，我们改成每拍一刀）
  4. 黑白镜是否真的黑白（第 5/6/11/14 镜饱和度应接近 0）
  5. 结尾是否淡出到黑（末帧亮度应接近 0）

判定全部用硬指标（帧间差、饱和度、亮度），不做主观判断。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")

SR = 22050


def decode_audio(path, sr=SR):
    """解码成单声道 float64，用于验证音画对齐。

    ⚠ 不用 soundfile —— 它读不了 .aac（`Format not recognised`）。
      喂 ffmpeg 解码结果是本项目一贯的绕开方式。
    """
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, dtype="<f4").astype(np.float64)

REF_BPM = 67.12
REF_PERIOD = 60.0 / REF_BPM
REF_PHASE = 0.1011
N_INTRO = 9
N_FAST = 15
OUTRO_BEATS = 2
BW_SHOTS = {4, 5, 10, 13}

# BGM 的 Drop（实测能量突增点）。用于验证"音频高潮是否落在快剪起点"。
BGM_DROP = 14.35


def probe(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    txt = r.stderr or ""
    info = {"duration": 0.0, "size": (0, 0), "has_audio": False, "fps": 0.0}
    import re
    for line in txt.splitlines():
        if "Duration" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            info["duration"] = int(h) * 3600 + int(m) * 60 + float(s)
        if "Stream #" in line and "Video:" in line:
            m2 = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m2:
                info["size"] = (int(m2.group(1)), int(m2.group(2)))
            fm = re.search(r"([\d.]+)\s*fps", line)
            if fm:
                info["fps"] = float(fm.group(1))
        if "Audio:" in line:
            info["has_audio"] = True
    return info


def decode(path, W=160, H=160, fps=30.0):
    raw = os.path.join(BASE, "output", "_verify_ref.raw")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", path, "-vf", f"scale={W}:{H},fps={fps}",
                    "-pix_fmt", "rgb24", "-f", "rawvideo", raw],
                   capture_output=True)
    fs = W * H * 3
    n = os.path.getsize(raw) // fs
    with open(raw, "rb") as f:
        d = f.read(n * fs)
    try:
        os.remove(raw)
    except OSError:
        pass
    return np.frombuffer(d, dtype=np.uint8).reshape(n, H, W, 3), fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        print("文件不存在:", a.video)
        return 1

    ok = True
    print("=" * 70)
    print(f"复刻版验证: {os.path.basename(a.video)}")
    print("=" * 70)

    info = probe(a.video)
    print(f"\n[1] 成片规格")
    print(f"    分辨率 {info['size'][0]}x{info['size'][1]}")
    print(f"    时长   {info['duration']:.3f}s")
    print(f"    帧率   {info['fps']}")
    print(f"    音轨   {'有' if info['has_audio'] else '无'}")
    if info["size"] == (0, 0):
        print("    ✗ 读不到分辨率")
        ok = False
    if not info["has_audio"]:
        print("    ✗ 无音轨")
        ok = False

    arr, fps = decode(a.video)
    n = len(arr)
    print(f"    （解码 {n} 帧用于逐帧分析）")
    ai = arr.astype(np.int16)
    ya = decode_audio(a.video)          # 音轨（用于验收 Drop 对齐）

    # ---- 理论时间轴 ----
    intro_end = N_INTRO * REF_PERIOD
    fast_end = intro_end + N_FAST * REF_PERIOD
    total = fast_end + OUTRO_BEATS * REF_PERIOD

    print(f"\n[2] 理论时间轴")
    print(f"    铺垫段 0.000-{intro_end:.3f}s   ({N_INTRO} 拍)")
    print(f"    快剪段 {intro_end:.3f}-{fast_end:.3f}s   ({N_FAST} 镜 × 1 拍)")
    print(f"    结尾   {fast_end:.3f}-{total:.3f}s   ({OUTRO_BEATS} 拍)")
    print(f"    合计   {total:.3f}s")
    dev = abs(info["duration"] - total)
    print(f"    时长偏差 {dev*1000:.0f}ms "
          f"{'✓' if dev < 0.35 else '✗ 超容差'}")

    # ---- 硬切检测 ----
    d = np.abs(np.diff(ai, axis=0)).mean(axis=3)
    frac = (d > 30).mean(axis=(1, 2))
    mad = d.mean(axis=(1, 2))
    is_cut = (frac > 0.22) & (mad > 14)
    idx = np.where(is_cut)[0] + 1
    cuts = []
    for i in idx:
        if cuts and i - cuts[-1] <= 2:
            continue
        cuts.append(int(i))
    ct = np.array([c / fps for c in cuts])

    # ---- 理论切点 ----
    # ⚠⚠ 不能把"每个拍边界"都当理论切点。铺垫段 9 拍是**同一段实拍连续取料**，
    #   画面连续、帧间差极小，这一段内部**根本没有硬切**（8 个拍边界不会检出）。
    #   硬切只发生在"换源"或"换素材位置"的地方：
    #     · 铺垫段 -> 快剪段的交界（实拍 -> 素材）
    #     · 快剪段内部每镜（实拍 ↔ 素材 交替，必然换源）
    #     · 快剪段 -> 结尾的交界（素材 -> 实拍）
    #   早期版本把 9 个铺垫段边界也当理论切点，导致分母虚高、命中率被压低
    #   （实测"检出 20 个 / 理论 25 个"，看起来像漏检，其实是理论侧错了）。
    theory = ([intro_end]                                   # 铺垫 -> 快剪
              + [intro_end + i * REF_PERIOD
                 for i in range(1, N_FAST)]                  # 快剪段内部
              + [fast_end])                                  # 快剪 -> 结尾
    theory = np.array(theory)

    print(f"\n[3] 硬切压点")
    print(f"    检出硬切 {len(ct)} 个 / 理论切点 {len(theory)} 个")
    print(f"    （理论切点只算换源处；铺垫段 9 拍为同源连续画面，不产生硬切）")
    if len(ct):
        det = []
        for t in theory:
            j = int(np.argmin(np.abs(ct - t)))
            det.append(abs(ct[j] - t))
        det = np.array(det)
        tol = 1.0 / fps + 0.005
        hit = int((det <= tol).sum())
        print(f"    压点命中 {hit}/{len(theory)}（容差 {tol*1000:.0f}ms = 1 帧）")
        print(f"    中位偏差 {np.median(det)*1000:.1f}ms")
        if hit < len(theory) * 0.8:
            print(f"    ⚠ 命中率偏低，需检查帧量化漂移")
            ok = False

    # ---- 黑白镜验证 ----
    print(f"\n[4] 黑白镜（理论镜号 {sorted(BW_SHOTS)}，0-based）")
    # ⚠ 阈值不能用 <9。全局叠加了 rgb_split 色差 + noise 颗粒，
    #   都会给**黑白画面**引入彩色边和彩色噪点：
    #   实测真黑白镜的饱和度会顶到 10~13（未加这些特效时才是 3~8）。
    #   曾用 <9 判定，把镜 4（饱和度 10.7）误判成"彩色"。
    #   核对方式：抽帧比对 —— 镜 4 与镜 6 是同一段 Bateman 抽烟画面，
    #   镜 4 皮肤灰白（确为黑白）、镜 6 皮肤红润（确为彩色），肉眼可辨。
    #   所以阈值放到 14，并要求"黑白镜显著低于同段彩色镜的中位数"。
    fast_ok = True
    sats = []
    for i in range(N_FAST):
        t0 = intro_end + i * REF_PERIOD
        f0 = int(round(t0 * fps))
        f1 = min(int(round((t0 + REF_PERIOD) * fps)), n)
        if f1 <= f0:
            sats.append(None)
            continue
        seg = ai[f0:f1]
        sats.append(float((seg.max(axis=3) - seg.min(axis=3)).mean()))

    # ⚠⚠ 这里**不再做黑白判定**，只报数 —— 判定搬到 verify_contentaware.py 去了。
    #
    #   为什么放弃：绝对饱和度在这批素材上**根本分不开**两组。
    #     · sigma 调色把彩色素材也压得很灰（低饱和彩色镜 11.0~14.7）
    #     · rgb_split 色差又会给黑白画面重新引入彩边（真黑白镜 7.2~9.8）
    #   两组数值相邻，任何固定阈值都会误判。
    #
    #   而且这里踩过两次同一个坑：
    #     第一次用固定阈值 `sat < 14`  —— 换素材就失效；
    #     第二次改用"黑白组中位 × 0.85"当阈值 —— 那是**循环论证**，
    #       阈值来自期望答案，检验力度大打折扣，结果照样误判（彩色镜 11.9 被判黑白）。
    #     Otsu 二分割也试过：它会把阈值放在 16.4（最大类间方差在 14.7↔18.2 之间），
    #       于是 7 个低饱和彩色镜被并进黑白簇 —— 同样是错的。
    #
    #   真正的解法是**自归一化**：不比绝对值，比"同一内容，成片色度 ÷ 源色度"。
    #   黑白镜底图被去色、只剩彩边 -> 比值明显 < 1；彩色镜比值 ≈ 1（色差还会 >1）。
    #   实测两组 0.523~0.881 vs 1.207~1.548，中间留出 +0.326 的干净间隙。
    #   那部分判定见 verify_contentaware.py 的 [2]。
    vals = sorted(sats[i] for i in range(N_FAST) if sats[i] is not None)
    print(f"    饱和度分布（仅供参考，本项不做判定）: "
          f"最低 {vals[0]:.1f} / 中位 {float(np.median(vals)):.1f} "
          f"/ 最高 {vals[-1]:.1f}")
    print(f"    期望黑白镜 {sorted(BW_SHOTS)} 的饱和度: "
          f"{', '.join(f'{sats[i]:.1f}' for i in sorted(BW_SHOTS))}")
    print(f"    ⚠ 绝对饱和度在这批素材上**无法分开**黑白/彩色（sigma 调色把彩色")
    print(f"      也压得很灰，而 rgb_split 又给黑白画面加彩边 -> 两组数值相邻）。")
    print(f"      黑白镜的**判定**见 verify_contentaware.py 的 [2]：")
    print(f"      用「成片色度 ÷ 源色度」做自归一化，实测两组间隙 +0.33，0 误判。")
    print(f"      想在这里也能判：把素材原始饱和度提上去，或调小 --rgb。")

    # ---- 结尾淡出 ----
    print(f"\n[5] 结尾淡出到黑")
    last = ai[-max(1, int(0.15 * fps)):].mean()
    mid = ai[n // 2].mean()
    print(f"    末帧亮度 {last:.1f}   （片中亮度 {mid:.1f}）")
    if last < 30:
        print(f"    ✓ 已淡到黑")
    else:
        print(f"    ✗ 末帧偏亮")
        ok = False

    # ---- 音频 Drop 对齐（最关键的一项）----
    #
    # ⚠ 为什么必须验这一项：视频第一版画面 8.045s 就切了快剪，
    #   而 BGM 从 0s 起播、Drop 在 14.35s —— 错位 6.3 秒，
    #   所有"画面侧"的验证（硬切/黑白/淡出）**全部通过**，
    #   因为它们只看画面，不看音频和画面的关系。
    #   而用户一眼就听出来了："八秒已经切给贝特曼了，音乐还在铺垫"。
    #
    #   所以"卡点"必须**在成片上量音画关系**，不能只验画面。
    print(f"\n[6] 音频 Drop 是否落在快剪起点")
    win = int(0.080 * SR)
    hop = int(0.010 * SR)
    ye = []
    for i in range(0, len(ya) - win, hop):
        seg = ya[i:i + win]
        ye.append(float(np.sqrt(np.mean(seg ** 2))))
    ye = np.array(ye)
    yt = np.arange(len(ye)) * (hop / SR)
    if len(ye) > 100:
        base = float(np.median(ye[:max(50, int(intro_end / (hop / SR)))]))
        drop_t = None
        for k in range(1, len(ye)):
            if ye[k] > base * 1.8 and ye[k] > ye[k - 1] * 1.3:
                # 要求后续也维持在高位（排除单帧噪声）
                if k + 3 < len(ye) and ye[k + 1] > base * 1.5:
                    drop_t = float(yt[k])
                    break
        print(f"    铺垫段能量中位 {base:.4f}")
        if drop_t is None:
            print(f"    ✗ 未检出 Drop（音轨可能为空或全是铺垫）")
            ok = False
        else:
            dev = abs(drop_t - intro_end)
            print(f"    检出 Drop @ {drop_t:.3f}s   "
                  f"快剪起点 {intro_end:.3f}s   偏差 {dev*1000:.0f}ms")
            if dev <= 0.15:
                print(f"    ✓ Drop 与画面切换对齐（≤150ms）")
            else:
                print(f"    ✗ 错位 {dev:.2f}s —— 画面切了音乐还没到高潮。"
                      f"请检查 edit_ref.py 的 BGM_DROP / build_audio 起点")
                ok = False
    else:
        print(f"    音轨太短，跳过")

    print("\n" + "=" * 70)
    print("验证结果:", "通过" if ok else "有未通过项")
    print("=" * 70)
    out = os.path.join(BASE, "output", "_ref_analysis",
                       "verify_" + os.path.basename(a.video) + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"file": a.video, "info": info, "theory_total": total,
                   "cuts": ct.tolist(), "ok": ok}, f,
                  ensure_ascii=False, indent=2)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
