"""验证硬切是否压在拍点上（不只是闪白）。

闪白和硬切虽然都来自同一条输出时间轴，但两者的失效模式不同：
    闪白靠表达式判定（时间对就闪），硬切靠 concat 的段边界。
如果 setpts/concat 在段边界丢帧，**切点会漂移而闪白不漂**，
于是出现"闪在拍上但切不在拍上"的错位 —— 这种错位肉眼很难判断，
必须用程序量出来。
"""
import os
import subprocess
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import edit_slop as es       # noqa: E402
import film_name as fn       # noqa: E402


def _default_video():
    """没给参数时，用 output 里**最新一个成片**；找不到就报错而不是静默解出空数组。

    ⚠ 早期版本把路径写死成临时测试文件（output/_test/...），
      清理临时目录后只会得到"zero-size array"这种难懂的报错。
    ⚠ 后来改成"演示成片 demo_卡点成片_*.mp4"，那批文件也早没了；
      而再往下的兜底是"按 mtime 取最新 mp4" —— 那会取到**刚录的素材**
      （`录制_*.mp4` 也在 output/ 里），然后报一堆看不懂的错。
      现在统一用 film_name.newest_film：判据是"有成片必有的 .plan.json 伴生文件"。
    """
    return fn.newest_film(es.OUT_DIR)


VID = sys.argv[1] if len(sys.argv) > 1 else _default_video()
if not VID or not os.path.exists(VID):
    print("找不到待验证的成片。用法: python verify_cuts.py <成片.mp4>")
    sys.exit(1)
W, H, FPS = 64, 114, 30.0


def frames_gray(path):
    cmd = [es.FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
           "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    n = len(r.stdout) // (W * H)
    a = np.frombuffer(r.stdout[:n * W * H], dtype=np.uint8)
    return a.reshape(n, W * H).astype(np.float32)


f = frames_gray(VID)
print("=" * 74)
print("硬切压点验证")
print("=" * 74)
print(f"成片 {os.path.basename(VID)}  {len(f)} 帧  {len(f)/FPS:.2f}s")

# 帧间差：硬切处会出现巨大跳变
# ⚠ 阈值不能用"99 分位" —— 那等于只取最大的 1%，最多只能检出 5 个切点，
#   会把"检出 3/23"误读成"切点没压上"，其实是检测器自己把范围限死了。
#   用**固定阈值**即可（帧间差呈明显双峰：非切点 <5，切点 >50）。
#
# ⚠⚠ 但这个阈值是**内容相关**的，换素材后要重测：
#   早期素材是"彩色测试卡 vs 暗调影片"，对比强烈，18.0 够用；
#   换成"暗调摄像头 vs 暗调影片"后，切点帧间差整体变小，要重新标定。
#   判断方法：打印中位/90%/99% 分位，看切点是否仍显著高于非切点。
#
# ⚠⚠⚠ 最重要的一条：**如果切点整体晚约 3 帧（~90ms），先查动态模糊**。
#   实测 `tmix`（时间维度混合）一启用就把画面内容拖后：
#       模糊帧数 0 -> 偏差 +31ms ；2 -> +65ms ；3+ -> +91ms（饱和）
#   因为 tmix 的输出是前 N 帧的加权平均，内容天然滞后。
#   已改用 gblur（空间模糊，不缓冲帧，零延迟）。
d = np.abs(np.diff(f, axis=0)).mean(axis=1)
THR = 18.0
cuts = [i + 1 for i in range(len(d)) if d[i] > THR]
print(f"\n帧间差: 中位 {np.median(d):.1f}  90% {np.percentile(d,90):.1f}  "
      f"99% {np.percentile(d,99):.1f}  最大 {d.max():.1f}")
print(f"阈值 {THR}  -> 超阈帧 {len(cuts)}")

# 合并相邻（一次切换可能跨 1-2 帧）
merged, cur = [], []
for i in cuts:
    if cur and i - cur[-1] <= 2:
        cur.append(i)
    else:
        if cur:
            merged.append(cur[0])
        cur = [i]
if cur:
    merged.append(cur[0])
cut_t = np.asarray(merged) / FPS

print(f"\n检出硬切 {len(cut_t)} 个")
print("  切点时刻: " + ", ".join(f"{t:.3f}" for t in cut_t[:30]))

# 理论切点：第 17 拍（Drop）之后每一拍一个
expect = np.array([es.DROP_T_VIDEO + k * es.PERIOD
                   for k in range(0, 23)])
print(f"\n理论切点（Drop 后每拍）共 {len(expect)} 个")
print("  " + ", ".join(f"{t:.3f}" for t in expect[:30]))

# 每个检出切点找最近的期望切点，算偏差
if len(cut_t):
    errs = [float(np.min(np.abs(expect - t))) * 1000 for t in cut_t]
    print(f"\n切点偏差: 均值 {np.mean(errs):.1f}ms  "
          f"中位 {np.median(errs):.1f}ms  最大 {np.max(errs):.1f}ms")
    good = sum(1 for e in errs if e < 40)
    print(f"偏差 <40ms 的切点: {good}/{len(cut_t)}")

    # 覆盖度：理论切点里有多少被检出了
    hit = sum(1 for t in expect
              if np.min(np.abs(cut_t - t)) < 0.04)
    print(f"理论切点被检出: {hit}/{len(expect)}")
    ok = np.median(errs) < 40 and hit >= len(expect) - 2
    print(f"\n判定: {'通过' if ok else '不通过'}")
