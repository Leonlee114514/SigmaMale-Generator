"""从 WorkBuddy 的 modify_backup 里还原被误删的录像。

背景（必须记下来，避免重犯）：
    我在清理临时文件时用了 `rm -f output/sigma_*.mp4`，
    把 output/ 里**所有** sigma_*.mp4 都删了 —— 其中包括
    `sigma_20260922_204319.mp4` 和 `sigma_20260922_210802.mp4`，
    这两个是**会话开始前就存在的、用户自己录的**，不是我的临时产物。

    这违反了"只删本次会话自己创建的临时物"这条规矩。
    正确做法：先 `ls` 列出候选、按时间区分"本次 vs 历史"，再动手；
    或者干脆只删自己刚生成的确切文件名，不用通配符。

    幸运的是 WorkBuddy 的 modify_backup 目录保留了副本，可以还原。
"""
import os
import re
import shutil
import subprocess

FF = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
      r"\resources\ffmpeg\ffmpeg.exe")
BK = (r"C:\Users\Lyonl\.workbuddy-ai\workspace\sessions"
      r"\4e45166d-d530-4e01-919b-427262bf2d77\modify_backup")
DST = r"C:\Users\Lyonl\Desktop\SigmaMaleGenerator\output"

PAIRS = [
    ("9.d.a5c308cc.sigma_20260922_204319.mp4", "sigma_20260922_204319.mp4"),
    ("9.d.dfd0f7ba.sigma_20260922_210802.mp4", "sigma_20260922_210802.mp4"),
]


def info(path):
    r = subprocess.run([FF, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    err = r.stderr or ""
    dur = re.search(r"Duration: (\S+),", err)
    res = re.search(r"(\d{3,4})x(\d{3,4})", err)
    fps = re.search(r"(\d+(?:\.\d+)?) fps", err)
    return (dur.group(1) if dur else "?",
            f"{res.group(1)}x{res.group(2)}" if res else "?",
            fps.group(1) if fps else "?")


os.makedirs(DST, exist_ok=True)
print("=" * 68)
print("从备份还原录像")
print("=" * 68)

for src, dst in PAIRS:
    sp, dp = os.path.join(BK, src), os.path.join(DST, dst)
    if not os.path.exists(sp):
        print(f"  ✗ 备份缺失: {src}")
        continue
    if os.path.exists(dp):
        print(f"  - 目标已存在，跳过: {dst}")
    else:
        shutil.copy2(sp, dp)
        print(f"  ✓ 已还原: {dst}")
    d, res, fps = info(dp)
    print(f"      {os.path.getsize(dp)/1024/1024:.2f}MB  {res}  {fps}fps  {d}")

print()
print("output/ 现状:")
for f in sorted(os.listdir(DST)):
    p = os.path.join(DST, f)
    if os.path.isfile(p):
        print(f"  {os.path.getsize(p)/1024/1024:8.2f}MB  {f}")
