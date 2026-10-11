"""为每个 slop 模板生成"抽帧对比图"，用于肉眼验收效果。

思路：从每个模板产物里等间隔抽 3 帧，横向拼成一张图。
变速类模板能从三帧的"画面推进速度"看出加速/减速是否生效；
倒放类能看出运动方向反转；正方/竖屏能看出构图变化。
"""
import os, subprocess

FF = r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format\resources\ffmpeg\ffmpeg.exe"
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
PREV = os.path.join(BASE, "previews")
os.makedirs(PREV, exist_ok=True)

# (标签, 文件名, 抽帧用的 tile 输出宽度)
ITEMS = [
    ("分段变速", "t_分段变速.mp4"),
    ("正弦平滑变速", "t_正弦平滑变速.mp4"),
    ("恒定变速", "t_恒定变速.mp4"),
    ("倒放", "t_倒放.mp4"),
    ("倒放+正方", "t_倒放+正方.mp4"),
    ("正方(正方形模糊填充)", "t_正方形模糊填充.mp4"),
    ("抖动", "t_抖动.mp4"),
    ("RGB分离", "t_RGB分离.mp4"),
    ("镜像hflip", "t_镜像hflip.mp4"),
    ("VHS质感", "t_VHS质感.mp4"),
    ("sigma调色", "t_sigma调色.mp4"),
    ("chad调色", "t_chad调色.mp4"),
    ("竖屏转换", "t_竖屏转换.mp4"),
    ("组合链(变速+调色+竖屏)", "t_组合链(变速+调色+竖屏).mp4"),
]

ok, fail = 0, []
for label, fn in ITEMS:
    src = os.path.join(OUT, fn)
    if not os.path.exists(src) or os.path.getsize(src) < 1024:
        fail.append(label)
        continue
    dst = os.path.join(PREV, f"{label}.jpg")
    # 每 1/3 处取一帧，共 3 帧，横向拼接
    vf = ("select='eq(n,5)+eq(n,15)+eq(n,25)',"
          "scale=300:-1,tile=3x1:padding=4:color=white")
    cmd = [FF, "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-vf", vf, "-frames:v", "1", "-q:v", "3", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode == 0 and os.path.exists(dst):
        ok += 1
        print(f"  [OK  ] {label:24} {os.path.getsize(dst)/1024:6.1f} KB")
    else:
        fail.append(label)
        print(f"  [FAIL] {label:24} {(r.stderr or '').strip().splitlines()[-1][:80]}")

print(f"\n生成对比图 {ok}/{len(ITEMS)}  目录: {PREV}")
if fail:
    print("失败:", fail)
