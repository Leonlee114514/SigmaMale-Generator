"""测试新增的效果原语：缩放脉冲 / 负片闪 / 动态模糊 / 分屏镜像。

为什么先单独测：
    这些都是"表达式驱动"的滤镜（随时间变化的参数），而本机 ffmpeg 的
    表达式规则每个滤镜都不同（见 SKILL.md 坑 1/2/3）。
    直接塞进 40 段的滤镜链里，一旦报错很难定位是哪一段的问题。
    先在小素材上把每种写法跑通，再拼装。
"""
import os
import subprocess

FF = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
      r"\resources\ffmpeg\ffmpeg.exe")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "_fx")
os.makedirs(OUT, exist_ok=True)

SRC = ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2"]

W, H = 320, 180
PERIOD = 0.46875

CASES = {
    # ---- 缩放脉冲：节拍点突然放大再收回 ----
    "zoompan 缩放脉冲":
        f"zoompan=z='1+0.35*exp(-mod(time,{PERIOD})*10)'"
        f":d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={W}x{H}:fps=30",

    # ---- crop 实现的缩放脉冲（备选，不依赖 zoompan）----
    "crop 缩放脉冲":
        f"crop=w='iw/(1+0.35*exp(-mod(t,{PERIOD})*10))'"
        f":h='ih/(1+0.35*exp(-mod(t,{PERIOD})*10))'"
        f":x='(iw-ow)/2':y='(ih-oh)/2',scale={W}:{H}",

    # ---- 负片闪：节拍点短暂色彩反转 ----
    "负片闪（negate + 时间门控）":
        f"negate=enable='lt(mod(t,{PERIOD}),0.06)'",

    # ---- 动态模糊：切换瞬间的拖影 ----
    "动态模糊 tmix":
        "tmix=frames=4:weights='1 2 3 4'",

    # ---- 动态模糊 + 门控 ----
    "动态模糊 门控版":
        f"tmix=frames=4:weights='1 2 3 4':enable='gt(mod(t,{PERIOD}),0.40)'",

    # ---- 分屏镜像：左半原图 + 右半镜像 ----
    "分屏镜像 hstack":
        f"split=2[a][b];[b]hflip[bv];[a][bv]hstack,scale={W}:{H}",

    # ---- 降饱和 / 黑白 ----
    "黑白 hue=s=0":
        "hue=s=0",

    # ---- 径向模糊近似：gblur 门控 ----
    "径向模糊近似 gblur 门控":
        f"gblur=sigma='18*exp(-mod(t,{PERIOD})*12)':steps=2",
}

print(f"{'用例':30} {'结果':5} 说明")
print("-" * 84)
for name, vf in CASES.items():
    dst = os.path.join(OUT, "fx.mp4")
    if os.path.exists(dst):
        os.remove(dst)
    cmd = [FF, "-hide_banner", "-loglevel", "error", "-y"] + SRC + \
          ["-vf", vf, "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    ok = r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1024
    msg = ""
    if not ok:
        ls = [l for l in (r.stderr or "").splitlines() if l.strip()]
        msg = (ls[0].split("]")[-1].strip() if ls else "无产物")[:52]
    print(f"{name:30} {'OK' if ok else 'FAIL':5} {msg}")
