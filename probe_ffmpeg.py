"""ffmpeg 表达式/滤镜规则探测（probe_ffmpeg.py）
====================================================================
把这次踩坑过程中"靠实测定下来"的规则固化成可复跑的探测脚本。
这些规则**无法靠读文档或凭经验得到** —— 本机 ffmpeg 8.1.1 的构建
在几个地方与直觉/文档不一致，而且报错信息会误导人。

规则汇总（全部实测）：

  | 滤镜      | 逗号处理              | 时间变量     | 裸减号 |
  |-----------|-----------------------|--------------|--------|
  | `setpts`  | 必须转义 `\\,`          | 大写 `T`     | 会炸   |
  | `eq`      | 引号内可不转义        | **小写 `t`** | 可用   |
  | `crop`    | **必须加引号**        | `t`/`n`      | 可用   |

  三个最容易误诊的点：
    1. `eq` 写大写 `T` 报 `Undefined constant or missing '(' in 'T,0.5)'`
       —— 报错显示的是逗号后面的片段，**看起来像逗号问题**，实际是变量名。
    2. `setpts` 的裸减号报 `Missing ')'`，**看起来像括号不配对**，实际是减号。
    3. 多输入滤镜（overlay/blend/concat）的输出被多个下游消费时，
       自动插入的 split **只在"尺寸确实发生变化"时**才暴露协商失败；
       若源尺寸与目标尺寸恰好相同，自动 split 也能通过 —— 所以这个坑
       会在"测试用小尺寸、实际用大尺寸"时突然出现。

用法: python probe_ffmpeg.py
"""
import os
import subprocess

FF = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
      r"\resources\ffmpeg\ffmpeg.exe")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "_fx")
os.makedirs(OUT, exist_ok=True)


def run(name, vf=None, fc=None, src="testsrc2=size=320x240:rate=30:duration=1",
        expect_ok=True):
    dst = os.path.join(OUT, "probe.mp4")
    if os.path.exists(dst):
        os.remove(dst)
    cmd = [FF, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", src]
    if fc:
        cmd += ["-filter_complex", fc, "-map", "[v]"]
    else:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt",
            "yuv420p", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ok = r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1024
    tag = "OK  " if ok == expect_ok else "??  "
    note = "" if ok == expect_ok else "  <-- 与预期不符！"
    print(f"  {tag} {name:40} {'通过' if ok else '失败'}{note}")
    if not ok:
        ls = [l for l in (r.stderr or "").splitlines() if l.strip()]
        if ls:
            print(f"        {ls[0][:100]}")
    return ok


print("=" * 78)
print("eq 的时间变量：大写 T 挂，小写 t 通")
print("=" * 78)
run("eq 大写 T", vf="eq=brightness='0.85*gte(T,0.5)':eval=frame",
    expect_ok=False)
run("eq 小写 t", vf="eq=brightness='0.85*gte(t,0.5)':eval=frame")
run("eq 闪白全式（小写 t + mod）",
    vf="eq=brightness='0.85*gte(t,0.5)*lt(mod(t+0.06875,0.46875),0.05)':eval=frame")
run("eq 逗号转义（也能通，故无必要）",
    vf="eq=brightness='0.85*gte(t\\,0.5)':eval=frame")

print("\n" + "=" * 78)
print("crop 的表达式：必须加引号（否则逗号被当滤镜分隔符）")
print("=" * 78)
run("crop 不加引号 + 逗号",
    vf="crop=iw-10:ih-10:5*gte(t,0.5)*sin(n):5*gte(t,0.5)*cos(n),scale=320:240",
    expect_ok=False)
run("crop 加引号 + 逗号",
    vf="crop=iw-10:ih-10:'5*gte(t,0.5)*sin(n)':'5*gte(t,0.5)*cos(n)',scale=320:240")
run("crop 仅 n（无逗号，可省引号）",
    vf="crop=iw-10:ih-10:5*sin(n):5*cos(n),scale=320:240")

print("\n" + "=" * 78)
print("setpts 的减号坑（报错信息指向括号，实际是减号）")
print("=" * 78)
run("setpts 裸减号", vf="setpts=(T<0.5)*(PTS-0.5)", expect_ok=False)
run("setpts 减号改 +-", vf="setpts=(T<0.5)*(PTS+-0.5)")
run("setpts 嵌套 if（无减号）",
    vf="setpts=if(lt(T\\,0.5)\\,PTS\\,PTS)")

print("\n" + "=" * 78)
print("多输入滤镜输出被多处消费：必须显式 split")
print("   （注意：源与目标同尺寸时自动 split 也能过，必须用**异尺寸**才暴露）")
print("=" * 78)


def norm(src_w, src_h, tgt_w, tgt_h):
    return (f"[0:v]split=2[bg][fg];"
            f"[bg]scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=increase,"
            f"crop={tgt_w}:{tgt_h},gblur=sigma=8[bgb];"
            f"[fg]scale={tgt_w}:-2[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30[n]")


for tag, (tw, th) in (("同尺寸 320x240", (320, 240)),
                      ("异尺寸 640x480", (640, 480))):
    print(f"\n  --- {tag} ---")
    N = norm(320, 240, tw, th)
    run(f"{tag} 自动 split",
        fc=f"{N};[n]trim=0:0.5,setpts=PTS-STARTPTS[a];"
           "[n]trim=0.5:0.5,setpts=PTS-STARTPTS[b];"
           "[a][b]concat=n=2:v=1:a=0[v]",
        expect_ok=False)
    run(f"{tag} 显式 split=2",
        fc=f"{N};[n]split=2[n0][n1];"
           "[n0]trim=0:0.5,setpts=PTS-STARTPTS[a];"
           "[n1]trim=0.5:0.5,setpts=PTS-STARTPTS[b];"
           "[a][b]concat=n=2:v=1:a=0[v]")

print("\n" + "=" * 78)
print("结论：三套规则各不相同，不能互相套用。写表达式前先跑这个脚本。")
print("=" * 78)
