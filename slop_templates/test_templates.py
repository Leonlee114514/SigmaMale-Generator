"""逐条实测 slop 模板滤镜链是否真能跑通"""
import os, sys, subprocess, importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("st", os.path.join(BASE, "slop_templates.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

FF = st.FFMPEG
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)

SRC = os.path.join(OUT, "test_src.mp4")
if not os.path.exists(SRC):
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=30:duration=4",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", SRC], timeout=180)
print(f"测试素材: {SRC}\n")

def test(label, vf, extra=None):
    outp = os.path.join(OUT, f"t_{label.replace(' ', '_').replace('/','-')}.mp4")
    cmd = [FF, "-hide_banner", "-loglevel", "error", "-y", "-i", SRC]
    if extra:
        cmd += extra
    cmd += ["-filter_complex" if ";" in vf else "-vf", vf,
            "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", outp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        # 关键：0 字节文件也算失败（ffmpeg 有静默失败的情形）
        size = os.path.getsize(outp) if os.path.exists(outp) else 0
        ok = r.returncode == 0 and size > 1024
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label:34} {size/1024:7.1f} KB")
        if not ok:
            for line in (r.stderr or "").strip().splitlines()[-4:]:
                print(f"          {line[:110]}")
        return ok
    except Exception as e:
        print(f"  [FAIL] {label:34} {e}")
        return False


def probe_duration(path):
    r = subprocess.run([FF, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Duration" in line:
            return line.strip().split(",")[0].replace("Duration:", "").strip()
    return "?"


print("=" * 66)
print("滤镜链实测")
print("=" * 66)

results = {}
results["分段变速"] = test("分段变速", st.speed_ramp_piecewise(duration=4.0))
results["正弦平滑变速"] = test("正弦平滑变速", st.speed_ramp_smooth())
results["恒定变速"] = test("恒定变速", st.speed_const(2.0))
results["光流插帧"] = test("光流插帧",
    f"{st.speed_const(0.5)},{st.with_motion_interpolation(60)}")
results["倒放"] = test("倒放", st.reverse_only())
results["Boomerang"] = test("Boomerang", st.boomerang_filters())
results["正方形模糊填充"] = test("正方形模糊填充", st.square_canvas(1080, True))
results["镜像hflip"] = test("镜像hflip", st.mirror("hflip"))
results["镜像拼接"] = test("镜像拼接", st.mirror_concat())
results["抖动"] = test("抖动", st.shake())
results["RGB分离"] = test("RGB分离", st.rgb_split())
results["闪白"] = test("闪白", st.flash())
results["sigma调色"] = test("sigma调色", st.sigma_grade())
results["chad调色"] = test("chad调色", st.chad_grade())
results["VHS质感"] = test("VHS质感", st.vhs_grade())
results["竖屏转换"] = test("竖屏转换", st.to_vertical())

# ---- 新增：素材"剪辑感"效果原语 ----
results["分屏镜像"] = test("分屏镜像", st.split_mirror())
results["缩放脉冲"] = test("缩放脉冲", st.zoom_pulse())
results["负片闪"] = test("负片闪", st.negate_flash())
results["动态模糊"] = test("动态模糊", st.motion_blur())
results["动态模糊门控"] = test("动态模糊门控",
    st.motion_blur(enable=f"gt(mod(t,{st.PERIOD if hasattr(st,'PERIOD') else 0.46875}),0.40)"))
results["黑白"] = test("黑白", st.bw(1.0))
results["半降饱和"] = test("半降饱和", st.bw(0.5))

# 组合链：最接近真实 slop 的一整条
combo = (f"{st.speed_ramp_piecewise(duration=4.0)},"
         f"{st.sigma_grade()},"
         f"{st.to_vertical()}")
results["组合链(变速+调色+竖屏)"] = test("组合链(变速+调色+竖屏)", combo)

# 倒放 + 变速 组合（用户特别问到的）
combo2 = f"{st.reverse_only()},{st.speed_const(1.5)}"
results["倒放+变速"] = test("倒放+变速", combo2)

# 正方 + 倒放 组合（用户明确提到的）
combo3 = f"{st.reverse_only()},{st.square_canvas(1080, True)}"
results["倒放+正方"] = test("倒放+正方", combo3)

print("\n" + "=" * 66)
print("[时长核对] 变速方向验证（源素材 4.00s）")
for lbl in ["分段变速", "正弦平滑变速", "恒定变速", "光流插帧", "倒放+变速"]:
    p = os.path.join(OUT, f"t_{lbl}.mp4")
    if os.path.exists(p) and os.path.getsize(p) > 1024:
        print(f"  {lbl:16} {probe_duration(p)}")

print("\n" + "=" * 66)
okn = sum(1 for v in results.values() if v)
print(f"通过 {okn}/{len(results)}")
if okn < len(results):
    print("失败项:", [k for k, v in results.items() if not v])
else:
    print("全部通过")
print("=" * 66)
