"""端到端验证：录制路径产出的帧不含调试读数（无 FPS/FACE/POSE 文字与黑条）。

用合成人脸图 + 真实 MediaPipe 模型跑一遍 process_frame，
比对 debug 画布与 clean 画布的差异：
  - clean 画布应当比 debug 画布"更亮"（没有顶部/底部黑条）
  - clean 画布不应当有底部那条黑带
"""
import os, sys, cv2, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import sigma_realtime as sr

print("加载模型...")
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

face_opts = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=sr.MODEL_FACE),
    running_mode=vision.RunningMode.VIDEO, num_faces=1)
pose_opts = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=sr.MODEL_POSE),
    running_mode=vision.RunningMode.VIDEO, num_poses=1)
face_lm = vision.FaceLandmarker.create_from_options(face_opts)
pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)

# 造一张有皮肤的合成图，尽量让人脸检测器有话可说（检测不到也不影响本测试）
frame = np.full((720, 1280, 3), 90, np.uint8)
cv2.rectangle(frame, (400, 120), (880, 620), (150, 175, 205), -1)   # 近似肤色块
cv2.circle(frame, (560, 300), 26, (40, 40, 40), -1)                 # 眼
cv2.circle(frame, (720, 300), 26, (40, 40, 40), -1)
cv2.ellipse(frame, (640, 430), (90, 40), 0, 0, 180, (60, 60, 90), -1)  # 嘴

state = sr.new_state()
out = sr.process_frame(frame, state, face_lm, pose_lm, "sigma", "Cam 0", True)
print(f"返回值个数: {len(out)}  (期望 4)")
debug_canvas, has_face, has_pose, clean_canvas = out
print(f"has_face={has_face} has_pose={has_pose}")
print(f"debug {debug_canvas.shape}  clean {clean_canvas.shape}")

# 1) 底部黑带检查：debug 最底部有一条半透明黑带，clean 完全没有。
#    注意不能直接用"平均亮度绝对值"判定 —— 画面本身可能是暗的，
#    clean 的底部亮度就等于背景亮度，未必很高。
#    正确做法是看 **debug 底部比 clean 底部暗多少**（黑带造成的压暗量）。
def band_mean(img, y0, y1):
    return float(img[y0:y1, :, :].mean())

h = debug_canvas.shape[0]
dbg_bottom = band_mean(debug_canvas, h - 70, h)
cln_bottom = band_mean(clean_canvas, h - 70, h)
dbg_top = band_mean(debug_canvas, 0, 58)
cln_top = band_mean(clean_canvas, 0, 58)

print(f"\n[底部 70px 平均亮度] debug={dbg_bottom:.1f}  clean={cln_bottom:.1f}  "
      f"(debug 应更暗)")
print(f"[顶部 58px 平均亮度] debug={dbg_top:.1f}  clean={cln_top:.1f}  "
      f"(debug 应更暗)")

diff = cv2.absdiff(debug_canvas, clean_canvas)
diff_ratio = float((diff.max(axis=2) > 18).mean()) * 100
print(f"[两图差异像素占比] {diff_ratio:.2f}%")

ok = True
if cln_bottom - dbg_bottom > 20:
    print("  PASS  clean 无底部黑带（debug 被压暗了）")
else:
    print("  FAIL  底部黑带判定异常"); ok = False
if cln_top - dbg_top > 8:
    print("  PASS  clean 无顶部压暗条")
else:
    print("  FAIL  顶部判定异常"); ok = False
if diff_ratio > 3:
    print("  PASS  两图确实不同（HUD 已分离）")
else:
    print("  FAIL  两图几乎相同，clean 未生效"); ok = False

# 2) 无 HUD 模式（--bare）：渲染层不该有金色文字/框。
#
#    ⚠ 这里必须直接比 **render_mode 的输出**，不能比 process_frame 的 clean 画布。
#      原因：那句 "IT'S NOT A PHASE..." 画在 h*0.90 处，而 hud() 会在底部
#      加一条黑带把它**盖住**。于是 clean 画布里金色像素恒为 0，
#      无论 bare 开没开 —— 测试会恒 FAIL，看起来像功能坏了。
#      直接测 render_mode 才能隔离出"到底画没画"。
GOLD = (60, 190, 250)   # BGR


def gold_count(img):
    # ⚠ SIGMA_GOLD = (60, 190, 250) 是 **BGR**，即 B=60 / G=190 / R=250。
    #   判据必须按这个写。曾错写成 `b>170 & r<130`（那是青色的范围），
    #   结果金色文字明明画了却检出 0 个像素，测试恒 FAIL。
    b = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    r = img[:, :, 2].astype(int)
    return int(((r > 200) & (g > 140) & (g < 235) & (b < 130)).sum())


r_normal = sr.render_mode(frame, frame.copy(), None, None, False, False,
                          "sigma", frame.shape[1], frame.shape[0], bare=False)
r_bare = sr.render_mode(frame, frame.copy(), None, None, False, False,
                        "sigma", frame.shape[1], frame.shape[0], bare=True)
g_normal = gold_count(r_normal)
g_bare = gold_count(r_bare)
print(f"\n[render_mode 金色像素数] 普通={g_normal}  bare={g_bare}  (bare 应为 0)")

if g_normal > 0 and g_bare == 0:
    print("  PASS  bare 模式已去掉金色 HUD 文字")
else:
    print("  FAIL  bare 判定异常（普通应为正数、bare 应为 0）"); ok = False

# 顺带确认 bare 版与普通版的差异只集中在文字区域，画面主体没被改动
d2 = cv2.absdiff(r_normal, r_bare)
ys = np.nonzero(d2.max(axis=2) > 18)[0]
if len(ys):
    print(f"  差异区域 y {ys.min()}-{ys.max()}（应集中在底部文字处，画面主体不变）")

os.makedirs(os.path.join(BASE, "output"), exist_ok=True)
cv2.imwrite(os.path.join(BASE, "output", "test_bare.jpg"),
            cv2.resize(r_bare, (360, 640)))
print("bare 渲染结果已存 output/test_bare.jpg")

cv2.imwrite(os.path.join(BASE, "output", "test_debug.jpg"),
            cv2.resize(debug_canvas, (360, 640)))
cv2.imwrite(os.path.join(BASE, "output", "test_clean.jpg"),
            cv2.resize(clean_canvas, (360, 640)))
print("\n对比图已存 output/test_debug.jpg 与 output/test_clean.jpg")
print("结果:", "全部通过" if ok else "存在失败项")
sys.exit(0 if ok else 1)
