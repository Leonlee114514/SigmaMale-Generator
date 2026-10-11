"""
Sigma Male Generator — 实时摄像头版
====================================
对着摄像头做动作，实时合成 sigma / gigachad 风格画面。

流程（对应"四层管线"）：
  1. 采集   OpenCV 读取摄像头帧
  2. 检测   MediaPipe 人脸 474 关键点 + 姿态 33 关键点
  3. 合成   按当前模式把关键点/模板渲染到画面上
  4. 交互   按键切换模式、录制、截图

按键：
  Q / ESC   退出
  M         切换模式（raw -> debug -> sigma -> chad -> scan）
            其中 raw = 零处理原图（配 grade.py 调色）
  R         开始 / 停止录制
  S         保存当前帧截图
  F         切换摄像头（集成 / 虚拟）
"""

import os
import sys
import time
import math
import subprocess
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------- 路径配置
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FACE = os.path.join(BASE, "models", "face_landmarker.task")
MODEL_POSE = os.path.join(BASE, "models", "pose_landmarker_lite.task")
OUTDIR = os.path.join(BASE, "output")
os.makedirs(OUTDIR, exist_ok=True)

FFMPEG = r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format\resources\ffmpeg\ffmpeg.exe"
# 注意：ffmpeg 是原生 Windows 程序，路径必须是 C:/... 形式；
# Git Bash 风格的 /tmp/... 或 /c/... 会报 "No such file or directory"。

# ---------------------------------------------------------------- 摄像头列表
CAMERAS = [
    ("Integrated Camera", 0),
    ("Lenovo Virtual Camera", 1),
]

# ---------------------------------------------------------------- 视觉参数
SIGMA_GOLD = (60, 190, 250)      # BGR 金色
CHAD_RED = (60, 60, 220)         # BGR 暗红
SCAN_GREEN = (150, 255, 120)     # BGR 荧光绿
KEYPOINT_CYAN = (255, 220, 80)   # BGR 青
TITLE_TXT = "SIGMA MALE GENERATOR"

# ---------------------------------------------------------------- 音频参数
# 采样率用 16000 而非 44100：人声判别只需要低频能量，16k 足够，
# 而且内存占用只有 1/3（按 15fps 计，16k/单声道约 2.1KB/帧）。
AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCK = 1024       # 每次读的采样点数（约 64ms）

# 检测分辨率上限：检测用缩小图，渲染用原图。
# 实测 1280x720 下完整循环仅 4.4 fps，降到 640 宽可提升约 4 倍。
# MediaPipe 返回的关键点是归一化坐标，缩放不影响渲染正确性。
DETECT_WIDTH = 640

# 姿态检测降频：每 N 帧检测一次并复用结果。
# 实测姿态(29ms)比人脸(20ms)更贵，而降频后骨架观感几乎无差别。
POSE_EVERY = 2

# 人脸关键点中值得高亮的索引（MediaPipe FaceMesh 拓扑）
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_L = [33, 160, 158, 133, 153, 144]
EYE_R = [362, 385, 387, 263, 373, 380]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 绘制工具
def draw_glow_line(img, pts, color, thickness=2, glow=6):
    """带发光效果的折线"""
    if len(pts) < 2:
        return
    overlay = img.copy()
    cv2.polylines(overlay, [np.array(pts, np.int32)], False, color, thickness + glow,
                  cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_corner_box(img, x, y, w, h, color, pad=0, thick=2, length_ratio=0.22):
    """四角括号式框——比普通矩形更有'瞄准框'感"""
    L = int(min(w, h) * length_ratio)
    pts_sets = [
        ((x, y + L), (x, y), (x + L, y)),
        ((x + w - L, y), (x + w, y), (x + w, y + L)),
        ((x + w, y + h - L), (x + w, y + h), (x + w - L, y + h)),
        ((x + L, y + h), (x, y + h), (x, y + h - L)),
    ]
    for a, b, c in pts_sets:
        cv2.polylines(img, [np.array([a, b, c], np.int32)], False, color, thick, cv2.LINE_AA)


def put_text_shadow(img, text, org, scale, color, thick=2):
    """带 8 向描边的文字，保证在任何背景上都清晰可读且无重影"""
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_DUPLEX,
                    scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def letterbox_square(frame, size=480):
    """把帧裁成正方形（居中裁剪），用于 gigachad 式构图"""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    sq = frame[y0:y0 + side, x0:x0 + side]
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def process_frame(frame, state, face_lm, pose_lm, mode, cam_label,
                  recording=False, bare=False):
    """处理单帧：检测 + 渲染 + HUD。

    主循环与自动化测试共用此函数，确保"测的就是跑的"。
    state 是可变 dict，用于跨帧保存时间戳、姿态缓存等状态。

    返回 (debug画布, 有脸, 有姿态, clean画布)。
    clean 画布是给录制用的"作品模式"HUD（不含调试读数），
    与 debug 画布共用同一份渲染结果，不额外跑检测。

    关于 bare（⚠ 2026-09-24 起语义已弱化，详见 render_mode 的说明）：
    bare=True 时 clean 由 `render_mode(bare=True)` 产生 ——
    它会**带上该模式自己的框与标签**（那些是"效果本体"），
    只是不像 canvas 那样再加状态栏 / 录制红点。
    要**逐像素等于摄像头原画**，请用 mode="raw"（`--raw` / `--no-grade`）。
    """
    frame = cv2.flip(frame, 1)                      # 镜像，符合照镜子的直觉
    orig = frame.copy()
    h, w = frame.shape[:2]

    # ---- 性能优化：检测用缩小图，渲染用原图 ----
    # 检测在 640 宽即可保持精度，但比 1280 快约 4 倍。
    # 关键点坐标是归一化的 (0~1)，所以缩放不影响后续渲染的正确性。
    if w > DETECT_WIDTH:
        scale = DETECT_WIDTH / w
        det_img = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    else:
        det_img = frame

    # 时间戳必须全局单调递增，否则 MediaPipe 会抛
    # "Input timestamp must be monotonically increasing"。
    # 用单调时钟 + 永不回退的计数器，避免切换摄像头/重连后崩掉。
    state["ts_ms"] = max(state["ts_ms"] + 1, int(time.monotonic() * 1000))
    rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        # 人脸每帧检测（追踪精度要求高）
        face_res = face_lm.detect_for_video(mp_img, state["ts_ms"])
        # 姿态隔帧检测并复用结果：姿态耗时比人脸高约 50%，
        # 但视觉上骨架略低帧率几乎看不出来，可省下可观时间。
        if state["frame_no"] % POSE_EVERY == 0:
            pose_res = pose_lm.detect_for_video(mp_img, state["ts_ms"])
            state["last_pose"] = pose_res
        else:
            pose_res = state["last_pose"]
    except Exception as e:
        log(f"检测异常: {e}")
        face_res = pose_res = None
    state["frame_no"] += 1

    has_face = bool(face_res and face_res.face_landmarks)
    has_pose = bool(pose_res and pose_res.pose_landmarks)

    # FPS 平滑
    now = time.time()
    dt = now - state["last_t"]
    state["last_t"] = now
    if dt > 0:
        state["fps"] = (state["fps"] * 0.9) + ((1.0 / dt) * 0.1)

    canvas = render_mode(frame, orig, face_res, pose_res, has_face, has_pose,
                         mode, w, h)
    # 两个 HUD 版本：带调试读数的（屏幕显示）和干净的（录制用）。
    # 两者共用同一份 render_mode 结果和同一次检测，不重复跑模型。
    if bare:
        # 无 HUD 录制：重新渲染一份"不画框/不写字"的画面。
        # ⚠ 这里会多跑一次 render_mode，但**不会重复跑 MediaPipe 检测**
        #   （检测结果复用），所以代价只是几次 OpenCV 绘制，可忽略。
        clean = render_mode(frame, orig, face_res, pose_res, has_face,
                            has_pose, mode, w, h, bare=True)
    else:
        clean = hud(canvas.copy(), mode, state["fps"], recording,
                    has_face, has_pose, cam_label, clean=True)
    canvas = hud(canvas, mode, state["fps"], recording, has_face, has_pose,
                 cam_label, state=state)
    return canvas, has_face, has_pose, clean


def new_state():
    return {
        "ts_ms": 0,
        "frame_no": 0,
        "last_pose": None,
        "fps": 0.0,
        "last_t": time.time(),
    }


# ---------------------------------------------------------------- 主程序
def main():
    # ---- 命令行参数 ----
    # 这个脚本原本是纯交互式的（无 argparse）。加 --bare 是因为它必须
    # 在**启动时**就确定，而不能做成录制中途切换的快捷键 ——
    # 中途切换会让同一条素材里既有带框的帧又有不带框的帧，没法用。
    # ⚠ 2026-09-24 起 `--bare` **不再等于"干净素材"** ——
    #   bare 已不抑制各模式自己的绘制（见 render_mode 的说明）。
    #   要逐像素原画请用 `--raw`。
    bare = "--bare" in sys.argv or "--raw" in sys.argv
    raw_mode = "--raw" in sys.argv
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__ or "")
        print("用法: python sigma_realtime.py [--bare | --raw]\n"
              "  --raw   以原图模式启动：零处理，逐像素等于摄像头采集帧\n"
              "          （后期要跑 grade.py 调色就用这个）\n"
              "  --bare  录制时只去掉状态栏 / 录制红点；\n"
              "          ⚠ 模式自己的框与标签仍会保留（要干净素材请用 --raw）\n")
        return 0

    log("正在加载 MediaPipe 模型...")
    if not os.path.exists(MODEL_FACE) or not os.path.exists(MODEL_POSE):
        log(f"模型缺失！请确认以下文件存在：\n  {MODEL_FACE}\n  {MODEL_POSE}")
        return 1
    if raw_mode:
        log("★ 原图模式：零处理，逐像素等于摄像头画面（配 grade.py 调色）")
    elif bare:
        log("★ 不叠状态栏/红点；⚠ 模式自己的框与标签仍会保留（要干净素材用 --raw）")

    face_opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FACE),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_POSE),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    face_lm = vision.FaceLandmarker.create_from_options(face_opts)
    pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)
    log("模型加载完成")

    cam_idx = 0
    cam_label = CAMERAS[cam_idx][0]
    cap = open_camera(cam_idx)
    if cap is None:
        log("无法打开摄像头")
        return 1
    log(f"摄像头已打开: {cam_label}")

    # raw 放首位（索引 0），与网页端保持一致 —— 它是"要原画"的那个选项。
    MODES = ["raw", "debug", "sigma", "chad", "scan"]
    # 默认 sigma（带框的效果版，保持用户熟悉的手感）；
    # `--raw` 启动则直接进原图模式。
    mode_i = 0 if raw_mode else 2
    recording = False
    frames = []
    rec_t0 = 0.0

    # ---- 关于音频：刻意**不采集麦克风** ----
    # 成片的音轨固定用 BGM（gigachad theme / phonk house），卡点也建立在该 BGM
    # 的节拍分析上（见 beat_analyze.py + bgm_pipeline.py）。
    # 因此录制阶段完全不需要收声 —— 好处有三个：
    #   1. 不再受麦克风静音/权限问题影响（本机麦克风此前实测几乎无信号）
    #   2. 省掉一个音频线程和每帧的队列抽干，录制更稳
    #   3. 成片音轨一律来自 BGM，音质与卡点都由 BGM 决定，可复现
    # 若将来要做"人声口播 + BGM 混音"，再启用 sounddevice 分支即可。
    log("录制不含麦克风；音轨将在后期用固定 BGM 混入（bgm_pipeline.py）")

    state = new_state()
    last_size = None

    win = "Sigma Male Generator"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    log("按 M 切换模式 / R 录制 / S 截图 / F 换摄像头 / Q 退出")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            log("读取帧失败，尝试重连...")
            cap.release()
            time.sleep(0.5)
            cap = open_camera(cam_idx)
            if cap is None:
                break
            continue

        mode = MODES[mode_i]
        canvas, has_face, has_pose, clean_canvas = process_frame(
            frame, state, face_lm, pose_lm, mode, cam_label, recording, bare)

        # ---- 录制 ----
        # 注意：录制帧**复用同一份渲染结果**，只把 HUD 换成 clean 版本。
        # 绝不能为了拿"干净的帧"再调一次 process_frame —— 那会对同一帧重复
        # 跑 MediaPipe 检测，既浪费算力又会推进 ts_ms/frame_no 破坏状态。
        # 因此让 process_frame 同时返回 clean 画面（见其返回值）。
        if recording:
            frames.append(clean_canvas)
            last_size = (clean_canvas.shape[1], clean_canvas.shape[0])

        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('m'):
            mode_i = (mode_i + 1) % len(MODES)
            log(f"模式 -> {MODES[mode_i]}")
        elif key == ord('r'):
            if not recording:
                recording = True
                frames = []
                rec_t0 = time.time()
                log("开始录制...（无声，音轨后期用固定 BGM 混入）")
            else:
                recording = False
                dur = time.time() - rec_t0
                if last_size:
                    # audio=None -> 只出无声画面轨。
                    # 音轨与卡点在后期由 bgm_pipeline.py 统一处理，
                    # 好处是"素材/配乐"解耦：换 BGM 不用重录。
                    save_recording(frames, dur, last_size[0], last_size[1],
                                   None, silent=True)
                else:
                    log("录制内容为空")
                frames = []
        elif key == ord('s'):
            fn = os.path.join(OUTDIR, f"shot_{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(fn, canvas)
            log(f"截图已保存: {fn}")
        elif key == ord('f'):
            if recording:
                log("录制中，暂不切换摄像头")
            else:
                cam_idx = (cam_idx + 1) % len(CAMERAS)
                cam_label = CAMERAS[cam_idx][0]
                cap.release()
                time.sleep(0.3)
                cap = open_camera(cam_idx)
                log(f"切换摄像头 -> {cam_label}")

    cap.release()
    cv2.destroyAllWindows()
    face_lm.close()
    pose_lm.close()
    log("已退出")
    return 0


def open_camera(idx, tries=3):
    """尝试用不同后端打开摄像头"""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        for attempt in range(tries):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
            time.sleep(0.3)
    return None


def render_mode(frame, orig, face_res, pose_res, has_face, has_pose, mode, w, h,
                bare=False):
    """按模式渲染。

    ⚠️ 2026-09-24 起 `bare` 的语义**大幅弱化**（按用户要求回滚）：

        bare  不再抑制各模式自己的框 / 标签 / 线框 —— 那些被视为
              **效果本体**（用户看过硬边矩形框后觉得"挺帅的"）。
              现在 bare 只影响 `hud()` 那一层（状态栏、录制红点等）。
        原图  改由 **mode="raw"** 独家承担：零处理、逐像素等于采集帧。

    ⚠ 因此：**录制（bare=True）现在会带各模式的框与标签**，这是**刻意的**。
      要干净素材请用 `mode="raw"`（`record_auto.py --raw` / `--no-grade`）。

    为什么曾经要抑制：框和文字**烧进像素**、后期擦不掉。
    但用户在实际看到两种效果后选择了保留硬边框，于是把"干净"这件事
    完全交给 raw 模式承担，而不是靠 bare 兼职 —— 一个开关只表达一件事。
    """
    if mode == "raw":
        # 原图模式：**零处理**，逐像素等于 process_frame 开头镜像后的采集帧。
        #
        #   屏幕显示时 hud() 仍会叠文字（MODE: RAW / 状态栏）—— 那是屏幕，
        #   不进录制；录制取的是 clean，也就是这里的返回值，所以录下来是纯原画。
        #
        # ⚠⚠ 必须 `.copy()`，不能 `return orig`。栽过一次，记下来：
        #   process_frame 里 canvas 与 clean 是**两次独立调用**的返回值，
        #   若两次都返回同一个 `orig` 对象（ndarray 是引用语义），
        #   那么随后 `hud(canvas, ...)` 的**原地绘制**会同时改到 clean ——
        #   于是"原图模式"录出来的画面上带着状态栏文字。
        #   实测症状：与采集帧平均差 13、最大差 252（那是白色文字像素），
        #   而 debug 分支因为写的是 `orig.copy()` 所以一直是恒等的。
        return orig.copy()

    if mode == "debug":
        out = orig.copy()
        # 线框与骨架不再受 bare 抑制（2026-09-24 按用户要求回滚 bare）。
        #
        #   原来 `--no-grade` 是借 "debug + bare" 来实现"不调色、录原始画面"
        #   的，所以这里必须裸奔。现在**原图路径已改由 raw 模式承担**
        #   （record_auto.py 的 --raw / --no-grade 已直接走 mode="raw"），
        #   debug 不必再兼职"原图"，于是可以安心画线框。
        #
        #   ⚠ 后果：debug 模式下录制会带青色线框 + 绿色骨架。
        #     要干净素材用 mode="raw"。
        draw_face_wire(out, face_res, w, h, KEYPOINT_CYAN)
        draw_pose_wire(out, pose_res, w, h, SCAN_GREEN)
        return out

    if mode == "sigma":
        out = orig.copy()
        # 暗化 + 金色调
        dark = cv2.addWeighted(out, 0.55, np.zeros_like(out), 0, 20)
        if has_face:
            lm = face_res.face_landmarks[0]
            bbox = face_bbox(lm, w, h)
            if bbox:
                x, y, bw, bh = bbox
                face_crop = out[y:y + bh, x:x + bw]
                if face_crop.size > 0:
                    # 暖色矩形**直接贴**（2026-09-24 按用户要求回滚）。
                    #
                    #   回滚原因：用户看过两种效果后要了硬边的 ——
                    #   直角框「挺帅的」，羽化椭圆「怪怪的」。
                    #
                    #   ⚠ 这不是无痕处理，而是**刻意的方形框效果**：
                    #     bbox 四条直边上是一个突变，实测边界台阶 23.6，
                    #     而同一帧的自然梯度只有 3.7。将来要正式做成
                    #     "带框"模式时再打磨（见交接文档的待办）。
                    #
                    #   要**逐像素等于摄像头原画**请用 mode="raw" —— 
                    #   raw 在函数开头就 return 了，不受这里影响。
                    dark[y:y + bh, x:x + bw] = warmify(face_crop)
                draw_corner_box(dark, x - 14, y - 14, bw + 28, bh + 28,
                                SIGMA_GOLD, thick=3)
                cx = x + bw // 2
                draw_face_vignette(dark, (cx, y + bh // 2), max(bw, bh))
        out = dark
        put_text_shadow(out, "IT'S NOT A PHASE. IT'S A TAX BRACKET.",
                        (int(w*0.04), int(h*0.90)), 0.62, SIGMA_GOLD, 1)
        return out

    if mode == "chad":
        # gigachad 式正方形黑白高对比构图
        sq = letterbox_square(orig, 480)
        gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        chad = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        chad = cv2.addWeighted(chad, 1.15, np.zeros_like(chad), 0, 10)
        # 贴回画布中央
        out = np.zeros_like(orig)
        y0 = (h - 480) // 2
        x0 = (w - 480) // 2
        y0 = max(0, y0); x0 = max(0, x0)
        ys, xs = min(480, h), min(480, w)
        out[y0:y0+ys, x0:x0+xs] = chad[:ys, :xs]
        # 标签不再受 bare 抑制（2026-09-24 按用户要求回滚 bare —— 字和框
        # 属于这个模式的效果本体）。要干净素材请用 mode="raw" / --raw。
        put_text_shadow(out, "SIGMA RULE #616",
                        (int(w*0.04), int(h*0.16)), 0.9,
                        (255, 255, 255), 2)
        put_text_shadow(out, "GRINDSET IS A TAX BRACKET",
                        (int(w*0.04), int(h*0.88)), 0.6, (220, 220, 220), 1)
        return out

    if mode == "scan":
        out = orig.copy()
        # 科技扫描风：绿色网格 + 水平扫描线
        g = out.copy()
        g[:, :, 0] = np.clip(g[:, :, 0].astype(np.int16) * 0.6, 0, 255).astype(np.uint8)
        g[:, :, 2] = np.clip(g[:, :, 2].astype(np.int16) * 0.6, 0, 255).astype(np.uint8)
        for x in range(0, w, 48):
            cv2.line(g, (x, 0), (x, h), (0, 60, 0), 1)
        for y in range(0, h, 48):
            cv2.line(g, (0, y), (w, y), (0, 60, 0), 1)
        sy = int((time.time() * 260) % h)
        cv2.line(g, (0, sy), (w, sy), SCAN_GREEN, 2, cv2.LINE_AA)
        cv2.line(g, (0, sy - 26), (w, sy - 26), (0, 90, 0), 1)
        # 锁定框、读数、骨架不再受 bare 抑制（2026-09-24 按用户要求回滚 bare）。
        # 要干净素材请用 mode="raw" / --raw。
        if has_face:
            lm = face_res.face_landmarks[0]
            bbox = face_bbox(lm, w, h)
            if bbox:
                x, y, bw, bh = bbox
                draw_corner_box(g, x - 10, y - 10, bw + 20, bh + 20,
                                SCAN_GREEN, thick=2)
                put_text_shadow(g, "TARGET LOCKED", (x - 10, y - 22), 0.6,
                                SCAN_GREEN, 1)
                put_text_shadow(g, f"CONF {0.87 + 0.1*np.random.rand():.2f}",
                                (x - 10, y + bh + 40), 0.55, SCAN_GREEN, 1)
        draw_pose_wire(g, pose_res, w, h, SCAN_GREEN, thin=True)
        put_text_shadow(g, "SCANNING FOR SIGMA TRAITS...",
                        (int(w*0.04), int(h*0.86)), 0.6, SCAN_GREEN, 1)
        return g

    # 兜底：未知模式按原图处理。**同样必须 copy** ——
    # 理由见上面 raw 分支的注释：返回 orig 本体会让 hud() 的原地绘制污染录制的 clean。
    return orig.copy()


def warmify(crop):
    """暖金色调 —— 用 LUT 实现，比逐通道浮点乘快得多"""
    lut = _get_warm_lut()
    return cv2.LUT(crop, lut)


# 注：`_WARM_MASK_CACHE` 与 `_warm_mask()`（羽化椭圆遮罩）已于 2026-09-24 移除。
# 用户看过"硬边矩形框"与"羽化椭圆"两种效果后选了**硬边**的
# （"直角框挺帅的，椭圆怪怪的"），见 render_mode 的 sigma 分支。
# 实测数据（矩形边界台阶 23.6 / 羽化后 2.15 / 自然梯度 3.7）与
# output/_probe_warm/ 的前后对照图都保留，将来做"带框"模式时可作参考。


_WARM_LUT = None


def _get_warm_lut():
    """BGR 三通道各自的查表：压蓝、提红，带轻微提亮"""
    global _WARM_LUT
    if _WARM_LUT is not None:
        return _WARM_LUT
    x = np.arange(256, dtype=np.float32)
    lut = np.zeros((256, 1, 3), np.uint8)
    lut[:, 0, 0] = np.clip(x * 0.72 * 1.08 + 6, 0, 255)   # B
    lut[:, 0, 1] = np.clip(x * 0.96 * 1.08 + 6, 0, 255)   # G
    lut[:, 0, 2] = np.clip(x * 1.18 * 1.08 + 6, 0, 255)   # R
    _WARM_LUT = lut
    return lut


def face_bbox(lm, w, h):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    if not xs:
        return None
    x0 = int(max(0, min(xs) * w))
    y0 = int(max(0, min(ys) * h))
    x1 = int(min(w, max(xs) * w))
    y1 = int(min(h, max(ys) * h))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def draw_face_wire(img, face_res, w, h, color):
    if not (face_res and face_res.face_landmarks):
        return
    lm = face_res.face_landmarks[0]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    # 脸型轮廓
    outline = [pts[i] for i in FACE_OUTLINE if i < len(pts)]
    draw_glow_line(img, outline + [outline[0]] if outline else [], color, 2, 5)
    # 眼睛
    for idxs in (EYE_L, EYE_R):
        e = [pts[i] for i in idxs if i < len(pts)]
        if e:
            draw_glow_line(img, e + [e[0]], color, 1, 3)
    # 关键点
    for i, p in enumerate(pts):
        if i % 3 == 0:
            cv2.circle(img, p, 1, color, -1, cv2.LINE_AA)
    bbox = face_bbox(lm, w, h)
    if bbox:
        x, y, bw, bh = bbox
        draw_corner_box(img, x - 14, y - 14, bw + 28, bh + 28, color, thick=2)


def draw_pose_wire(img, pose_res, w, h, color, thin=False):
    if not (pose_res and pose_res.pose_landmarks):
        return
    CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),(23,25),(25,27),
            (24,26),(26,28),(27,29),(29,31),(28,30),(30,32)]
    lm = pose_res.pose_landmarks[0]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    t = 1 if thin else 2
    for a, b in CONN:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], color, t, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if i < 33:
            r = 3 if i in (0, 11, 12, 23, 24) else 2
            cv2.circle(img, p, r, color, -1, cv2.LINE_AA)


# 暗角遮罩缓存：float32 逐像素运算在 1280x720 上要 75ms/帧，
# 改为预生成单通道 uint8 遮罩（只算一次），再用 cv2.multiply 整数乘（走 SIMD）。
_VIGNETTE_CACHE = {}


def _get_vignette_mask(w, h, radius):
    """按 (w,h,radius) 缓存 uint8 暗角遮罩，值为亮度系数(0-255)。仅首次计算。"""
    key = (w, h, radius)
    m = _VIGNETTE_CACHE.get(key)
    if m is not None:
        return m

    xs = np.arange(w, dtype=np.float32) - w / 2.0
    ys = np.arange(h, dtype=np.float32) - h / 2.0
    dist = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)

    inner = radius * 1.25
    outer = radius * 2.6
    a = np.clip((dist - inner) / max(outer - inner, 1.0), 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)              # smoothstep
    mask = (255.0 * (1.0 - a * 0.45)).astype(np.uint8)

    _VIGNETTE_CACHE.clear()                  # 只保留最近一个尺寸，避免内存累积
    _VIGNETTE_CACHE[key] = mask
    return mask


def draw_face_vignette(img, center, radius):
    """柔和径向暗角 —— 预计算 uint8 遮罩 + 整数乘法，避免每帧浮点开销"""
    h, w = img.shape[:2]
    mask = _get_vignette_mask(w, h, radius)
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.multiply(img, m3, dst=img, scale=1.0 / 255.0)


def hud(img, mode, fps, recording, has_face, has_pose, cam_label,
        clean=False, state=None):
    """叠加 HUD。

    clean=True 时只画"作品的一部分"：标题、标语、录制红点。
    调试读数（FPS / FACE / POSE / 相机名 / 快捷键提示）全部不画。

    为什么要分两种：录制时若把调试读数也烧进画面，成片会被这些文字和
    顶部/底部黑条永久占据，后期没法去掉 —— 这是个只能提前规避、
    事后无法补救的问题。所以录制走 clean 路径。
    """
    h, w = img.shape[:2]

    if clean:
        # 作品模式：只留标题 + 录制红点，且不加黑条，最大限度保留画面
        put_text_shadow(img, TITLE_TXT, (18, 30), 0.72, SIGMA_GOLD, 2)
        if recording:
            r = 7 + int(2 * math.sin(time.time() * 6))
            cv2.circle(img, (w - 34, 84), r + 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (w - 34, 84), r, (40, 40, 240), -1, cv2.LINE_AA)
        return img

    # 顶部条
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.42, img, 0.58, 0, img)

    put_text_shadow(img, TITLE_TXT, (18, 26), 0.72, SIGMA_GOLD, 2)
    put_text_shadow(img, f"MODE: {mode.upper()}", (18, 50), 0.5, (235, 235, 235), 1)

    fps_txt = f"{fps:5.1f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    put_text_shadow(img, fps_txt, (w - tw - 20, 26), 0.55, (235, 235, 235), 1)

    det = f"FACE:{'Y' if has_face else '-'}  POSE:{'Y' if has_pose else '-'}"
    (dw, _), _ = cv2.getTextSize(det, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
    put_text_shadow(img, det, (w - dw - 20, 50), 0.45,
                    (150, 255, 150) if has_face else (150, 150, 150), 1)

    # 底部：单独加一条半透明条，避免文字与画面/标语互相干扰
    bar2 = img.copy()
    cv2.rectangle(bar2, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar2, 0.50, img, 0.50, 0, img)

    put_text_shadow(img, cam_label, (18, h - 48), 0.44, (170, 170, 170), 1)
    put_text_shadow(img, "Q quit | M mode | R rec | S shot | F cam",
                    (18, h - 22), 0.5, (225, 225, 225), 1)

    # 录制指示
    # 原本这里还有一条"麦克风电平条"。它存在的前提是录制要收声 ——
    # 一旦改为固定 BGM 后期混音，录制阶段就没有麦克风可言了，
    # 留着一条永远为空的电平条反而会让人以为"麦克风坏了"。
    # 故一并移除。
    if recording:
        r = 7 + int(2 * math.sin(time.time() * 6))
        cv2.circle(img, (w - 34, 84), r, (40, 40, 240), -1, cv2.LINE_AA)
        put_text_shadow(img, "REC", (w - 100, 90), 0.6, (40, 40, 240), 2)
    return img


# ---------------------------------------------------------------- 音频说明
# 早期版本在这里有 _mic_cb()（sounddevice 回调）与 voice_beat_summary()
# （按 RMS 分位判定说话帧）。现已**整体移除**，因为成片音轨改由固定 BGM
# 提供，卡点建立在该 BGM 的节拍分析上（beat_analyze.py / bgm_pipeline.py），
# 录制阶段不再收声。
#
# 保留这段注记是为了避免东西去哪了的困惑 —— 不是遗漏，是刻意的设计变更。
# 若将来要做人声口播，需要重建：sounddevice 回调必须（1）拷贝底层缓冲
# （会被复用）（2）绝不阻塞（实时音频线程）；（3）判定阈值要用**分位数**
# 而不是中位数 —— RMS 分布是双峰的，中位数会落进两峰之间的空档，
# 导致阈值比说话电平还高，实测 47/50 被误判为静音（经典统计陷阱）。

def _robust_remove(path, tries=6, delay=0.12):
    """删除临时文件，失败时重试。返回是否最终删掉。

    ⚠ 为什么必须重试（实测踩过）：
      Windows 上 `os.remove` 会**间歇性**抛 PermissionError —— 原因是
      `cv2.VideoWriter` 写完 `.release()` 后、以及 ffmpeg 刚读完输入文件时，
      系统句柄未必立刻释放。实测同一个测试连跑 4 次，有 1~2 次删不掉。

      这个失败非常隐蔽：它只在日志里打一行，若测试输出被重定向到 /dev/null
      就完全看不见，只留下 output/ 里越积越多的 `_raw_*.mp4`。
      "单次尝试失败就放弃"是这里真正的错误 —— 删除是个**时序问题**，
      重试几次就好，不该一次定生死。
    """
    for i in range(tries):
        try:
            if not os.path.exists(path):
                return True
            os.remove(path)
            return True
        except Exception:
            time.sleep(delay * (i + 1))     # 递增退避
    return not os.path.exists(path)


def save_recording(frames, duration, w, h, audio=None, silent=False,
                   out_path=None):
    """把帧序列交给 ffmpeg 编码成竖屏 9:16 短视频。

    out_path
        指定**最终文件路径**（网页录制想自己命名时用；不传则按时间戳命名，
        命令行行为完全不变）。**成功返回最终路径，失败返回 None** ——
        调用方（网页）需要知道结果文件在哪，才能立刻加进成片库。

    silent=True / audio=None
        导出**无声**成片。这是当前的正常路径：录制阶段不收声，
        音轨在后期由 `bgm_pipeline.py` 用固定 BGM 混入并做卡点。
        素材与配乐解耦，换 BGM 不必重录。

    audio=<bytes 列表>
        兼容路径：把内存里的麦克风 PCM 写成 wav 再混入。
        目前主流程不再走这条分支（见录制处的说明），保留是为了
        将来做"人声口播 + BGM 混音"时能直接复用。
    """
    if not frames:
        log("没有录到帧")
        return None
    # 用实际帧率而非硬编码 25fps，否则录制卡顿时会导致视频快放/慢放
    real_fps = len(frames) / max(duration, 0.001)
    real_fps = max(8.0, min(real_fps, 60.0))     # 限制在合理区间
    log(f"录制结束，共 {len(frames)} 帧 / {duration:.1f}s "
        f"(实际 {real_fps:.1f} fps)，开始编码...")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw = os.path.join(OUTDIR, f"_raw_{ts}.mp4")     # 中间文件始终放 OUTDIR
    if out_path:
        final = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(final), exist_ok=True)
    else:
        final = os.path.join(OUTDIR, f"sigma_{ts}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(raw, fourcc, real_fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()

    if not os.path.exists(raw):
        log("原始写入失败")
        return None

    # ---- 把内存里的录音写成 wav ----
    wav = None
    if audio:
        try:
            import wave
            wav = os.path.join(OUTDIR, f"_raw_{ts}.wav")
            with wave.open(wav, "wb") as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_RATE)
                wf.writeframes(b"".join(audio))
            if os.path.getsize(wav) < 1024:
                _robust_remove(wav)     # 同样要重试，见 _robust_remove 说明
                wav = None
        except Exception as e:
            log(f"写音轨失败: {e}")
            wav = None

    # 转 9:16 竖屏：模糊背景 + 居中原始画面
    vf = (
        f"split=2[bg][fg];"
        f"[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,gblur=sigma=28[bgb];"
        f"[fg]scale=1080:-2[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,format=yuv420p"
    )
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", raw]
    if wav:
        cmd += ["-i", wav]
    cmd += ["-vf", vf, "-r", f"{real_fps:.3f}", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-movflags", "+faststart"]
    if wav:
        # 音轨长度与画面未必严格相等，用 apad 补齐、以视频为准截断，
        # 否则容器会因为音视频时长不符出现尾帧冻结。
        cmd += ["-c:a", "aac", "-b:a", "128k", "-af", "apad",
                "-shortest"]
    cmd.append(final)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and os.path.exists(final) \
                and os.path.getsize(final) > 1024:
            size = os.path.getsize(final) / 1024 / 1024
            log(f"竖屏视频已导出: {final} ({size:.1f} MB"
                f"{'，含音轨' if wav else '，无音轨'})")
        else:
            log(f"ffmpeg 转码失败: {(r.stderr or '')[-400:]}")
            return None
    except Exception as e:
        log(f"ffmpeg 异常: {e}")
        return None

    # ---- 清理中间文件 ----
    # 必须放在转码成功的分支**之外**单独 try：
    # 本机的删除走回收站机制，可能失败（文件被占用/回收站不可用）。
    # 如果把它和转码结果判断混在一个 try 里，清理失败会被 catch 捕获并
    # 误报成"ffmpeg 异常" —— 让用户以为转码挂了，实际成片是好的。
    # 中间文件删不掉只是留点垃圾，不该影响"导出成功"这个结论。
    # 用 _robust_remove 重试，避免 Windows 句柄未释放导致的间歇性残留。
    for tmp in (raw, wav):
        if not tmp:
            continue
        if not _robust_remove(tmp):
            log(f"中间文件清理失败（不影响成片，可手动删除）: "
                f"{os.path.basename(tmp)}")

    return final


if __name__ == "__main__":
    sys.exit(main())
