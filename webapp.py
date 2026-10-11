"""本地前端服务 —— 西格玛男人生成器（webapp.py）
====================================================================
把终端里的链路搬到网页上：实时预览 / 录制 / 生产线 / 成片库。
**算法一行不动** —— 全部复用现有脚本与 sigma_realtime.process_frame。

用法：
    python webapp.py                 # 127.0.0.1:8760，自动开浏览器
    python webapp.py --cam 1         # 换摄像头（0=集成摄像头，1=虚拟摄像头）
    python webapp.py --port 9000 --no-browser

也可双击 `启动网页.bat`。

设计约定（每条都对应这个项目踩过的坑，别再犯）：

  · **摄像头只能被一个进程打开。** 预览与录制共用同一个采集线程，
    所以状态条自检**只做文件存在性检查**，绝不调 check_launch.py ——
    它会打开摄像头，结果是两边都抓不到帧（check_launch 已经因此间歇失败过）。

  · **不依赖 shell。** 本机 bash 缺 head / grep / tail / dirname，
    所有子进程都直接传参数列表，不走管道。

  · **ffmpeg 是原生 Windows 程序**，路径必须是 C:/... 形式；
    Git Bash 风格的 /c/... 会报 "No such file or directory"。

  · **前端不解析中文日志。** 结构化数据一律走 /api/*，
    靠正则匹配中文输出换个措辞就失效。

  · **Flask 必须 use_reloader=False。** 否则模块被加载两遍，
    第二个进程开摄像头必然失败。
"""
import argparse
import atexit
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import sigma_realtime as sr            # noqa: E402

from flask import (Flask, Response, jsonify, request,          # noqa: E402
                   send_from_directory)

WEB = os.path.join(BASE, "web")
OUT = os.path.join(BASE, "output")
WORK = os.path.join(OUT, "_pipeline")
CACHE_FILE = os.path.join(BASE, ".webapp_cache.json")

# 生产线进度文件（由 slop_pipeline.py --progress **原子写入**）。
# 走文件而不是解析子进程 stdout —— 项目约定：**前端不解析中文日志**，
# 靠正则匹配中文输出换个措辞就静默失效。
PIPELINE_PROGRESS = os.path.join(WORK, "_web_progress.json")

# raw 放首位：它是"我要完全摄像头原画"的那个选项，语义上最该先被看到。
#
# 为什么需要它 —— 2026-09-24 起本项目是"效果 / 原画"两条路：
#   其余模式（sigma / chad / scan / debug）录下来都**带各自的框和标签**
#   （用户看过效果后选了硬边框，把那当作效果本体，所以 bare 不再抑制它们）；
#   只有 raw 是**零处理**，逐像素等于摄像头采集帧（仅镜像）。
#   「录制所见」视图把这份画布直接给你看 —— 所见即所录。
MODES = ["raw", "debug", "sigma", "chad", "scan"]
VIEWS = ["hud", "bare"]

# 录制帧数上限。帧是**整帧存在内存里**的（1280×720×3 ≈ 2.6 MB/帧），
# 不设上限的话录久了会把机器拖垮（900 帧就 2.3 GB）。
# 触顶时提前收尾并照常保存 —— 已录到的部分不会丢。
REC_MAX_FRAMES = 800

# 交错素材（参考片）的常见位置。只用来做"就绪提示"，不是硬依赖 ——
# 用户也可能把素材放在别处，页面上可以手填。
MATERIAL_HINTS = [
    # ★ 贝特曼**混剪版**素材（卡点版专用，2026-10-11 重做）：7.6s / 228 帧，
    #   **480×360（4:3）**，12 段各 19 帧（=0.6328125s，正好等于卡点版的入点步长）。
    #   **从 4 个源跨源拼**（面具 + 西装近景/微笑 + 场景 + 脸部表情）。
    #   放在第一位 = 网页默认用它。
    #
    #   ⚠ 为什么不用单一源了：用户看过各候选源的对比图后明确要**四个源都要**
    #     （①魔性表情 ②表情包层出不穷 ③4K原片 ④2022年的剪辑）。
    #     《美国精神硬人》是**特效版**（脸已被硬汉变形过）→ 排除。
    #   ⚠ 为什么是 4:3 而不是 1:1：用户反馈「整屏全是人脸看起来怪怪的」——
    #     1:1 方裁会把 16:9 的源裁掉 44% 宽度、只剩中间一张脸。
    #     4:3 只裁 25%；窄于 4:3 的源（①9:16 ④1:1）用同帧模糊补边，不留黑边。
    #   ⚠ 段长必须 == 0.6328125s（卡点版素材镜的推进步长）：
    #     短了会把一镜切成两段，长了会让一镜跨两段（画面中途跳）。
    #     复刻版步长是 0.8939213s，**不可通约**，所以另有一份专用素材。
    os.path.join(BASE, "output", "交错素材_贝特曼混剪.mp4"),
    # ★ 同一份配方的**复刻版专用**（10.73s / 322 帧，段长 0.8939213s）
    os.path.join(BASE, "output", "交错素材_贝特曼混剪_复刻.mp4"),
    # 场景版（8.4s，只用了"表情包层出不穷"一个源）—— 留着备选
    os.path.join(BASE, "output", "交错素材_贝特曼场景.mp4"),
    # 表情版 v2（8.0s，全是脸部特写）—— 留着备选
    os.path.join(BASE, "output", "交错素材_贝特曼表情_v2.mp4"),
    # 表情版 v1（10.2s，开头那段用户指认不是贝特曼）
    os.path.join(BASE, "output", "交错素材_贝特曼表情.mp4"),
    # 原素材（30s，前 19s 是抽雪茄，后段才是别镜头）—— 留着备用。
    # ⚠ 它已被移到 `sigma face/` 下；旧的 Desktop 与 output/ 两条路径
    #   早就不存在了，却一直"静默失效"（这个列表只做就绪提示，不报错）。
    #   → 失效的那两条已删掉，只留真实存在的这一条。
    os.path.join(BASE, "sigma face", "2022年的剪辑 belike.mp4"),
]


# ---------------------------------------------------------------- 运行日志
class Log:
    """有界的运行日志，供页面底部日志条读取。

    刻意**不落盘**：这是"当前会话发生了什么"，不是审计日志。
    生产线真正的日志由 slop_pipeline.py 写进 output/_pipeline/log_*.txt。
    """

    def __init__(self, n=300):
        self.buf = deque(maxlen=n)
        self.lock = threading.Lock()

    def add(self, msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        with self.lock:
            self.buf.append(line)
        print(line, flush=True)

    def tail(self, n=60):
        with self.lock:
            return list(self.buf)[-n:]


LOG = Log()


# ---------------------------------------------------------------- 采集线程
class CameraWorker(threading.Thread):
    """独占摄像头 + MediaPipe 推理 + JPEG 编码。

    为什么把录制也放在这里（而不是起 record_auto.py 子进程）：
        同一台摄像头不能被两个进程同时打开。子进程方案要么先释放摄像头
        （预览黑屏几秒 + 弹出一个多余的 OpenCV 窗口），要么两边都失败。
        共用这一个采集线程，就能做到"预览和录制是同一路画面"。

    一次 process_frame(bare=True) 同时产出两份画面：
        canvas = 带 HUD 的调试画面（能确认人脸/姿态有没有被检测到）
        clean  = 录制所见（零叠加）
    这正是 record_auto.py 的用法，保证网页预览与录制内容同源。
    """

    def __init__(self, cam=0, mode="sigma"):
        super().__init__(daemon=True)
        self.cam = cam
        self.mode = mode
        self.view = "hud"
        self.state_name = "init"
        self.err = ""
        self.stat = {}
        self.cap = None
        self.face_lm = None
        self.pose_lm = None
        self._cond = threading.Condition()
        self._seq = 0
        self._jpeg = None
        self._stop = False

        # ---- 录制状态机 ----
        # 为什么放在采集线程里、而不是起 record_auto.py 子进程：
        # **摄像头只能被一个进程打开**，子进程方案两边都抓不到帧。
        # 录制就是"同一路帧顺手存一份"，顺带保证预览与录制**同源**。
        self._rec = {
            "state": "idle",   # idle | countdown | recording | saving | done | error
            "secs": 25.0, "countdown": 5.0,
            "left": 0.0,       # 倒数剩余秒
            "elapsed": 0.0,    # 已录秒数
            "frames": 0,
            "out": "",         # 保存后的文件名
            "err": "",
            "mode": "",        # 开始录制时的画面模式（写进文件名）
        }
        self._rec_frames = None
        self._rec_lock = threading.Lock()

    # ---- HTTP 线程读这里 ----
    def snapshot(self, after=-1, timeout=2.0):
        with self._cond:
            if self._seq <= after:
                self._cond.wait_for(
                    lambda: self._seq > after or self._stop, timeout=timeout)
            return self._jpeg, self._seq

    def status(self):
        s = dict(self.stat)
        with self._rec_lock:
            rec = {k: v for k, v in self._rec.items() if k != "t0"}
        s.update(state=self.state_name, err=self.err, cam=self.cam,
                 mode=self.mode, view=self.view, rec=rec)
        return s

    def set_mode(self, m):
        if m in MODES:
            self.mode = m
            LOG.add(f"画面模式 → {m}")

    def set_view(self, v):
        if v in VIEWS and v != self.view:
            self.view = v
            LOG.add("预览 → " + ("调试画面（含 HUD）" if v == "hud"
                                else "录制所见（零叠加）"))

    # ---- 录制 ----
    def rec_start(self, secs=25.0, countdown=5.0):
        """开始录制。返回 (ok, 错误信息)。"""
        if self.state_name != "running":
            return False, "摄像头未就绪，稍等一下再试"
        with self._rec_lock:
            if self._rec["state"] in ("countdown", "recording", "saving"):
                return False, "正在录 / 正在编码，先等它结束"
            secs = max(1.0, min(float(secs), 120.0))
            countdown = max(0.0, min(float(countdown), 10.0))
            self._rec.update(
                state="countdown" if countdown > 0 else "recording",
                secs=secs, countdown=countdown, left=countdown,
                elapsed=0.0, frames=0, out="", err="", face_rate=None,
                mode=self.mode, t0=time.time())
            self._rec_frames = []
            self._rec_face_hit = 0      # 录制期间的人脸命中统计 ——
            self._rec_face_tot = 0      # 录完要据此判断"这段能不能做特写"
        LOG.add(f"开始录制：{secs:.0f}s（倒数 {countdown:.0f}s，模式 {self.mode}）")
        return True, ""

    def rec_stop(self):
        """提前结束 → 立刻进入编码。"""
        with self._rec_lock:
            if self._rec["state"] not in ("countdown", "recording"):
                return False, "当前没有在录制"
            self._finish_rec("手动停止")
        return True, ""

    def _tick_rec(self, clean, now, has_face=False):
        """在采集线程里推进录制状态机（每帧调一次）。"""
        with self._rec_lock:
            st = self._rec["state"]
            if st == "countdown":
                left = self._rec["countdown"] - (now - self._rec["t0"])
                self._rec["left"] = round(max(0.0, left), 1)
                if left > 0:
                    return
                self._rec.update(state="recording", t0=now, left=0.0)
                self._rec_frames = []
            elif st == "recording":
                self._rec["elapsed"] = round(now - self._rec["t0"], 1)
                if self._rec_frames is not None:
                    self._rec_frames.append(clean)
                    self._rec["frames"] = len(self._rec_frames)
                    # 顺手统计人脸命中率：人脸特写 / 锁头都依赖人脸检测，
                    # 录到一段"检测不到脸"的素材，后期才发现就白录了。
                    self._rec_face_tot += 1
                    if has_face:
                        self._rec_face_hit += 1
                    if len(self._rec_frames) >= REC_MAX_FRAMES:
                        self._finish_rec(f"达到帧数上限 {REC_MAX_FRAMES}")
                        return
                if now - self._rec["t0"] >= self._rec["secs"]:
                    self._finish_rec("到时")

    def _finish_rec(self, reason=""):
        """收尾：把帧丢给**独立线程**去编码。

        绝不能在采集线程里直接编码 —— save_recording 要跑几秒到几十秒，
        会把预览钉死（画面不动，看起来像崩了）。
        调用方必须已持有 self._rec_lock。
        """
        frames = self._rec_frames
        self._rec_frames = None
        if not frames:
            self._rec.update(state="idle", frames=0)
            LOG.add(f"录制取消（没录到帧）{'：' + reason if reason else ''}")
            return
        dur = max(time.time() - self._rec["t0"], 0.01)
        h, w = frames[0].shape[:2]
        mode = self._rec["mode"] or self.mode
        name = f"录制_{mode}_{datetime.now().strftime('%m%d_%H%M%S')}.mp4"
        rate = self._rec_face_hit / max(self._rec_face_tot, 1)
        self._rec.update(state="saving", frames=len(frames), out=name,
                         face_rate=round(rate, 3))
        # 人脸命中率太低就明说 —— 这段素材做人脸特写/锁头会**静默失效**
        # （特写依赖人脸检测，检测不到时它只是把画面中心裁一刀，
        #   看上去像"特写没生效"，很容易以为是功能坏了）。
        # 实测差别：有脸的素材 99.6% 检出、能切出证件照构图；
        #           没脸的素材 0% 检出、裁切窗口占满 90% 画面。
        if rate < 0.3:
            LOG.add(f"⚠ 这段素材人脸命中仅 {rate:.0%} —— "
                    f"做人脸特写 / 锁头会失效（它们依赖人脸检测）。"
                    f"要特写的话建议重录：正对镜头、脸在画面里、别太远")
        LOG.add(f"录制结束（{reason}）：{len(frames)} 帧 / {dur:.1f}s，编码中…"
                "（预览不受影响）")
        threading.Thread(
            target=self._save_rec,
            args=(frames, dur, w, h, os.path.join(OUT, name), name),
            daemon=True).start()

    def _save_rec(self, frames, dur, w, h, path, name):
        """后台保存。录到的就是「录制所见」那份画布（clean），与预览同源。"""
        try:
            got = sr.save_recording(frames, dur, w, h, None, silent=True,
                                    out_path=path)
        except Exception as e:
            got = None
            LOG.add(f"录制保存异常：{e}")
        finally:
            frames.clear()          # 1~2 GB 级别的引用，尽早放掉
        with self._rec_lock:
            if got:
                self._rec.update(state="done", out=name, err="")
                LOG.add(f"成片已保存：{name}（已进成果库，可直接播放）")
            else:
                self._rec.update(state="error", err="编码失败，详见控制台")
                LOG.add("录制保存失败：编码返回空（细节见控制台输出）")

    def stop(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()

    # ---- 采集线程主体 ----
    def run(self):
        # 摄像头打不开**不直接躺平**：设备可能只是被别的程序临时占着
        # （跑测试、开实时程序都会占）。每 5 秒重试一次，页面会跟着恢复。
        # 失败信息只播报一次，免得日志被刷屏。
        told = False
        while not self._stop:
            try:
                self._open()
                break
            except Exception as e:
                self.state_name = "error"
                self.err = str(e)
                if not told:
                    told = True
                    LOG.add(f"摄像头初始化失败：{e}（每 5 秒重试一次）")
                for _ in range(50):
                    if self._stop:
                        return
                    time.sleep(0.1)
        if self._stop:
            return

        LOG.add(f"摄像头已连接：cam {self.cam}，画面模式 {self.mode}")
        state = sr.new_state()
        label = f"Cam {self.cam}"
        fps = 0.0
        last = time.time()

        while not self._stop:
            ok, fr = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            canvas, has_face, has_pose, clean = sr.process_frame(
                fr, state, self.face_lm, self.pose_lm, self.mode, label,
                False, True)

            img = clean if self.view == "bare" else canvas
            ok2, buf = cv2.imencode(".jpg", img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ok2:
                continue

            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            # 录制状态机。帧来源**固定是 clean**（录制所见），与当前预览视图
            # 无关 —— 否则切一下视图就会录到带 HUD 的那份画面。
            self._tick_rec(clean, now, has_face)

            h, w = img.shape[:2]
            with self._cond:
                self._jpeg = buf.tobytes()
                self._seq += 1
                self.stat = {"fps": round(fps, 1),
                             "frames": state["frame_no"],
                             "has_face": has_face, "has_pose": has_pose,
                             "w": w, "h": h}
                self._cond.notify_all()

        self._close()

    def _open(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.state_name = "loading"
        for p, what in [(sr.MODEL_FACE, "人脸模型"),
                        (sr.MODEL_POSE, "姿态模型")]:
            if not os.path.exists(p):
                raise RuntimeError(f"{what}缺失：{p}")

        self.face_lm = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=sr.MODEL_FACE),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False))
        self.pose_lm = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=sr.MODEL_POSE),
                running_mode=vision.RunningMode.VIDEO, num_poses=1))

        cap = cv2.VideoCapture(self.cam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(
                f"摄像头 {self.cam} 打不开 —— 可能被别的程序占用"
                f"（实时程序 / 自动录制 / 自检脚本都会占）")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(10):        # 预热：摄像头前几帧曝光不稳
            cap.read()
        self.cap = cap
        self.state_name = "running"

    def _close(self):
        """释放摄像头。

        ⚠ **不要 close() MediaPipe 的 landmarker**：实测它的清理要 80+ 秒
        （record_auto.py 里就因为这个把清理时间算进了录制时长，导出成慢放）。
        进程退出时一起回收即可，没必要卡在退出路径上。
        """
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.state_name = "stopped"


WORKER = None

# 生产线子进程。它处理的都是**已有视频文件、不碰摄像头**，
# 所以这里可以放心起子进程 —— 与录制的情况正好相反（录制必须挂在采集线程里）。
PIPELINE = {"proc": None, "logf": None, "log": "", "started": 0.0}


# ---------------------------------------------------------------- 小工具
def probe_dur(path):
    """读视频时长。本机没有 ffprobe，只能解析 ffmpeg -i 的 stderr。"""
    if not sr.FFMPEG or not os.path.exists(sr.FFMPEG):
        return 0.0
    r = subprocess.run([sr.FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True)
    for l in (r.stderr or "").splitlines():
        if "Duration" in l:
            t = l.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            return round(int(h) * 3600 + int(m) * 60 + float(s), 2)
    return 0.0


_dur_cache = {}
_cache_lock = threading.Lock()


def dur_cached(path):
    """带缓存的时长查询。

    每个 mp4 起一次 ffmpeg 约 100ms，成片库有几十个文件，
    不做缓存每次刷新页面都要等好几秒。缓存键带上 mtime+size，
    文件被重新渲染（同名覆盖）会自动失效。
    """
    try:
        st = os.stat(path)
    except OSError:
        return 0.0
    key = f"{os.path.abspath(path)}|{int(st.st_mtime)}|{st.st_size}"
    with _cache_lock:
        if key in _dur_cache:
            return _dur_cache[key]
    d = probe_dur(path)
    with _cache_lock:
        _dur_cache[key] = d
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_dur_cache, f)
        except Exception:
            pass
    return d


def load_cache():
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            _dur_cache.update(json.load(f))
    except Exception:
        pass


# ---------------------------------------------------------------- 应用
app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/static/<path:fn>")
def static_files(fn):
    return send_from_directory(WEB, fn)


@app.get("/stream.mjpg")
def stream():
    """MJPEG 预览流。

    view 参数在流开始时生效（服务端只有一个采集线程，因此同一时刻
    只有一种预览视图 —— 本机单用户工具，够用）。
    """
    if WORKER is not None:
        WORKER.set_view(request.args.get("view", "hud"))
    boundary = b"--frame\r\n"

    def gen():
        last = -1
        while True:
            jpeg, seq = WORKER.snapshot(last, timeout=2.0)
            if jpeg is None:
                time.sleep(0.1)
                continue
            if seq == last:
                continue
            last = seq
            yield (boundary
                   + b"Content-Type: image/jpeg\r\n"
                   + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    """状态条 + 摄像头读数 + 日志尾巴，一次请求全给。

    自检**只做文件存在性检查**，不启动任何会开摄像头的脚本。
    """
    checks = []
    checks.append({
        "name": "人脸模型", "ok": os.path.exists(sr.MODEL_FACE),
        "hint": "models/face_landmarker.task"})
    checks.append({
        "name": "姿态模型", "ok": os.path.exists(sr.MODEL_POSE),
        "hint": "models/pose_landmarker_lite.task"})
    checks.append({
        "name": "复刻版 BGM", "ok": os.path.exists(
            os.path.join(BASE, "bgm", "ref_bgm.m4a")),
        "hint": "bgm/ref_bgm.m4a（从参考片提取）"})
    checks.append({
        "name": "卡点版 BGM", "ok": os.path.exists(
            os.path.join(BASE, "bgm", "gigachad_phonk.aac")),
        "hint": "bgm/gigachad_phonk.aac"})
    mat = next((p for p in MATERIAL_HINTS if os.path.exists(p)), None)
    checks.append({
        "name": "交错素材", "ok": bool(mat),
        "hint": (os.path.basename(mat) if mat
                 else "未找到参考片；出片前需指定")})
    core = ["action_detect.py", "clip_extract.py", "grade.py", "face_crop.py",
            "edit_ref.py", "edit_slop.py", "slop_pipeline.py",
            "verify_ref.py", "verify_contentaware.py",
            "verify_flash.py", "verify_cuts.py"]
    missing = [c for c in core if not os.path.exists(os.path.join(BASE, c))]
    checks.append({
        "name": "算法脚本", "ok": not missing,
        "hint": "11 个齐全" if not missing else "缺：" + "、".join(missing)})

    return jsonify({
        "checks": checks,
        "camera": WORKER.status() if WORKER else {"state": "absent"},
        "modes": MODES,
        "views": VIEWS,
        "log": LOG.tail(60),
        "material": mat,
        "pipeline": pipeline_snapshot(),
    })


@app.post("/api/cam/mode")
def api_cam_mode():
    d = request.get_json(silent=True) or {}
    m = d.get("mode")
    if m not in MODES:
        return jsonify({"ok": False, "err": f"未知模式：{m}"}), 400
    if WORKER:
        WORKER.set_mode(m)
    return jsonify({"ok": True, "mode": m})


@app.post("/api/rec/start")
def api_rec_start():
    """开始录制。

    录制**不起子进程**（摄像头独占），而是在采集线程里顺手存一份 clean 帧，
    所以录到的内容与「录制所见」预览严格同源。
    """
    if WORKER is None:
        return jsonify({"ok": False, "err": "采集线程未启动"}), 503
    d = request.get_json(silent=True) or {}
    ok, err = WORKER.rec_start(d.get("secs", 25), d.get("countdown", 5))
    return jsonify({"ok": ok, "err": err}), (200 if ok else 409)


@app.post("/api/rec/stop")
def api_rec_stop():
    """提前结束录制（立刻进入编码，不丢已录部分）。"""
    if WORKER is None:
        return jsonify({"ok": False, "err": "采集线程未启动"}), 503
    ok, err = WORKER.rec_stop()
    return jsonify({"ok": ok, "err": err}), (200 if ok else 409)


# ---------------------------------------------------------------- 生产线
def pipeline_alive():
    p = PIPELINE.get("proc")
    return p is not None and p.poll() is None


def _kill_tree(pid):
    """杀掉整棵进程树。

    slop_pipeline 自己还会拉起 action_detect / edit_slop / verify_* 等子进程，
    只杀父进程会留下孤儿继续吃 CPU（而且它们在写 output/，更难排查）。
    """
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, timeout=20)
    except Exception as e:
        LOG.add(f"结束生产线进程失败：{e}")


def pipeline_snapshot():
    """给 /api/status 用：合并「进程还活着吗」与「流水线写的进度 JSON」。"""
    p = PIPELINE.get("proc")
    running = pipeline_alive()

    snap = {"running": running, "state": "idle", "steps": [],
            "failed": "", "outputs": [], "inputs": {},
            "elapsed": 0.0, "log": PIPELINE.get("log", "")}
    try:
        with open(PIPELINE_PROGRESS, encoding="utf-8") as f:
            d = json.load(f)
        for k in ("state", "steps", "failed", "outputs", "inputs",
                  "elapsed"):
            if k in d:
                snap[k] = d[k]
    except Exception:
        pass            # 还没跑过 / 文件正在被替换 —— 用默认值即可

    if running:
        snap["state"] = "running"
    elif p is not None and p.poll() is not None:
        # 进程已结束。若进度文件还停在 running，说明它是**异常退出**
        # （没走到写 finish/fail 的分支），要如实标出来，别让页面显示"进行中"。
        rc = p.returncode
        if snap["state"] in ("running", "idle"):
            snap["state"] = "done" if rc == 0 else "failed"
        if rc != 0 and not snap["failed"]:
            snap["failed"] = f"进程以退出码 {rc} 结束（详见日志文件）"
        # 进程结束后关掉父进程这份日志句柄，避免每次启动漏一个
        if PIPELINE.get("logf"):
            try:
                PIPELINE["logf"].close()
            except Exception:
                pass
            PIPELINE["logf"] = None
    return snap


@app.post("/api/pipeline/start")
def api_pipeline_start():
    """起生产线子进程。

    参数全部走 argv 列表（不经 shell），路径里带空格/中文都不会出问题 ——
    本机 bash 缺 head/grep 这类工具，凡是拼 shell 字符串的写法都不可靠。
    """
    d = request.get_json(silent=True) or {}
    if pipeline_alive():
        return jsonify({"ok": False, "err": "生产线正在跑，先等它结束或点停止"}), 409

    material = (d.get("material") or "").strip().strip('"')
    clips = (d.get("clips") or "").strip().strip('"')
    # 实拍素材**可以多段**：前端给数组（每行一段），这里也容错逗号/换行
    # 分隔的字符串。多段会由流水线先合并 —— 只给一段的话成片镜头全来自
    # 那一段，容易反复用同一小段。
    raw_foot = d.get("footage")
    if isinstance(raw_foot, str):
        raw_foot = raw_foot.replace("\n", ",").split(",")
    footages = [str(x).strip().strip('"') for x in (raw_foot or [])
                if str(x).strip()]
    template = d.get("template") or "both"
    aspect = d.get("aspect") or "vertical"

    if template not in ("ref", "slop", "both"):
        return jsonify({"ok": False, "err": f"未知模板：{template}"}), 400
    if aspect not in ("vertical", "square"):
        return jsonify({"ok": False, "err": f"未知画幅：{aspect}"}), 400
    if not material:
        return jsonify({"ok": False, "err": "交错素材没填"}), 400
    if not os.path.exists(material):
        return jsonify({"ok": False,
                        "err": f"交错素材不存在：{material}"}), 400
    if not footages and not clips:
        return jsonify({"ok": False,
                        "err": "要填实拍素材（或已有的映射表）"}), 400
    for p in footages:
        if not os.path.exists(p):
            return jsonify({"ok": False,
                            "err": f"实拍素材不存在：{p}"}), 400

    os.makedirs(WORK, exist_ok=True)
    cmd = [sys.executable, os.path.join(BASE, "slop_pipeline.py"),
           "--template", template, "--material", material,
           "--aspect", aspect, "--progress", PIPELINE_PROGRESS]
    if clips:
        cmd += ["--clips", clips]
    else:
        for p in footages:          # 多段就各传一个 --footage
            cmd += ["--footage", p]
    if d.get("grade"):
        cmd.append("--grade")
    if d.get("face_crop"):
        cmd += ["--face-crop", str(d["face_crop"])]
        # ⚠ --face-lock 只在有 --face-crop 时才有意义（它是特写的运镜方式），
        #   所以挂在里面而不是并列。漏传它会让"锁头"静默失效 ——
        #   页面勾了、命令里没有，出片照常，只是没锁头。
        if d.get("face_lock"):
            cmd.append("--face-lock")
    if d.get("dry_run"):
        cmd.append("--dry-run")     # 只验参数与步骤，几秒就跑完

    ts = datetime.now().strftime("%m%d_%H%M%S")
    logf_path = os.path.join(WORK, f"log_web_{ts}.txt")
    try:
        fh = open(logf_path, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=BASE, stdout=fh,
                             stderr=subprocess.STDOUT)
    except Exception as e:
        return jsonify({"ok": False, "err": f"启动失败：{e}"}), 500

    PIPELINE.update(proc=p, logf=fh, log=os.path.basename(logf_path),
                    started=time.time())
    LOG.add(f"生产线启动：模板 {template} / 画幅 {aspect}"
            + ("（含调色）" if d.get("grade") else "")
            + f"，日志 {os.path.basename(logf_path)}")
    return jsonify({"ok": True, "pid": p.pid,
                    "log": os.path.basename(logf_path)})


@app.post("/api/pipeline/stop")
def api_pipeline_stop():
    p = PIPELINE.get("proc")
    if p is None or p.poll() is not None:
        return jsonify({"ok": False, "err": "生产线没在跑"}), 409
    _kill_tree(p.pid)
    if PIPELINE.get("logf"):
        try:
            PIPELINE["logf"].close()
        except Exception:
            pass
        PIPELINE["logf"] = None
    LOG.add("生产线已手动停止")
    return jsonify({"ok": True})


@app.get("/api/artifacts")
def api_artifacts():
    """成片库：扫 output/ 与 output/_pipeline/ 里的 mp4。

    验收徽章要等 verify_*.py 支持 --json 之后才点亮 ——
    在那之前**不猜**（不靠正则解析中文日志），卡片上如实显示"待接入"。
    """
    rows = []
    for d, group in [(OUT, "output"), (WORK, "_pipeline")]:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".mp4"):
                continue
            # ⚠ 只按**开头**判断 `_raw`，绝不能用 `"_raw_" in f` 这种宽匹配：
            #   本项目正常素材的命名就是 `录制_raw_0924_082604.mp4`，中间带
            #   `_raw_`，宽匹配会把它一起杀掉 ——
            #   症状是"刚录完的素材在成果库和素材候选里都看不见"，
            #   而文件明明好好地躺在 output/ 里（最难查的那种）。
            #   要排除的中间文件是 save_recording 写的 `_raw_<时间戳>.mp4`，
            #   它以 `_raw` 开头，startswith 足够精确。
            if f.startswith("_raw") or f.endswith(".tmp.mp4"):
                continue
            p = os.path.join(d, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            plan = os.path.splitext(p)[0] + ".plan.json"
            gates = os.path.join(WORK, f"验收_{os.path.splitext(f)[0]}.json")
            rows.append({
                "name": f,
                "group": group,
                # rel 是相对 output/ 的，**只够给播放接口用**（/media/<rel>）。
                "rel": os.path.relpath(p, OUT).replace("\\", "/"),
                # path 是相对**项目根**的，可以直接当 --footage / --material 喂给
                # 流水线（它以项目根为工作目录）。少了这个字段，前端只能给
                # "录制_x.mp4"，而那个路径在项目根下并不存在 ——
                # 症状是"明明在成果库里选了素材，出片却报素材不存在"。
                "path": os.path.relpath(p, BASE).replace("\\", "/"),
                "size_mb": round(st.st_size / 1024 / 1024, 2),
                "mtime": datetime.fromtimestamp(st.st_mtime)
                                  .strftime("%m-%d %H:%M"),
                "mtime_raw": st.st_mtime,
                "dur": dur_cached(p),
                "has_plan": os.path.exists(plan),
                "gate_file": os.path.exists(gates),
                "gates": [],
            })
    rows.sort(key=lambda r: -r["mtime_raw"])
    return jsonify({"artifacts": rows[:120],
                    "badge_ready": any(r["gate_file"] for r in rows)})


@app.get("/media/<path:rel>")
def media(rel):
    return send_from_directory(OUT, rel, conditional=True)


# ---------------------------------------------------------------- 启动
def main():
    global WORKER
    ap = argparse.ArgumentParser(description="西格玛男人生成器 · 本地前端")
    ap.add_argument("--cam", type=int, default=0,
                    help="摄像头索引（0=集成摄像头，1=虚拟摄像头）")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--host", default="127.0.0.1",
                    help="默认只监听本机；改成 0.0.0.0 才会被局域网访问")
    ap.add_argument("--mode", default="sigma", choices=MODES)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(WEB, "index.html")):
        print(f"页面文件缺失：{os.path.join(WEB, 'index.html')}")
        return 1

    load_cache()
    WORKER = CameraWorker(a.cam, a.mode)
    WORKER.start()

    def bye():
        # 先收生产线：它是**独立进程树**，留着会在后台继续吃 CPU、继续写
        # output/，而用户以为"关掉网页就停了"。
        if pipeline_alive():
            _kill_tree(PIPELINE["proc"].pid)
        if PIPELINE.get("logf"):
            try:
                PIPELINE["logf"].close()
            except Exception:
                pass
        if WORKER:
            WORKER.stop()
            WORKER.join(timeout=3)

    atexit.register(bye)

    url = f"http://{a.host}:{a.port}/"
    LOG.add(f"服务已启动 {url}")
    LOG.add("提示：同一台摄像头只能被一个进程占用，"
            "别同时开实时的桌面程序或自动录制脚本")

    # 中文说明一律在这里打印，不放进 .bat ——
    # 实测：.bat 里混 "chcp 65001" + UTF-8 中文 会让 cmd 的批处理读取偏移错位，
    # 行被劈成碎片当成命令执行（set PY= 那行就是这么被毁掉的）。
    # Python 走 WriteConsoleW，控制台是哪个码页都能正常显示中文，
    # 所以把文案放这儿既好看又不会踩那个坑。
    print()
    print("  " + "=" * 58)
    print("   西格玛男人生成器 · 前端页面")
    print("  " + "=" * 58)
    print(f"   地址     {url}")
    print("   浏览器应已自动打开。没打开就手动访问上面这个地址。")
    print()
    print("   Ctrl+C 退出 —— 会释放摄像头。")
    print("   同一台摄像头只能被一个进程占用：用这个页面时，别再开")
    print("   sigma_realtime.py / record_auto.py / 自检脚本，")
    print("   否则两边都抓不到帧。")
    print("  " + "=" * 58)
    print(flush=True)

    if not a.no_browser:
        # 自动开浏览器是"尽力而为"：系统没关联默认浏览器时会静默失败，
        # 那用户就只会看到一个黑窗口、以为没启动。所以失败要说出来，
        # 并指向上面那行地址（手动打开即可）。
        def _open():
            try:
                if not webbrowser.open(url):
                    LOG.add(f"没能自动打开浏览器，请手动访问 {url}")
            except Exception as e:
                LOG.add(f"自动打开浏览器失败（{e}），请手动访问 {url}")

        threading.Timer(0.8, _open).start()
    try:
        app.run(host=a.host, port=a.port, threaded=True,
                debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        bye()
    return 0


if __name__ == "__main__":
    sys.exit(main())
