"""自动录制素材（record_auto.py）
====================================================================
不需要按键，倒数后自动录 N 秒，产出**无 HUD** 的干净素材。

为什么单独写一个：
    `sigma_realtime.py` 是交互式的（要按 R 开始/停止），
    在自动化流程里没法用。而且它默认录的 clean 画面里还有标题、
    标语、录制红点 —— 那些同样会烧进像素。
    这个脚本直接走 `process_frame(bare=True)`，录制帧零叠加。

保留 sigma 调色（压暗 + 暖金 + 人脸提亮 + 暗角），因为那是片子的"look"；
但不画人脸框、不写任何文字。

⚠ 2026-09-24 起本项目只有两条路，选错会白录：

  1. **要效果** —— 用 sigma / chad / scan / debug 任一模式直接录。
     画面会**带该模式的框、标签、线框**（用户看过效果后选了硬边框，
     视其为"效果本体"）。这些东西烧进像素，后期擦不掉。
  2. **要原画** —— 用 `--raw`（= `--no-grade`，= `--mode raw`）：
     **零处理**，逐像素等于摄像头采集帧（仅镜像），不带任何叠加。
     后期要跑 `grade.py` 必须走这条，否则会和 sigma 的压暗叠加成双重压暗。

用法：
    python record_auto.py                 # 录 25 秒（sigma 调色，无 HUD）
    python record_auto.py --secs 30
    python record_auto.py --cam 1 --countdown 5
    python record_auto.py --raw           # 完全原画：零调色，逐像素等于采集帧
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import sigma_realtime as sr       # noqa: E402
from mediapipe.tasks import python as mp_python      # noqa: E402
from mediapipe.tasks.python import vision            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="自动录制无 HUD 素材")
    ap.add_argument("--secs", type=float, default=25.0, help="录制秒数（默认 25）")
    ap.add_argument("--cam", type=int, default=1, help="摄像头索引（默认 1）")
    ap.add_argument("--countdown", type=int, default=5, help="开始前倒数秒数")
    ap.add_argument("--mode", default="sigma",
                    choices=["sigma", "chad", "scan", "debug", "raw"],
                    help="sigma/chad/scan = 三种风格；debug = 检测可视化；"
                         "raw = 完全原画（零处理）")
    # --raw 与 --no-grade 是同一件事的两个名字：
    #   --no-grade 是从"后期 grade.py 会再调一次色"的角度命名的（老名字），
    #   --raw 是从"我要摄像头原画"的角度命名的（更直白）。
    # 两个都留着，避免改坏已有脚本里的写法。
    ap.add_argument("--no-grade", "--raw", dest="no_grade", action="store_true",
                    help="完全原画：不调色、零处理，逐像素等于摄像头采集帧"
                         "（等价于 --mode raw）")
    ap.add_argument("--raw-out", metavar="PATH",
                    help="同时保存**原始分辨率**的帧（按当前模式渲染、"
                         "不带状态栏/红点、不做竖屏合成）。"
                         "⚠ 人脸特写应该用这个版本裁 —— 竖屏合成里有 "
                         "`scale=1080:-2` 会把画面缩小，实测人脸从 242px 掉到 187px（-29%%）")
    ap.add_argument("--out", help="输出路径")
    a = ap.parse_args()

    if not os.path.exists(sr.MODEL_FACE):
        print(f"模型缺失: {sr.MODEL_FACE}")
        return 1

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=sr.MODEL_FACE),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_lm = vision.FaceLandmarker.create_from_options(opts)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=sr.MODEL_POSE),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)

    cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"摄像头 {a.cam} 打不开")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, probe = cap.read()
    if not ok:
        print("读不到帧")
        return 1
    h, w = probe.shape[:2]
    print(f"摄像头 {a.cam}  {w}x{h}")

    # 预热：摄像头前几帧往往曝光不稳，先丢掉
    for _ in range(15):
        cap.read()

    win = "REC (预览)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 480, 720)

    # ---- 倒数 ----
    print(f"\n{a.countdown} 秒后开始录制 {a.secs:.0f} 秒，请就位...")
    for k in range(a.countdown, 0, -1):
        t0 = time.time()
        while time.time() - t0 < 1.0:
            ok, fr = cap.read()
            if not ok:
                break
            big = cv2.resize(fr, (480, 720))
            cv2.putText(big, str(k), (200, 400), cv2.FONT_HERSHEY_DUPLEX,
                        6.0, (0, 0, 0), 16, cv2.LINE_AA)
            cv2.putText(big, str(k), (200, 400), cv2.FONT_HERSHEY_DUPLEX,
                        6.0, (60, 190, 250), 8, cv2.LINE_AA)
            cv2.imshow(win, big)
            cv2.waitKey(1)
        print(f"  {k}")

    # ---- 录制 ----
    print(f"\n录制中...（目标 {a.secs:g}s，动作：抬头 / 喝水 / 看镜头）")
    state = sr.new_state()
    frames = []
    t_start = time.time()
    last_report = 0.0
    t_prev = t_start
    while True:
        elapsed = time.time() - t_start
        if elapsed >= a.secs:
            break
        t_a = time.time()
        ok, fr = cap.read()
        t_b = time.time()
        if not ok:
            break
        # ⚠ --no-grade / --raw 现在直接走 raw 模式，**不能再用 debug 代替**：
        #   debug 从 2026-09-24 起会画检测线框（bare 回滚后不再抑制），
        #   借它"录原图"会录到青色线框 + 绿色骨架。
        mode = "raw" if a.no_grade else a.mode
        canvas, has_face, has_pose, bare = sr.process_frame(
            fr, state, face_lm, pose_lm, mode, f"Cam {a.cam}", False, True)
        t_c = time.time()
        frames.append(bare)

        # 预览（只显示，不录进去）
        prev = cv2.resize(canvas, (480, 720))
        cv2.putText(prev, f"REC {elapsed:4.1f}/{a.secs:.0f}s",
                    (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (60, 60, 240), 2, cv2.LINE_AA)
        cv2.putText(prev, f"{len(frames)} frames", (12, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(win, prev)
        cv2.waitKey(1)
        t_d = time.time()

        # 前 5 帧打印各阶段耗时，定位瓶颈
        if len(frames) <= 5:
            print(f"  帧{len(frames)}: 读帧 {(t_b-t_a)*1000:.0f}ms  "
                  f"检测+渲染 {(t_c-t_b)*1000:.0f}ms  "
                  f"预览 {(t_d-t_c)*1000:.0f}ms  "
                  f"总计 {(t_d-t_prev)*1000:.0f}ms")
        t_prev = t_d

        if elapsed - last_report >= 5.0:
            last_report = elapsed
            print(f"  {elapsed:4.1f}s / {len(frames)} 帧  "
                  f"({len(frames)/max(elapsed,0.01):.1f} fps)")

    # ⚠ dur 必须在**释放资源之前**算。
    #   踩过的坑：一开始写在 face_lm.close() 之后，结果把 MediaPipe 的
    #   清理时间（实测 80+ 秒）也算进了录制时长 —— 于是 63 帧被当成
    #   87.4s 录制，算出 0.7fps，导出的视频变成 8fps 慢放。
    #   真实情况是每帧 41-119ms（约 20fps），完全正常。
    dur = time.time() - t_start

    cap.release()
    cv2.destroyAllWindows()
    face_lm.close()
    pose_lm.close()
    if len(frames) < 10:
        print(f"帧数太少（{len(frames)}），放弃")
        return 1
    print(f"\n录制结束: {len(frames)} 帧 / {dur:.1f}s "
          f"({len(frames)/dur:.1f} fps)")

    # ---- 导出（无 HUD，竖屏 9:16 + 模糊填充）----
    if a.out:
        # 走 save_recording 但指定输出名：它内部用时间戳命名，这里先录再改名
        sr.save_recording(frames, dur, w, h, None, silent=True)
        # 找到刚生成的文件并改名
        import glob
        cands = sorted(glob.glob(os.path.join(sr.OUTDIR, "sigma_*.mp4")),
                       key=os.path.getmtime)
        if cands:
            os.replace(cands[-1], a.out)
            print(f"完成 -> {a.out}")
    else:
        sr.save_recording(frames, dur, w, h, None, silent=True)

    # ---- 另存原始分辨率版本（给人脸特写用）----
    if a.raw_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        raw_tmp = a.raw_out + ".tmp.mp4"
        vw = cv2.VideoWriter(raw_tmp, fourcc, len(frames) / max(dur, 0.01),
                             (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        r = subprocess.run(
            [sr.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-i", raw_tmp, "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", a.raw_out],
            capture_output=True, text=True)
        os.remove(raw_tmp)
        if r.returncode == 0:
            print(f"原始分辨率 -> {a.raw_out}  {w}x{h} "
                  f"({len(frames)/max(dur,0.01):.1f} fps)")
        else:
            print(f"原始版导出失败: {(r.stderr or '')[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
