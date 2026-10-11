"""人脸特写素材生成器（face_crop.py）
====================================================================
把一段录像里的人脸**追踪并放大裁切**成特写素材，替代原始全身画面。

为什么值得做：
    原始录像里人脸只占画面很小一块（实测约 315px / 1920px 高，占 16%），
    缩到竖屏后脸部信息量极低。裁成特写后，同样的画布上脸能占 40%+，
    配合卡点剪辑的信息密度完全不是一个量级。

顺带解决了 HUD 问题 —— **但前提是素材本身没有 HUD**：
    录像里的 HUD（人脸框、状态栏、底部文字）是**烧进像素**的、擦不掉。
    试过"选金色像素 + inpaint"擦除角标，**结论是不可行**：
      · sigma 模式给整张脸加了暖金色调，**脸本身就落在"金色"范围内**，
        掩膜会把整张脸涂掉，inpaint 后脸上出现大片色块（实测比原图更糟）。
      · 根本原因：**要擦的东西和被保护的东西颜色相近**，无法靠颜色区分。

    => 正解是**从源头不画**：用 `sigma_realtime.py --bare` 或
       `record_auto.py` 录一份无 HUD 素材。

⚠⚠ 素材该用哪一份：**优先用原始分辨率的版本，不要用竖屏合成版**。
    竖屏合成里有 `scale=1080:-2`，会把 1280 宽的摄像头画面缩到 1080。
    实测同一段素材：

        合成版 1080x1920 里的人脸高  187px   -> 放大到 1920 是 5.7x
        原始版 1280x720  里的人脸高  242px   -> 放大到 1920 是 4.4x

    **差 29%**。所以录制时加 `--raw-out` 另存一份原始版，
    人脸特写从原始版裁，最后再由编辑流程合成为竖屏。

    摄像头本身的上限也测过：
        · 最高支持 1280x960（不是 1080p），但只比 720 多 8%，收益有限
        · 不支持硬件变焦（CAP_PROP_ZOOM 返回 -1）
    所以真正的杠杆是"别让合成步骤先把画面缩小"。

关键实现点（都是踩过坑的）：

  1. **必须平滑轨迹**。逐帧 bbox 抖动很大（检测噪声 + 人脸微动），
     直接拿来裁切会让画面疯狂晃动，比不裁还难看。
     这里用 EMA 平滑 + 死区（变化小于 dead_zone 就不动），
     死区是关键 —— 它把"微抖"直接吃掉，而不是把它平滑成慢速漂移。

  2. **检测失败要"保持"而不是"回中"**。偶发丢检时若立刻回中，
     画面会突然跳一下。正确做法是沿用上一帧位置，连续丢检超阈值才回中。

  3. **检测用下采样**（640 宽），坐标再乘回原尺寸。原脚本就是这么做的，
     全分辨率检测没必要且慢好几倍。

  4. **裁切框要夹在画面内**。人脸靠近边缘时裁切框会越界，必须 clamp，
     否则 OpenCV 会静默返回空图或黑边。

用法：
    python face_crop.py 输入.mp4 -o 输出.mp4
    python face_crop.py 输入.mp4 --zoom 2.4 --headroom 0.42 --chin 0.02
    python face_crop.py 输入.mp4 --out-size 1080x1350    # 4:5 竖构图
"""
import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FACE = os.path.join(BASE, "models", "face_landmarker.task")
# 两级检测的"粗定位"模型（人脸太小时整图检测会静默失效，靠它兜底）
MODEL_POSE = os.path.join(BASE, "models", "pose_landmarker_lite.task")
FFMPEG = (r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format"
          r"\resources\ffmpeg\ffmpeg.exe")
DETECT_WIDTH = 640

import mediapipe as mp                                    # noqa: E402
from mediapipe.tasks import python as mp_python           # noqa: E402
from mediapipe.tasks.python import vision                 # noqa: E402


def face_bbox(lm, w, h):
    """478 个关键点的包围盒。与原脚本同口径，保证与 HUD 位置推算一致。"""
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


def active_band(frame, rel=0.05, floor=0.35, margin=0):
    """检测"实际画面"的纵向范围（排除模糊填充条）。

    为什么需要：竖屏录像 = 9:16 画布 + 中间一条摄像头画面 + 上下模糊填充。
    人脸特写若不管这个，裁切框会伸进模糊区，画面上下各带一条糊边。

    ⚠ 判据必须用**高频能量**，不能用逐行标准差。
      试过 std，失败：模糊填充是**同一场景的模糊版**，仍保留低频结构
      （渐变、大色块），行 std 可高达 38，和实际画面的 20-40 完全重叠。
      实测 std>10 给出 [542, 1766]，比真实范围（660-1260）宽了一倍多。

      模糊的物理本质是**高频被低通滤掉**，所以判据要用
      `|原图 - 高斯模糊|` 的逐行均值。实测它能精确给出 [654, 1267]。

    ⚠ 还要取**最长连续段**而不是 min/max。模糊区偶发有一两行 HF 偏高
      （噪声/压缩伪影），用 min/max 会被这些离群点拉宽。
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = np.abs(g - cv2.GaussianBlur(g, (0, 0), 3.0)).mean(axis=1)
    thr = max(floor, hf.max() * rel)
    above = hf > thr

    # 找最长连续 True 段
    best = (0, 0)
    start = None
    for i, v in enumerate(above):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(above) - start > best[1] - best[0]:
        best = (start, len(above))

    lo, hi = int(best[0]), int(best[1])
    if hi <= lo:
        return None
    if margin:
        lo = max(0, lo - margin)
        hi = min(g.shape[0], hi + margin)
    return lo, hi


def hud_mask(bgr):
    """HUD 特征色掩膜 —— **已废弃，仅作记录**。

    试过用"选金色像素 + inpaint"擦除烧进画面的角标，**结论是不可行**：

      1. 颜色判据极易写错通道。`SIGMA_GOLD = (60,190,250)` 是 BGR，
         写成 `b>170 & r<130`（青色范围）会一个像素都选不中，
         表现为"角标明明在却擦不掉"。
      2. **改对之后更糟**：sigma 模式给整张脸加了暖金色调，
         脸本身就落在"金色"范围内 → 掩膜把整张脸都涂了，inpaint 后
         脸上出现大片色块，比原图还难看。

    根本原因：**要擦的东西和被保护的东西颜色相近**，无法靠颜色区分。
    正解是**从源头不画** —— 用 `sigma_realtime.py --bare` 录一份无 HUD 素材。
    """
    b = bgr[:, :, 0].astype(int)
    g = bgr[:, :, 1].astype(int)
    r = bgr[:, :, 2].astype(int)
    gold = (r > 190) & (g > 130) & (g < 240) & (b < 140)
    green = (g > 190) & (r > 90) & (r < 210) & (b < 180)
    red = (r > 140) & (g < 110) & (b < 110)
    m = (gold | green | red).astype(np.uint8) * 255
    if m.any():
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    return m


def smooth(prev, cur, alpha, dead_zone):
    """EMA 平滑 + 死区（**逐帧因果**，只在无法离线处理时用）。

    ⚠ 这个函数有个固有缺陷：因果滤波**必然带滞后**。alpha 越小越稳，
    但头在快速移动时会被"拖在后面"，实测垂直漂移能到 31px。
    离线处理（我们这种）应该用 `smooth_track_noncausal()` —— 对称高斯
    能看到"未来"，所以零滞后。
    """
    if prev is None:
        return cur
    out = []
    for p, c in zip(prev, cur):
        d = c - p
        if abs(d) < dead_zone:
            out.append(p)
        else:
            out.append(p + alpha * (d - np.sign(d) * dead_zone))
    return tuple(out)


def smooth_track_noncausal(vals, fps, win_sec, dead_zone=0.0):
    """**非因果**轨迹平滑：对称高斯 + 缺失段线性插值 + 中值去野值。

    为什么必须用非因果（借鉴自 Codex 那版 lockhead.py 的实现思路）：
        因果滤波（EMA/一阶低通）只看过去，**必然产生相位滞后** ——
        头动得越快，滞后越大。而我们整段视频都在手上，完全可以"看到未来"。
        对称高斯核的群延迟为 0，所以**平滑掉抖动的同时零滞后**。
        实测这一点把垂直漂移从 31px 压到个位数。

    三步：
        1. 缺失帧线性插值（检测丢失时不能填 0，否则轨迹被拉向原点）
        2. 中值滤波去野值（偶发误检会造出尖刺，高斯抹不平只会拖出长尾）
        3. 对称高斯平滑（win_sec 秒标准差）
    """
    a = np.asarray(vals, float)
    n = len(a)
    if n == 0:
        return a
    # 1) 缺失（0）→ 线性插值
    idx = np.arange(n)
    good = a != 0
    if good.sum() >= 2:
        a = np.interp(idx, idx[good], a[good])
    elif good.sum() == 1:
        a = np.full(n, a[good][0])
    # 2) 中值去野值（窗口 5 帧）
    k = 5
    if n >= k:
        pad = np.pad(a, k // 2, mode="edge")
        med = np.array([np.median(pad[i:i + k]) for i in range(n)])
        # 偏离中值超过 3 倍 MAD 的点视为野值，替换掉
        mad = np.median(np.abs(a - med)) + 1e-6
        bad = np.abs(a - med) > 3.0 * mad
        a = np.where(bad, med, a)
    # 3) 对称高斯（sigma = win_sec * fps，截断到 ±3σ）
    sigma = max(0.5, win_sec * fps)
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    ap = np.pad(a, r, mode="edge")
    return np.convolve(ap, ker, mode="valid")


# ------------------------------------------------------------------ 人脸检测
#
# ⚠⚠ 为什么需要"两级检测"（2026-09-24 加的，踩过一次真实事故）
#
#   MediaPipe 的人脸检测器**先看脸占画面的相对比例**，再看绝对像素。
#   实测（同一段素材，只改检测图宽度）：
#
#       检测图  512 / 640 / 800 / 960 / 1080 / 1280 宽  →  检出率全部 0%
#
#   提高分辨率**完全无效** —— 因为检测器内部总要把输入缩到它固定的输入
#   尺寸，等比缩放**不改变"脸占画面"的比例**。把门槛（detection /
#   presence confidence）降到 0.1 也只从 0/44 变成 3/44，是噪声级。
#
#   真正决定成败的是相对比例。实测三档：
#       脸高占画面 1.5%  → 检出 0%      （镜头拉得远，画面看着完全正常）
#       脸高占画面 2.3%  → 检出 79%
#       脸高占画面 5.4%  → 检出 80%
#   门槛大致在 2% 上下，1.5% 就掉线了，而且**不报错** —— 表现为
#   "特写/锁头静默失效，输出只是把画面中心裁了一刀"。
#
#   解法只有一个方向：**先粗定位脸，再把那块放大到检测器认得的尺寸**。
#   姿态模型（PoseLandmarker）恰好能在脸很小时稳定给出位置（实测
#   鼻/眼/耳可见度 0.99），于是：整图检测失败 → 用姿态框出脸 → 裁出来
#   放大到 DETECT_WIDTH 再检。实测检出率 0% → 100%。
def to_mp_image(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pose_face_roi(pose_res, W, H, grow=2.0, min_vis=0.5):
    """从姿态结果里框出人脸区域（原图像素，正方形）。

    只用鼻(0)/左眼(2)/右眼(5)/左耳(7)/右耳(8) 五点 —— 它们由头骨决定，
    是最稳的脸部定位。`grow` 是相对五点跨度的放大倍数：
    2.0 表示 ROI 边长 = 2×五点跨度（脸约占 ROI 的 50%，对检测器足够）。
    可见度不足时返回 None（人背对 / 出画 / 躺着）。
    """
    if not (pose_res and pose_res.pose_landmarks):
        return None
    lm = pose_res.pose_landmarks[0]
    ids = (0, 2, 5, 7, 8)
    if min(lm[i].visibility for i in ids) < min_vis:
        return None
    xs = [lm[i].x * W for i in ids]
    ys = [lm[i].y * H for i in ids]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * grow
    side = max(side, 48.0)
    side = min(side, float(min(W, H)))      # 不能比画面还大
    # 完整落在画面内（必要时贴着边，但保证不越界 —— 越界会让裁切坐标
    # 落到负数，下游的反射填充虽然能兜住，但坐标语义会变得难追踪）
    x0 = min(max(cx - side / 2, 0.0), W - side)
    y0 = min(max(cy - side / 2, 0.0), H - side)
    return int(x0), int(y0), int(side), int(side)


def track_faces(video, fps, W, H, roi_mode, pose_every=4, verbose=True):
    """逐帧检测人脸轨迹，返回 (raw_boxes, raw_eyes, detected, n, pose_hit)。

    roi_mode=False  整图缩到 DETECT_WIDTH 检测（快，脸太小时失效）
    roi_mode=True   姿态定位脸区域 → 裁出放大再检（脸占比被放大）
    """
    if not os.path.exists(MODEL_FACE):
        raise RuntimeError("模型缺失: %s" % MODEL_FACE)

    flm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FACE),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False))
    plm = None
    if roi_mode:
        if not os.path.exists(MODEL_POSE):
            flm.close()
            raise RuntimeError("姿态模型缺失，无法启用两级检测: %s" % MODEL_POSE)
        plm = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_POSE),
                running_mode=vision.RunningMode.VIDEO, num_poses=1))

    cap = cv2.VideoCapture(video)
    raw_boxes, raw_eyes = [], []
    detected = 0
    pose_hit = 0
    idx = 0
    last_roi = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        ts = int(idx * 1000 / fps)

        bb = eyes = None
        if not roi_mode:
            # ---- 路径 A：整图检测 ----
            sc = DETECT_WIDTH / w if w > DETECT_WIDTH else 1.0
            det = (cv2.resize(frame, (DETECT_WIDTH, int(h * sc)))
                   if sc != 1.0 else frame)
            res = flm.detect_for_video(to_mp_image(det), ts)
            if res and res.face_landmarks:
                lm = res.face_landmarks[0]
                bb = face_bbox(lm, det.shape[1], det.shape[0])
                if bb:
                    bb = tuple(int(v / sc) for v in bb)     # 乘回原尺寸
                    # 两眼中点（索引 33 / 263）
                    # ⚠ 归一化坐标 × 检测图尺寸 = 检测图像素，再 /sc = 原图
                    ex = (lm[33].x + lm[263].x) / 2 * det.shape[1]
                    ey = (lm[33].y + lm[263].y) / 2 * det.shape[0]
                    eyes = (ex / sc, ey / sc)
        else:
            # ---- 路径 B：姿态定位 ROI → 裁出放大 ----
            # 姿态不必每帧跑（脸移动慢），隔 pose_every 帧刷新一次 ROI
            if last_roi is None or idx % pose_every == 0:
                sc = DETECT_WIDTH / w if w > DETECT_WIDTH else 1.0
                small = (cv2.resize(frame, (DETECT_WIDTH, int(h * sc)))
                         if sc != 1.0 else frame)
                pr = plm.detect_for_video(to_mp_image(small), ts)
                r = pose_face_roi(pr, w, h)
                if r:
                    last_roi = r
                    pose_hit += 1
            if last_roi is not None:
                rx, ry, rw, rh = last_roi
                tile = frame[ry:ry + rh, rx:rx + rw]
                if tile.size:
                    dw = DETECT_WIDTH
                    dh = max(1, int(round(rh * dw / float(rw))))
                    interp = (cv2.INTER_AREA if dw < rw else cv2.INTER_CUBIC)
                    det = cv2.resize(tile, (dw, dh), interpolation=interp)
                    res = flm.detect_for_video(to_mp_image(det), ts)
                    if res and res.face_landmarks:
                        lm = res.face_landmarks[0]
                        bbd = face_bbox(lm, det.shape[1], det.shape[0])
                        if bbd:
                            # ⚠⚠ 坐标换算必须走"两条腿"，不能凭直觉乘除。
                            #   `lm[].x / lm[].y` 是**归一化**坐标(0~1)，
                            #   而 `face_bbox` 返回的是**检测图像素**。
                            #   本路径的单位链是：
                            #     归一化 --×dw--> 检测图像素 --×k--> ROI 内像素
                            #            --+rx/ry--> 原图像素
                            #   其中 k = rw/dw = rh/dh（resize 等比，两者相等，
                            #   所以 x 用 k、y 也用它，**不能乘 dw/dh**）。
                            #
                            #   这里曾出过一个严重 bug：把归一化值当成像素值，
                            #   写成 `(lm.x + lm.x)/2 / sc * w`（多除一次 sc），
                            #   眼睛坐标被放大 2 倍 → 两眼中点飞出脸框 4~5 个脸宽，
                            #   28% 的帧落到画面外 → 裁切窗被 clamp 到边缘，
                            #   裁出来**全是背景、没有脸**。经验：写换算前先把
                            #   单位链写出来，再核对代码走的是哪条腿。
                            k = rw / float(dw)          # 检测图像素 → 原图
                            bb = (int(bbd[0] * k) + rx, int(bbd[1] * k) + ry,
                                  int(bbd[2] * k), int(bbd[3] * k))
                            ex = (lm[33].x + lm[263].x) / 2 * dw * k + rx
                            ey = (lm[33].y + lm[263].y) / 2 * dh * k + ry
                            eyes = (ex, ey)

        if bb:
            detected += 1
        raw_boxes.append(bb)
        raw_eyes.append(eyes)
        idx += 1

    cap.release()
    flm.close()
    if plm is not None:
        plm.close()
    if verbose:
        tag = "姿态定位 ROI" if roi_mode else "整图 640"
        print(f"  [{tag}] 检出 {detected}/{idx} 帧"
              f" ({detected / max(1, idx) * 100:.1f}%)"
              + (f"｜姿态命中 {pose_hit}" if roi_mode else ""))
    return raw_boxes, raw_eyes, detected, idx, pose_hit


def main():
    ap = argparse.ArgumentParser(description="人脸特写素材生成器")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", help="输出路径")
    ap.add_argument("--zoom", type=float, default=3.0,
                    help="裁切高度 = 人脸高 × zoom（默认 3.0）。"
                         "⚠ 人脸框只覆盖额到下巴，不含头发和肩 —— "
                         "zoom 3.0 约等于**证件照构图**（头上留 1 个脸高、"
                         "头下留 1 个脸高，把肩也带进来）。"
                         "调小会更紧（只看到脸），调大会更远（半身）")
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="头顶留白占人脸高的比例（默认 1.0，给头发留空间）")
    ap.add_argument("--face-pos", type=float, default=0.50,
                    help="⚠ **推荐用这个控制构图**：人脸中心在输出画面里的"
                         "目标垂直位置（0=顶部, 0.5=居中, 1=底部）。"
                         "默认 0.50 = 居中。"
                         "旧版用 headroom 推导出的位置约 0.66（人脸偏下、"
                         "头顶留白过多），已不再作为默认。"
                         "想让人脸更靠上（更多上半身）就调小，"
                         "比如 0.42 会让头顶留白更紧凑")
    ap.add_argument("--chin", type=float, default=0.02,
                    help="下巴下方留白占人脸高的比例（默认 0.02，几乎不留）")
    ap.add_argument("--lock", action="store_true",
                    help="锁头运镜：头部在画面里固定不动。"
                         "会把平滑降到 0（像素级锁定）—— "
                         "代价是背景会随检测噪声轻微抖动，这是锁头的固有特性")
    ap.add_argument("--smooth-sec", type=float, default=0.06,
                    help="轨迹平滑窗（秒，对称高斯标准差）。默认 0.06 —— "
                         "这个值是实测出来的**取舍点**："
                         "0 = 头钉死但背景带检测噪声抖；"
                         "0.25 = 背景顺了但头漂到 10px。"
                         "注意用的是**非因果**平滑，所以加大它也不会带来滞后")
    ap.add_argument("--aspect", type=float, default=9 / 16,
                    help="裁切宽高比，默认 0.5625（9:16）")
    ap.add_argument("--out-size", help="输出尺寸，如 1080x1920；不填则保持裁切原尺寸")
    ap.add_argument("--auto-band", action="store_true",
                    help="自动检测实际画面范围，把裁切框限制在里面 —— "
                         "避免上下带进模糊填充条。"
                         "竖屏录像（9:16 画布 + 中间摄像头条）建议开启")
    ap.add_argument("--safe-y", help="安全带 TOP:BOTTOM，裁切框必须落在此区间内。"
                                     "用于避开烧进画面的 HUD。"
                                     "例：--safe-y 702:1161（本机录像的实测边界）")
    ap.add_argument("--alpha", type=float, default=None,
                    help="EMA 平滑系数（越小越稳但越迟钝）。"
                         "默认 0.12；--lock 时自动设为 1.0")
    ap.add_argument("--dead-zone", type=float, default=None,
                    help="死区像素（低于此值的位移被忽略，消除微抖）。"
                         "默认 6；--lock 时自动设为 0")
    ap.add_argument("--max-lost", type=int, default=15,
                    help="连续丢检多少帧后回中（默认 15，约 0.5s）")
    ap.add_argument("--fixed-zoom", action="store_true",
                    help="用固定裁切窗口（不随人脸缩放）。默认是**自适应**："
                         "窗口随人脸大小缩放，再统一缩放到固定输出尺寸。"
                         "⚠ 人脸前后移动大时必须用自适应，否则近景会把脸裁断")
    ap.add_argument("--crf", type=int, default=18, help="输出质量")
    ap.add_argument("--no-roi", action="store_true",
                    help="关闭两级检测。默认开启：整图检出率低于 "
                         "--roi-below 时，自动改用姿态定位人脸区域、裁出放大"
                         "再检。人脸离镜头远（脸高占画面 <2%）时整图检测会"
                         "**静默失效**，特写和锁头都会退化成'裁画面中心一刀'，"
                         "这时只有放大才能救回来")
    ap.add_argument("--roi-below", type=float, default=0.35,
                    help="整图检出率低于此值就启用两级检测（默认 0.35）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只分析并打印轨迹统计，不输出视频")
    a = ap.parse_args()

    # ---- 锁头运镜 ----
    # 借鉴 Codex 版 lockhead.py 的实测结论：锁头不是"开关"而是一条取舍曲线。
    #   smooth=0    头钉死，但背景带着检测噪声抖（实测 33.9px/帧）
    #   smooth=0.25 背景顺了，但头漂到 10px
    #   smooth=0.06 折中点（已设为默认）
    #
    # ⚠ 所以 `--lock` **不把平滑清零** —— 清零是那条曲线的极端端点，
    #   不是最优。`--lock` 只是语义标记（表明你要的是"头固定"这个效果），
    #   实际用的是默认的折中值。想走极端可以显式 --smooth-sec 0。
    #
    # ⚠ 这里的平滑是**非因果**的（对称高斯），所以加大平滑**不会带来滞后** ——
    #   这跟 EMA 那种因果滤波完全不同，不能套用"平滑越强越迟钝"的直觉。
    if a.smooth_sec < 0:
        a.smooth_sec = 0.0
    if a.alpha is None:
        a.alpha = 0.12
    if a.dead_zone is None:
        a.dead_zone = 6.0

    if not os.path.exists(a.video):
        print(f"文件不存在: {a.video}")
        return 1
    if not os.path.exists(MODEL_FACE):
        print(f"模型缺失: {MODEL_FACE}")
        return 1

    out = a.out or os.path.splitext(a.video)[0] + "_face.mp4"
    if a.out_size:
        ow, oh = (int(x) for x in a.out_size.lower().split("x"))
    else:
        ow = oh = None

    # ---- 读视频 ----
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("打不开视频")
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 反射填充量：让窗口能伸到原画面之外。35% 足够覆盖实测 177px 的纵向移动
    PAD = int(H * 0.35)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"输入 {os.path.basename(a.video)}  {W}x{H}  {fps:.2f}fps  "
          f"{n_total} 帧 ({n_total/fps:.2f}s)")

    # ---- 两级人脸检测 ----
    # 第一级：整图缩到 640 检测（快）。第二级（自动触发）：整图检出率太低，
    # 说明脸在画面里占比太小，改用姿态定位 + 裁出放大 —— 见 track_faces
    # 上方那段实测数据。
    cap.release()               # 检测与输出各开各的 cap，互不干扰

    raw_boxes, raw_eyes, detected, idx, _ = track_faces(
        a.video, fps, W, H, roi_mode=False)

    if (not a.no_roi) and idx and detected / idx < a.roi_below:
        print(f"  ⚠ 整图检出率仅 {detected / idx * 100:.1f}% —— 多半是人脸在"
              f"画面里占比太小，检测器认不出。启用两级检测重跑…")
        rb2, re2, d2, n2, ph = track_faces(
            a.video, fps, W, H, roi_mode=True)
        if n2 and d2 > detected:
            print(f"  ✓ 两级检测救回：{detected}/{idx} → {d2}/{n2}"
                  f"（{d2 / max(1, n2) * 100:.1f}%）")
            raw_boxes, raw_eyes, detected = rb2, re2, d2
        else:
            print(f"  ✗ 两级检测也没能改善（{d2}/{n2}），保持原结果")

    if idx and detected / idx < 0.2:
        print()
        print("  " + "!" * 64)
        print(f"  ⚠⚠ 只有 {detected}/{idx} 帧检出人脸。人脸特写与锁头运镜都依赖")
        print("     人脸检测，此时它们会**静默失效** —— 输出只是把画面中心裁")
        print("     一刀，不会有任何运镜（这个坑踩过）。")
        print("     建议重录：离镜头近一点、正对镜头，让脸在画面里更大。")
        print("  " + "!" * 64)
        print()

    # ---- 非因果平滑（零滞后）----
    # 把原始轨迹拆成 (cx, cy, bh) 三条独立序列，各自做对称高斯平滑，
    # 再拼回 boxes。缺失帧（None）记 0，由平滑函数内部线性插值补上。
    n_all = len(raw_boxes)
    cx_raw = np.array([0.0 if b is None else b[0] + b[2] / 2 for b in raw_boxes])
    cy_raw = np.array([0.0 if b is None else b[1] + b[3] / 2 for b in raw_boxes])
    bh_raw = np.array([0.0 if b is None else b[3] for b in raw_boxes])
    bw_raw = np.array([0.0 if b is None else b[2] for b in raw_boxes])

    if n_all:
        # 优先用两眼中点当锁定点（更稳）；两眼中点缺失时退回框中心
        eye_x = np.array([0.0 if e is None else e[0] for e in raw_eyes])
        eye_y = np.array([0.0 if e is None else e[1] for e in raw_eyes])
        use_eyes = (eye_x != 0).sum() > n_all * 0.5
        if use_eyes:
            # 两眼中点相对框中心的偏移，也一起平滑，保证一致性
            dx = np.where(eye_x != 0, eye_x - cx_raw, 0.0)
            dy = np.where(eye_y != 0, eye_y - cy_raw, 0.0)
            sx = smooth_track_noncausal(dx, fps, a.smooth_sec)
            sy = smooth_track_noncausal(dy, fps, a.smooth_sec)
        cx_s = smooth_track_noncausal(cx_raw, fps, a.smooth_sec)
        cy_s = smooth_track_noncausal(cy_raw, fps, a.smooth_sec)
        bh_s = smooth_track_noncausal(bh_raw, fps, a.smooth_sec)
        bw_s = smooth_track_noncausal(bw_raw, fps, a.smooth_sec)
        if use_eyes:
            cx_s = cx_s + sx
            cy_s = cy_s + sy

        boxes = []
        for i in range(n_all):
            if bh_s[i] < 1 or bw_s[i] < 1:
                boxes.append(None)
                continue
            boxes.append((cx_s[i] - bw_s[i] / 2, cy_s[i] - bh_s[i] / 2,
                          bw_s[i], bh_s[i]))
        if use_eyes:
            print(f"  锁定点：两眼中点（{int((eye_x != 0).sum())}/{n_all} 帧可用）")
        else:
            print("  锁定点：人脸框中心（两眼中点不可用）")
        print(f"  轨迹平滑：对称高斯 {a.smooth_sec:g}s（非因果，零滞后）"
              f"｜检测命中 {detected}/{n_all} 帧")
    else:
        boxes = []

    print(f"检测到人脸 {detected}/{idx} 帧 ({detected/max(1,idx)*100:.1f}%)")

    # ---- 计算裁切框 ----
    #
    # ⚠⚠ 裁切**尺寸必须全局固定**，只有位置随人脸移动。
    #   踩过的坑：一开始让 ch 跟着人脸大小逐帧变（ch = bh * zoom），
    #   结果 `cv2.VideoWriter` 是**固定尺寸**的，尺寸不匹配的帧被**静默丢弃** ——
    #   5.09s 的素材只输出了 0.23s（3 帧），而且不报任何错。
    #   所以这里先按"人脸高度的中位数"定一个固定尺寸，
    #   之后所有帧都用这个尺寸，只是窗口位置不同。
    safe = None
    if a.safe_y:
        try:
            lo, hi = (int(v) for v in a.safe_y.split(":"))
            safe = (lo, hi)
            print(f"安全带: y ∈ [{lo}, {hi}]（高 {hi-lo}px）")
        except ValueError:
            print(f"--safe-y 格式错误: {a.safe_y}（应为 TOP:BOTTOM）")
            return 1

    # 自动检测实际画面范围（排除模糊填充条）
    if a.auto_band:
        cap0 = cv2.VideoCapture(a.video)
        samples = []
        n0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for k in range(12):
            cap0.set(cv2.CAP_PROP_POS_FRAMES, int(k * n0 / 12))
            ok0, f0 = cap0.read()
            if not ok0:
                break
            b0 = active_band(f0)
            if b0:
                samples.append(b0)
        cap0.release()
        if samples:
            # 取所有采样的**并集**：宁可宽一点，也不要切掉画面
            blo = min(s[0] for s in samples)
            bhi = max(s[1] for s in samples)
            print(f"自动检测实际画面: y ∈ [{blo}, {bhi}]（高 {bhi-blo}px）")
            safe = (max(safe[0], blo), min(safe[1], bhi)) if safe else (blo, bhi)
            print(f"  与安全带取交集后: y ∈ [{safe[0]}, {safe[1]}]")
        else:
            print("自动检测失败，跳过")

    # ---- 定裁切尺寸 ----
    #
    # ⚠⚠ 两种模式，默认自适应：
    #
    #   【自适应】(默认) 每帧的裁切窗口 = 该帧人脸高 × zoom，
    #     然后**缩放到固定输出尺寸**。构图恒定 —— 不管人凑近还是退远，
    #     脸在画面里的大小始终一致。
    #
    #   【固定】( --fixed-zoom ) 所有帧用同一个窗口尺寸，只移动位置。
    #     ⚠ 只适合"人与镜头距离基本不变"的素材。
    #
    # 输出尺寸取"人脸高中位数 × zoom"：这样中位帧是 1:1 无缩放，
    # 近景帧下采样、远景帧上采样，整体最省分辨率。
    bhs = [b[3] for b in boxes if b]
    med_bh = float(np.median(bhs)) if bhs else H * 0.3
    ch = int(med_bh * a.zoom)
    ch = min(ch, H)
    if safe:
        ch = min(ch, safe[1] - safe[0])
    cw = min(int(ch * a.aspect), W)

    # ⚠⚠ 关键：限制"最大裁切窗口"，避免顶到源画面边界。
    #
    #   踩过的坑：人脸在素材里最大到 386px（源高 720），× zoom 3.0 = 1158px
    #   —— 窗口被夹住、无法居中，**头部垂直漂移 138px**（实测），锁头完全失效。
    #   （水平方向因为源够宽，漂移只有 5.8px。）
    #
    #   但上限不能简单地取 `min(H, p90*zoom)` —— 那样会得到 720 = 整个画面高，
    #   窗口**完全无法上下移动**，同样锁不住头。
    #   正确的上限要同时满足两件事：
    #     a) 窗口能装下人脸：  win ≥ 该帧人脸高 × zoom
    #     b) 窗口能居中人脸：  win/2 ≤ 人脸中心到上下边界的最近距离
    #   取 (b) 的约束就能保证任何一帧都能把脸放在窗口中心。
    if bhs:
        # ⚠ 上限要用 **max** 而不是 P90。
        #   用 P90 时，最大的那 10% 帧会被压到上限 —— 窗口变小、脸占比变大、
        #   头部在画面里的位置跟着偏，实测垂直漂移卡在 31.5px 下不去。
        #   有反射填充兜底，窗口可以放心取到"最大人脸也装得下"的尺寸。
        max_win = int(min(a.zoom * max(bhs), H + 2 * PAD))
        if safe:
            max_win = min(max_win, safe[1] - safe[0])
        print(f"  窗口上限 {max_win}px（人脸最大 {int(max(bhs))}px×{a.zoom:g}="
              f"{max(bhs)*a.zoom:.0f}，填充后可用高 {H + 2*PAD}）")
    else:
        max_win = H + 2 * PAD

    mode = "固定窗口" if a.fixed_zoom else "自适应变焦"
    print(f"{mode}：输出 {cw}x{ch}"
          f"（按人脸高中位数 {med_bh:.0f}px × zoom {a.zoom:g}）")
    if bhs:
        lo, hi = int(min(bhs)), int(max(bhs))
        print(f"  人脸高范围 {lo}-{hi}px"
              f"{'  ⚠ 波动大，必须用自适应' if hi > lo * 1.6 else ''}")

    def crop_of(sb, w, h):
        """返回裁切框 (x0, y0, 窗口宽, 窗口高)。

        ⚠ y0 允许为负 —— 因为写入时会用**反射填充过**的画面（上下各 PAD），
          所以窗口可以伸到原画面之外，人脸永远能居中。这是"锁头 + 证件照构图"
          能同时成立的关键：不再需要在"居中"和"不越界"之间二选一。

        水平方向源够宽（1280 vs 需要 ~360），一般不会越界，所以 x0 仍夹到 [0, w-ww]。
        """
        if sb is None:
            cx, cy = w // 2, h // 2
            ww, hh = cw, ch
        else:
            x, y, bw, bh = (int(v) for v in sb)
            want = ch if a.fixed_zoom else int(bh * a.zoom)
            hh = max(60, min(want, max_win))
            ww = min(int(hh * a.aspect), w)
            cx = x + bw // 2
            # ⚠ 构图用 `--face-pos` 直接指定，不再用 headroom 反推。
            #
            #   人脸中心在窗口里的相对位置 = bh/(2*hh) + frac_top，
            #   令它等于 face_pos 即可解出：
            #       frac_top = face_pos - bh/(2*hh)
            #   hh ≈ zoom*bh（绝大多数帧成立），故 bh/(2*hh) ≈ 1/(2*zoom)。
            #
            #   旧写法 `frac_top = headroom/(headroom+1+chin)` 在 zoom=3.0 时
            #   给出 0.495 → 人脸位置 ≈ 0.66，**明显偏下、头顶留白过多**
            #   （成片实测人脸中心在 0.725）。换成 face_pos 后构图可控。
            frac_top = a.face_pos - 1.0 / (2.0 * max(a.zoom, 0.1))
            frac_top = min(max(frac_top, 0.0), 0.95)
            cy = int(y - hh * frac_top + hh / 2)
        x0 = int(max(0, min(cx - ww // 2, w - ww)))
        y0 = int(cy - hh // 2)          # ← 不夹，允许进填充区
        return x0, y0, int(ww), int(hh)

    crops = [crop_of(b, W, H) for b in boxes]

    # ---- 反射填充量：按**实际轨迹**动态确定 ----
    #
    # ⚠⚠ 这里不能写死 `PAD = int(H * 0.35)`。
    #
    #   窗口中心完全由人脸轨迹决定，可能向上超出画面很多。本素材
    #   y0（窗口顶边，原图坐标）低到 **-320**，而 35% 只有 252px，
    #   于是第二遍取 tile 时 `y0 + PAD = -320 + 252 = -68` 是**负数** ——
    #   Python/numpy 的负索引会从数组末尾倒数，切出**空数组**，
    #   紧接着 `cv2.resize` 抛
    #       `(-215:Assertion failed) !ssize.empty()`
    #
    #   ⚠ 这个坑是在修好"两眼中点坐标放大 2 倍"那个 bug **之后**才暴露的：
    #     之前裁切中心被错误地算到画面右下方，y0 min = 34 恰好为正，
    #     越界被错误的位置掩盖了。坐标一修对，窗口上移到正确位置，
    #     y0 变负，问题才现形。
    #     **修一个 bug 常常会暴露出被它掩盖的另一个 bug** ——
    #     所以修完必须重新端到端跑，不能只看那个 bug 的指标变好就收工。
    #
    #   正确做法是按"窗口实际需要伸出去多少"来定：
    #     · 上方需要 max(0, -min(y0))
    #     · 下方需要 max(0, max(y0 + hh) - H)
    #   取两者与基准值的最大者，再留一点切片余量。
    if crops:
        pad_top = max(0, -min(c[1] for c in crops))
        pad_bot = max(0, max(c[1] + c[3] for c in crops) - H)
        if max(pad_top, pad_bot) + 8 > PAD:
            PAD = max(pad_top, pad_bot) + 8
            print(f"  反射填充加大到 {PAD}px"
                  f"（轨迹需要 上{pad_top} / 下{pad_bot}）")

    # ---- 轨迹统计（判断平滑是否有效）----
    cx_t = np.array([c[0] + c[2] / 2 for c in crops], float)
    cy_t = np.array([c[1] + c[3] / 2 for c in crops], float)
    if len(cx_t) > 1:
        jx = np.abs(np.diff(cx_t)).mean()
        jy = np.abs(np.diff(cy_t)).mean()
        print(f"裁切中心逐帧位移: 水平 {jx:.2f}px  垂直 {jy:.2f}px  "
              f"（越小越稳；>3px 说明还在抖）")
        print(f"裁切中心范围: x {cx_t.min():.0f}-{cx_t.max():.0f}  "
              f"y {cy_t.min():.0f}-{cy_t.max():.0f}")
    # ⚠ 这里**绝对不能**写 `cw, ch = crops[0][2], crops[0][3]` ——
    #   那会把上面定好的"输出尺寸"覆盖成第一帧的"裁切窗口尺寸"。
    #   早期设计里两者相等所以看不出问题；改成逐帧自适应窗口后，
    #   这一行会让 VideoWriter 用错误的尺寸（实测 360x641 被改成 297x528），
    #   而且**不报错**，只是画面被静默缩小。
    cw0, ch0 = crops[0][2], crops[0][3]
    print(f"首帧裁切窗口: {cw0}x{ch0}"
          f"（占原画面 {cw0/W*100:.0f}% × {ch0/H*100:.0f}%）"
          f"　输出尺寸: {cw}x{ch}")
    print(f"裁切 y 范围: {min(c[1] for c in crops)} - "
          f"{max(c[1]+c[3] for c in crops)}")

    if a.dry_run:
        print("\n[dry-run] 未输出视频")
        return 0

    # ---- 第二遍：裁切 + 编码 ----
    tmp = tempfile.mkdtemp(prefix="facecrop_")
    raw_path = os.path.join(tmp, "raw.mp4")
    vw = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (cw, ch))
    if not vw.isOpened():
        print("VideoWriter 打不开")
        return 1

    cap = cv2.VideoCapture(a.video)
    i = 0
    bad = 0
    # ⚠⚠ 关键修复：给源画面做**反射填充**，让裁切窗口永远能居中。
    #
    #   问题：窗口必须居中于人脸才能"锁头"，但窗口一大就会超出画面边界，
    #   被夹住后人脸就跑位了。实测人脸中心在 720px 高的画面里纵向移动
    #   183→360px，于是窗口高被限制在 2×183=366px —— 远小于证件照需要的
    #   642px（3×人脸高）。**锁头和构图二选一**。
    #
    #   解法：把画面上下各反射填充 PAD 像素，窗口就能在填充后的画面上
    #   自由居中。反射区在人脸之外（是头发/背景的镜像），观感可接受，
    #   而且**只在人脸贴近边界时才用到**。
    #   （PAD 已在上方"反射填充量动态确定"处按实际轨迹算好，这里不再赋值 ——
    #     曾经这里写死 `PAD = int(H * 0.35)`，把上面动态算的结果覆盖掉了）
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if PAD > 0:
            frame = cv2.copyMakeBorder(frame, PAD, PAD, 0, 0,
                                       cv2.BORDER_REFLECT_101)
        x0, y0, ww, hh = crops[min(i, len(crops) - 1)]
        ty = y0 + PAD
        # ⚠⚠ 显式越界检查。**不要**依赖 numpy 的切片"宽容"行为：
        #   负起点会被当成"从末尾倒数"，切出空数组或错位的区域，
        #   而且**不报错**，一直到 cv2.resize 才抛
        #   `!ssize.empty()`，报错信息完全指不到真正的根因。
        #   宁可在这里明确失败，也不要让它变成一个看不懂的下游报错。
        if ty < 0 or ty + hh > frame.shape[0] or x0 < 0 or x0 + ww > frame.shape[1]:
            print(f"错误：第 {i} 帧裁切窗口越界，反射填充量 PAD={PAD} 不足")
            print(f"      x0={x0} y0={y0} ww={ww} hh={hh}  "
                  f"填充后画面 {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
            vw.release()
            return 1
        tile = frame[ty:ty + hh, x0:x0 + ww]
        if tile.size == 0:
            print(f"错误：第 {i} 帧切出空图（x0={x0} y0={y0} ww={ww} hh={hh}）")
            cap.release()
            vw.release()
            return 1
        # ⚠ 自适应模式下每帧窗口尺寸不同，必须统一缩放到固定输出尺寸。
        #   VideoWriter 是固定尺寸的，尺寸不符的帧会被**静默丢弃**
        #   （实测 5.09s 只输出 0.23s，且不报错）。
        #   缩小时用 INTER_AREA（抗锯齿），放大时用 INTER_LANCZOS4（更锐）。
        if tile.shape[1] != cw or tile.shape[0] != ch:
            interp = (cv2.INTER_AREA if tile.shape[1] > cw
                      else cv2.INTER_LANCZOS4)
            tile = cv2.resize(tile, (cw, ch), interpolation=interp)
            bad += 1
        vw.write(tile)
        i += 1
    cap.release()
    vw.release()
    if bad:
        print(f"  已缩放 {bad}/{i} 帧到输出尺寸 {cw}x{ch}")

    # ---- 用 ffmpeg 转码（OpenCV 的 mp4v 兼容性差，且要统一像素格式）----
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", raw_path]
    if ow and oh:
        cmd += ["-vf", f"scale={ow}:{oh}:flags=lanczos"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
            "-pix_fmt", "yuv420p", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"ffmpeg 转码失败: {(r.stderr or '')[:200]}")
        return 1

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n完成 -> {out}  {os.path.getsize(out)/1024/1024:.2f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
