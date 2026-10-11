"""动作检测（action_detect.py）
====================================================================
从素材里自动找出"有戏"的动作片段，输出带时间码的事件列表（JSON）。

为什么做这个：
    slop 的"难绷"之处在于 —— **把无意义的动作剪得很燃**。
    纯按拍点切是"节奏对但内容随机"；如果切点正好落在"举手""喝一口""傻笑"
    这些动作上，荒诞感会成倍放大。
    所以需要先把动作找出来，再让剪辑去卡这些动作。

检测源（两条，互补）：
    · 人脸关键点 478 点  -> 脸的位置/大小/朝向  -> 抬头/低头/转头/凑近/后退
    · 表情系数 52 个     -> 嘴/眼/眉的形变      -> 微笑/张嘴/眨眼/嘟嘴/挑眉
    · 姿态关键点 33 点   -> 肩肘腕的几何关系    -> 举手/双手举/摸头/手到嘴/耸肩

设计要点：

  1. **特征要归一化到"人脸高度"为单位**。
     否则人一凑近，所有像素距离都变大，检测器全乱。
     比如"手腕到鼻子的距离"要除以脸高，才是尺度无关的量。

  2. **事件必须做时间平滑 + 最小时长**。
     逐帧的原始信号抖动很大（检测噪声），不加平滑会切出一堆 1-2 帧的假事件。
     这里先做滑动平均，再要求持续 ≥ min_dur 才算一个事件。

  3. **同一动作要冷却**，避免一次举手被切成十几个事件。

  4. **丢失检测要"保持"而不是清零**。
     MediaPipe 偶发丢帧，若直接把特征设 0，会误判成"动作结束"。
     这里沿用上一个有效值，超过阈值才真正置空。

用法：
    python action_detect.py 素材.mp4                    # 打印事件列表
    python action_detect.py 素材.mp4 -o 动作.json        # 存 JSON
    python action_detect.py 素材.mp4 --list              # 只列出可用动作类型
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FACE = os.path.join(BASE, "models", "face_landmarker.task")
MODEL_POSE = os.path.join(BASE, "models", "pose_landmarker_lite.task")
DETECT_WIDTH = 640

import mediapipe as mp                                     # noqa: E402
from mediapipe.tasks import python as mp_python            # noqa: E402
from mediapipe.tasks.python import vision                  # noqa: E402

# 两级检测：整图检不到脸时用姿态定位 + 裁出放大（见 face_crop 里的实测）。
# ⚠ 这里特别要命：本脚本的**所有特征**都挂在人脸结果上 ——
#   人脸丢了 `rec` 就是 None，连手腕/姿态类信号也会被写成 0。
#   于是一段"脸离得远"的素材会得到**全零的特征**，选段时没有任何峰值
#   可用，只能走钳位兜底（表现：同一帧被重复取用）。所以这个兜底不是
#   锦上添花，而是让选段逻辑有东西可用的前提。
from face_crop import pose_face_roi, to_mp_image            # noqa: E402

# 姿态关键点索引（MediaPipe Pose 33 点标准）
NOSE, L_SH, R_SH = 0, 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24


def smooth(x, win):
    """滑动平均。边界用 edge padding，避免首尾被拉向 0。"""
    if win <= 1 or len(x) < win:
        return x
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, k, mode="valid")[:len(x)]


def runs_above(sig, thr, min_len, cooldown, valid=None):
    """找出 sig > thr 的**连续有效**段，过滤掉太短的，并做冷却。

    返回 [(start, end, peak)]，end 为开区间。

    ⚠⚠ `valid` 必须是**逐帧掩码**（布尔数组），不能用"窗口内有效率 ≥ 某比例"。
      这是踩过的坑，值得记清楚：

        负阈值信号 `wrist_head = -dmin`（dmin = 手腕到鼻子的距离 / 脸高）
        真值**必然 ≤ 0**（dmin ≥ 0）。而检测丢失的帧被补成 0.0，
        0.0 > -0.85 —— 于是"这一帧没检测到手"反而被判成"手在头上"。

        旧实现用的是 `valid[i:j].mean() >= 0.5`，**允许最多 49% 的无效帧
        混进区间**。于是 5~39 帧的窗口里只要有 1 帧是补零的，
        `peak = max(sig[i:j])` 就会被拉成 0.0。
        实测症状：`摸头 5.63-6.94` 的 peak = 0.0 —— 而任何真实测量都
        不可能是 0.0，**只可能来自补零**。
        （旧注释声称"加了有效率检查后修好了"，其实只挡住了
          "整个窗口过半无效"的情况，单帧污染照样漏过去。）

      改成掩码后，无效帧直接**截断**区间，0.0 永远进不了 peak 的计算。
      代价：真实动作中间若丢了检测，会被切成两个事件 —— 这是可接受的，
      因为 `extract()` 里已经有"丢失 ≤8 帧沿用上一帧值"的保持逻辑，
      短暂丢失本来就不会产生无效帧。
    """
    if valid is None:
        valid = np.ones(len(sig), dtype=bool)
    m = (sig > thr) & valid
    out = []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i >= min_len:
                peak = float(np.max(sig[i:j]))
                if out and i - out[-1][1] < cooldown:
                    if peak > out[-1][2]:
                        out[-1] = (out[-1][0], j, peak)
                else:
                    out.append((i, j, peak))
            i = j
        else:
            i += 1
    return out


def extract(video, verbose=True):
    """逐帧提取特征。返回 dict of arrays + fps + 尺寸。"""
    face_opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_FACE),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True,           # ← 表情检测的基础
        output_facial_transformation_matrixes=False)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_POSE),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    flm = vision.FaceLandmarker.create_from_options(face_opts)
    plm = vision.PoseLandmarker.create_from_options(pose_opts)
    flm_roi = None          # 惰性创建：只有整图检不到脸时才需要它
    roi_try = roi_hit = 0

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"打不开: {video}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    F = {"cx": [], "cy": [], "fh": [], "nose_dx": [], "nose_dy": [],
         "smile": [], "jaw": [], "blink": [], "brow": [], "pucker": [],
         "wrist_above": [], "wrist_head": [], "wrist_mouth": [],
         "shrug": [], "both_up": [], "ok": [], "pose_ok": []}
    last = None
    lost = 0
    pose_lost = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        sc = DETECT_WIDTH / fw
        det = cv2.resize(frame, (DETECT_WIDTH, int(fh * sc)))
        rgb = cv2.cvtColor(det, cv2.COLOR_BGR2RGB)
        ts = int(i * 1000 / fps)
        fr = flm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        pr = plm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        # ---- 两级检测的兜底：整图检不到，就用姿态定位那块裁出来放大重检 ----
        # ⚠ 必须用**独立实例**：MediaPipe 要求同一实例的时间戳严格递增，
        #   同一帧对同一个 landmarker 调两次同 ts 会抛。
        # 坐标一律换算回"整帧归一化"，下游（脸高 fh 做尺度）才不受路径影响。
        sx, sy, ox, oy = 1.0, 1.0, 0.0, 0.0
        if not (fr and fr.face_landmarks):
            roi = pose_face_roi(pr, W, H)
            if roi:
                roi_try += 1
                if flm_roi is None:
                    flm_roi = vision.FaceLandmarker.create_from_options(face_opts)
                rx, ry, rw, rh = roi
                tile = frame[ry:ry + rh, rx:rx + rw]
                if tile.size:
                    dw = DETECT_WIDTH
                    dh = max(1, int(round(rh * dw / float(rw))))
                    interp = (cv2.INTER_AREA if dw < rw else cv2.INTER_CUBIC)
                    t2 = cv2.resize(tile, (dw, dh), interpolation=interp)
                    r2 = flm_roi.detect_for_video(to_mp_image(t2), ts)
                    if r2 and r2.face_landmarks:
                        fr = r2
                        roi_hit += 1
                        k = rw / float(dw)      # 检测图像素 -> 原图像素
                        sx, sy = k * dw / W, k * dh / H
                        ox, oy = rx / W, ry / H

        rec = None
        if fr and fr.face_landmarks:
            lm = fr.face_landmarks[0]
            # ⚠ 把"检测图归一化坐标"映射成"整帧归一化坐标"，两条路径共用：
            #   整图路径 = 恒等 (sx,sy,ox,oy)=(1,1,0,0)
            #   ROI 路径  = 归一化 --×dw/dh--> 检测图像素 --×k--> ROI内像素
            #               --+rx/ry--> 整帧，再 ÷W/÷H
            #   下游用 fh（脸高）当尺度去量手腕距离，所以必须是整帧语义。
            xs = [p.x * sx + ox for p in lm]
            ys = [p.y * sy + oy for p in lm]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            bw, bh = (x1 - x0), (y1 - y0)
            if bh > 1e-6:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                # 鼻尖关键点索引 1（MediaPipe 标准）
                nx, ny = lm[1].x * sx + ox, lm[1].y * sy + oy
                bs = {}
                if fr.face_blendshapes:
                    bs = {c.category_name: c.score
                          for c in fr.face_blendshapes[0]}
                rec = {
                    "cx": cx, "cy": cy, "fh": bh,
                    # 鼻子相对脸中心的偏移，除以脸宽 -> 尺度无关的转头量
                    "nose_dx": (nx - cx) / max(bw, 1e-6),
                    "nose_dy": (ny - cy) / max(bh, 1e-6),
                    "smile": (bs.get("mouthSmileLeft", 0)
                              + bs.get("mouthSmileRight", 0)) / 2,
                    "jaw": bs.get("jawOpen", 0),
                    "blink": (bs.get("eyeBlinkLeft", 0)
                              + bs.get("eyeBlinkRight", 0)) / 2,
                    "brow": bs.get("browInnerUp", 0),
                    "pucker": bs.get("mouthPucker", 0),
                }

        # 姿态（与脸独立，脸丢了也可能有姿态）
        pose_valid = 0.0
        if pr and pr.pose_landmarks:
            pose_valid = 1.0
            pm = pr.pose_landmarks[0]
            def P(k):
                return np.array([pm[k].x, pm[k].y])
            sh = (P(L_SH) + P(R_SH)) / 2
            fh_ = rec["fh"] if rec else 0.25      # 无脸时用经验值
            # 肩到腕的"上举量"：正值表示手腕高于肩膀
            up_l = (sh[1] - P(L_WR)[1]) / max(fh_, 1e-6)
            up_r = (sh[1] - P(R_WR)[1]) / max(fh_, 1e-6)
            head = P(NOSE)
            d_l = np.linalg.norm(P(L_WR) - head) / max(fh_, 1e-6)
            d_r = np.linalg.norm(P(R_WR) - head) / max(fh_, 1e-6)
            dmin = min(d_l, d_r)
            # 手腕到"嘴"的估计位置（鼻下 0.25 脸高）
            mouth = head + np.array([0, 0.25 * fh_])
            m_l = np.linalg.norm(P(L_WR) - mouth) / max(fh_, 1e-6)
            m_r = np.linalg.norm(P(R_WR) - mouth) / max(fh_, 1e-6)
            if rec is not None:
                rec["wrist_above"] = max(up_l, up_r)
                rec["both_up"] = min(up_l, up_r)
                rec["wrist_head"] = -dmin      # 越近值越大
                rec["wrist_mouth"] = -min(m_l, m_r)
                # 耸肩：肩膀相对鼻子的竖直距离变小
                rec["shrug"] = -(sh[1] - head[1]) / max(fh_, 1e-6)

        if rec is not None:
            lost = 0
            last = rec
        else:
            lost += 1
            if lost <= 8 and last is not None:
                rec = dict(last)               # 丢帧时保持，而不是清零
            else:
                rec = None

        # 姿态的丢失要单独计数（它和人脸是两次独立检测）
        if pose_valid > 0:
            pose_lost = 0
        else:
            pose_lost += 1
            if pose_lost > 8:
                # 丢太久：把姿态类信号置成"必然不触发"的哨兵值
                # ⚠ 不能用 0 —— 负阈值信号（wrist_head 等）会被 0 误触发
                if rec is not None:
                    rec = dict(rec)
                    rec.update(wrist_above=-9, both_up=-9, wrist_head=-9,
                               wrist_mouth=-9, shrug=-9)
                pose_valid = 0.0

        if rec is None:
            for k in F:
                if k not in ("ok", "pose_ok"):
                    F[k].append(0.0)
            F["ok"].append(0.0)
            F["pose_ok"].append(0.0)
        else:
            for k in F:
                if k in ("ok", "pose_ok"):
                    continue
                F[k].append(float(rec.get(k, 0.0)))
            F["ok"].append(1.0)
            F["pose_ok"].append(pose_valid)
        i += 1

    cap.release()
    flm.close()
    plm.close()
    if flm_roi is not None:
        flm_roi.close()
    if verbose:
        n = len(F["ok"])
        print(f"提取完成: {n} 帧 / {n/max(fps,1):.1f}s   人脸命中 "
              f"{sum(F['ok'])/max(n,1)*100:.1f}%")
        if roi_try:
            print(f"  （其中整图失败的 {roi_try} 帧走了姿态定位 ROI 重检，"
                  f"救回 {roi_hit} 帧）")
        if sum(F["ok"]) / max(n, 1) < 0.2:
            print("  ⚠⚠ 人脸命中过低 —— 表情类动作（微笑/嘟嘴/眨眼/挑眉）会"
                  "几乎全部丢失，")
            print("     而**所有特征都挂在人脸结果上**，丢了脸连手腕类也会被"
                  "置 0，")
            print("     结果就是「没有峰值可选」，选段只能退回钳位。"
                  "建议重录：离镜头近一点。")
    return {"F": {k: np.asarray(v) for k, v in F.items()},
            "fps": fps, "W": W, "H": H, "n": len(F["ok"])}


# ------------------------------------------------------------------ 动作定义
# 每个动作：需要的特征、阈值、平滑窗、最小时长、冷却
#
# `rel=True` 表示"阈值作用在**相对本人的基线增量**上"，而不是绝对值。
#
#   为什么表情类必须做基线校正：
#     表情 blendshape 的绝对值**因人而异**。有人中性表情下
#     `mouthSmile*` 就有 0.4~0.5（嘴角天然上扬），于是固定阈值 0.45
#     会把「整段静止的脸」全程判成微笑。
#     实测症状：`微笑 0.03-4.72`（一个事件持续 4.69 秒）——
#     人不可能连续微笑 4.7 秒不动，这是**基线偏高导致的误判**，
#     不是动作本身。
#
#   反之**不**对几何类信号做基线校正：
#     `wrist_above`（肩到腕的上举量）、`wrist_head`（手腕到鼻的距离）
#     这些量以「脸高」为单位，物理含义明确、跨人可比，
#     而且现有阈值已经按绝对值调好。若再减基线会把阈值语义搞乱
#     （比如 wrist_head 的基线是 -1.2，减完基线后 -0.85 的阈值
#      会变成"几乎恒真"，反而全线误触发）。
ACTIONS = {
    "举手":       dict(sig="wrist_above", thr=0.15, smooth=5, min_dur=0.25, cd=0.5),
    "双手举":     dict(sig="both_up",     thr=0.00, smooth=5, min_dur=0.25, cd=0.5),
    "摸头":       dict(sig="wrist_head",  thr=-0.85, smooth=5, min_dur=0.20, cd=0.4),
    "手到嘴边":   dict(sig="wrist_mouth", thr=-0.75, smooth=5, min_dur=0.20, cd=0.4),
    # ↓ 以下 4 个动作**必须**用 rel（相对基线），否则阈值与信号符号不符
    "耸肩":       dict(sig="shrug",  thr=0.70, smooth=5, min_dur=0.25, cd=0.5, rel=True),
    "微笑":       dict(sig="smile",  thr=0.55, smooth=7, min_dur=0.35, cd=0.6, rel=True),
    "张嘴":       dict(sig="jaw",    thr=0.08, smooth=5, min_dur=0.15, cd=0.4, rel=True),
    "眨眼":       dict(sig="blink",  thr=0.40, smooth=3, min_dur=0.03, cd=0.25, rel=True),
    "挑眉":       dict(sig="brow",   thr=0.25, smooth=5, min_dur=0.20, cd=0.5, rel=True),
    "嘟嘴":       dict(sig="pucker", thr=0.30, smooth=5, min_dur=0.20, cd=0.5, rel=True),
}

# 姿态类信号：它们的值只有在"**人脸和姿态同时检测到**"时才是真实的。
#
# ⚠ 这里曾经用一个隐蔽的错误条件 —— 只查 `pose_ok`。
#   但 `extract()` 里手腕相关特征的写入被包在 `if rec is not None:` 里
#   （rec 来自人脸检测）。于是"人脸丢了、姿态还在"的帧：
#       pose_ok = 1.0（姿态确实检测到了）
#       但 wrist_* 一个字都没写 -> 被 `float(rec.get(k, 0.0))` 补成 0.0
#   这一帧就带着"假的 0.0"通过了姿态有效性检查。
#   → 正确的有效性是 **两个检测都要有**。
POSE_SIGS = {"wrist_above", "both_up", "wrist_head", "wrist_mouth", "shrug"}

# 派生动作：靠"变化量"而不是"绝对值"判定，单独实现
DERIVED = ["抬头", "低头", "转头", "凑近镜头", "后退"]


def validity(F, sig_name):
    """该信号的**逐帧有效掩码**（布尔数组）。

    · 姿态类（wrist_* / shrug）：要求**人脸 + 姿态同时检测到**。
      只查 pose_ok 是错的 —— `extract()` 里手腕特征的写入被人脸存在性包着，
      "人脸丢了、姿态还在"的帧 pose_ok=1 但值是补零的 0.0（详见 POSE_SIGS 说明）。
    · 人脸类（cy / nose_dx / fh）+ 表情类：只要人脸检测到。
    """
    ok = F["ok"] > 0
    if sig_name in POSE_SIGS:
        return ok & (F["pose_ok"] > 0)
    return ok


def robust_baseline(sig, valid):
    """稳健基线：有效帧的中位数。

    用中位数而不是均值 —— 中位数抗离群，几次大笑/张嘴不会把
    "这个人的静止表情基线"抬高。这正是基线校正要的东西。
    """
    v = sig[valid]
    return float(np.median(v)) if v.size else 0.0


def mk_events(name, sig, runs, fps):
    """把 runs_above 的区间转成事件字典，并补上**峰值时刻 peak_t**。

    为什么单独给 peak_t：
      有些动作天然是"状态"而不是"动作" —— 一次微笑可以持续 4.7 秒。
      这种区间当素材用没意义（成片一镜才 0.89s），
      但它的**峰值时刻**恰好就是切点该压住的那一帧。
      所以区间给人工筛选用，peak_t 给剪辑器做吸附用，两者都要。
    """
    out = []
    for a, b, peak in runs:
        pidx = a + int(np.argmax(sig[a:b]))
        out.append(dict(action=name,
                        start=round(a / fps, 3), end=round(b / fps, 3),
                        peak=round(peak, 3), peak_t=round(pidx / fps, 3),
                        frames=[a, b]))
    return out


def detect(data, min_dur_scale=1.0, verbose=True):
    """跑所有检测器，返回事件列表（按时间排序）。"""
    F, fps = data["F"], data["fps"]
    events = []

    for name, cfg in ACTIONS.items():
        raw = F[cfg["sig"]]
        if not np.any(raw):
            continue
        valid = validity(F, cfg["sig"])
        sig = raw
        if cfg.get("rel"):
            sig = raw - robust_baseline(raw, valid)
        s = smooth(sig, cfg["smooth"])
        ml = max(2, int(cfg["min_dur"] * min_dur_scale * fps))
        cl = int(cfg["cd"] * fps)
        events.extend(mk_events(name, s, runs_above(s, cfg["thr"], ml, cl, valid),
                                fps))

    # ---- 派生：头部姿态与距离变化 ----
    # 用"相对中位数的偏移"判定，而不是绝对阈值 —— 因为每个人的基准姿态不同
    # ⚠ 基线也要只取**有效帧**。旧写法用 `sig != 0` 排除无效帧，
    #   但那是个不可靠的判据：真实值也可能恰好是 0（比如抬头时 cy 不变），
    #   而"无效"的帧恰恰就是被补成 0 的那些 —— 判据本身分不开两者。
    def deriv(sig_name, name, thr, direction, min_dur=0.25, win=9):
        sig = F[sig_name]
        valid = validity(F, sig_name)
        base = robust_baseline(sig, valid)
        d = smooth(sig - base, win) * direction
        ml = max(2, int(min_dur * fps))
        events.extend(mk_events(name, d, runs_above(d, thr, ml, int(0.4 * fps),
                                                    valid), fps))

    # cy 是归一化坐标（0=顶），增大 = 低头
    deriv("cy", "低头", 0.06, +1)
    deriv("cy", "抬头", 0.06, -1)
    deriv("nose_dx", "转头", 0.10, +1)
    deriv("nose_dx", "转头", 0.10, -1)
    # 脸高变化率 -> 凑近/后退（用相对变化，避免绝对值影响）
    fh = F["fh"].copy()
    ok = F["ok"] > 0
    base = np.median(fh[ok]) if np.any(ok) else 1.0
    rel = smooth(fh / max(base, 1e-6) - 1.0, 7)
    ml = max(2, int(0.25 * fps))
    events.extend(mk_events("凑近镜头", rel,
                            runs_above(rel, 0.18, ml, int(0.5 * fps), ok), fps))
    events.extend(mk_events("后退", -rel,
                            runs_above(-rel, 0.18, ml, int(0.5 * fps), ok), fps))

    events.sort(key=lambda e: (e["start"], e["action"]))
    if verbose:
        print(f"检出 {len(events)} 个动作事件")
    return events


# ------------------------------------------------------------------ 缓存 / 统计
#
# ⚠⚠ 缓存必须绑定"它来自哪个视频文件"（2026-09-24 加，踩过一次严重事故）
#
#   原来的缓存是**固定路径 + 不校验来源**：`load_cache` 直接返回 npz 里的
#   特征，从不管它当初是从哪个视频提取的。而流水线用的缓存路径是固定的
#   `output/_pipeline/动作分析/features.npz`，于是：
#
#       第一次出片（素材 A）  -> 写入 A 的特征
#       第二次出片（素材 B）  -> **静默复用 A 的特征**
#
#   实测现场：素材 n=434 / 人脸命中 0%，而缓存里是 n=429 / 命中 88.3%
#   （另一段素材）。后果是锚点位置全部对不上真实动作 -> 内容感知选段
#   找不到可用峰值 -> 大面积走钳位兜底 -> 同一帧被反复取用
#   （用户看到的就是"同一个动作连续回放四次"）。
#
#   这种错的可怕之处在于**完全不报错**：事件列表照样输出、切片照样成功、
#   成片照样渲染，只是内容与素材无关。
#
#   所以：存的时候记下源文件指纹，读的时候比对，不符就视为没有缓存。
def src_fingerprint(video):
    """源视频指纹：绝对路径 + 文件大小 + mtime。"""
    ap = os.path.abspath(video)
    try:
        st = os.stat(ap)
        return "%s|%d|%d" % (ap, st.st_size, int(st.st_mtime))
    except OSError:
        return "%s|missing" % ap


def save_cache(path, data, video=None):
    """把逐帧特征存成 npz。

    为什么值得存：特征提取是这整条链路里**唯一昂贵的一步**
    （MediaPipe 逐帧跑人脸+姿态，25s 素材要约 1 分钟）。
    而调阈值、改动作定义、加新动作都**只需要特征**，
    不该每次都重跑检测。存下来之后调参是秒级的。
    """
    d = {f"F_{k}": v for k, v in data["F"].items()}
    d.update(fps=data["fps"], W=data["W"], H=data["H"], n=data["n"])
    if video is not None:
        d["src_fp"] = src_fingerprint(video)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **d)
    print(f"特征缓存 -> {path}（{os.path.getsize(path)/1024:.0f} KB）")


def load_cache(path, video=None):
    """读缓存。指纹不符（或旧格式没有指纹）时返回 None，强制重新提取。"""
    z = np.load(path, allow_pickle=False)
    F = {k[2:]: z[k] for k in z.files if k.startswith("F_")}
    data = {"F": F, "fps": float(z["fps"]), "W": int(z["W"]),
            "H": int(z["H"]), "n": int(z["n"])}
    if video is None:
        return data
    want = src_fingerprint(video)
    got = str(z["src_fp"]) if "src_fp" in z.files else None
    if got != want:
        print(f"  ⚠ 缓存来源不符，丢弃并重新提取：")
        print(f"      缓存属于: {got or '（旧格式，未记录来源）'}")
        print(f"      当前素材: {want}")
        return None
    return data


def print_stats(data):
    """打印各信号的分位数，用于**有依据地**定阈值。"""
    F = data["F"]
    order = ["wrist_above", "both_up", "wrist_head", "wrist_mouth", "shrug",
             "smile", "jaw", "blink", "brow", "pucker",
             "cy", "nose_dx", "fh"]
    print()
    print("=" * 78)
    print("信号分布（只统计有效帧；`增量` = 相对中位数的分位提升，供 rel 阈值参考）")
    print("=" * 78)
    print(f"{'信号':<13}{'有效%':>7}{'中位':>9}{'p90':>9}{'p99':>9}"
          f"{'最大':>9}{'增量p90':>9}{'增量p99':>9}")
    print("-" * 78)
    for k in order:
        if k not in F:
            continue
        sig = F[k]
        valid = validity(F, k)
        v = sig[valid]
        if v.size == 0:
            print(f"{k:<13}{0.0:>7.1f}  (无有效帧)")
            continue
        med = float(np.median(v))
        p90, p99 = float(np.percentile(v, 90)), float(np.percentile(v, 99))
        print(f"{k:<13}{valid.mean()*100:>7.1f}{med:>9.3f}{p90:>9.3f}"
              f"{p99:>9.3f}{float(v.max()):>9.3f}"
              f"{p90-med:>9.3f}{p99-med:>9.3f}")
    print()
    print("  · rel=True 的动作（微笑/张嘴/眨眼/挑眉/嘟嘴）阈值应取在")
    print("    `增量p90` 与 `增量p99` 之间 —— 低于 p90 会一直触发，")
    print("    高于 p99 则只有最强的几次能被检出。")


def main():
    ap = argparse.ArgumentParser(description="动作检测")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", help="输出 JSON 路径")
    ap.add_argument("--list", action="store_true", help="只列出可用动作类型")
    ap.add_argument("--min-dur-scale", type=float, default=1.0,
                    help="最小时长倍率（调小=更敏感，默认 1.0）")
    ap.add_argument("--cache",
                    default=os.path.join(BASE, "output", "动作分析",
                                         "features.npz"),
                    help="特征缓存路径（默认 output/动作分析/features.npz）")
    ap.add_argument("--refresh", action="store_true",
                    help="强制重新提取特征（忽略已有缓存）")
    ap.add_argument("--stats", action="store_true", help="打印信号分布")
    a = ap.parse_args()

    if a.list:
        print("可检测动作：")
        for k, c in ACTIONS.items():
            tag = "（相对基线）" if c.get("rel") else ""
            print(f"  {k:10} 阈值 {c['thr']:>6.2f}{tag}")
        for k in DERIVED:
            print(f"  {k:10} 派生")
        return 0

    if not os.path.exists(a.video):
        print(f"文件不存在: {a.video}")
        return 1

    data = None
    if a.cache and os.path.exists(a.cache) and not a.refresh:
        data = load_cache(a.cache, a.video)
    if data is not None:
        print(f"读特征缓存 {a.cache}"
              f"（{data['n']} 帧 / {data['n']/max(data['fps'],1):.1f}s）")
    else:
        data = extract(a.video)
        if data is None:
            return 1
        if a.cache:
            save_cache(a.cache, data, a.video)

    if a.stats:
        print_stats(data)

    events = detect(data, a.min_dur_scale)

    print(f"\n{'动作':<10}{'起':>8}{'止':>8}{'时长':>8}{'强度':>8}")
    print("-" * 46)
    from collections import Counter
    cnt = Counter()
    for e in events:
        cnt[e["action"]] += 1
        print(f"{e['action']:<10}{e['start']:8.2f}{e['end']:8.2f}"
              f"{e['end']-e['start']:8.2f}{e['peak']:8.2f}")
    print("\n按类型统计：")
    for k, v in cnt.most_common():
        print(f"  {k:10} {v} 次")

    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(d, exist_ok=True)      # 目录不存在时自动建，别让它报错
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"video": os.path.abspath(a.video),
                       "fps": data["fps"], "size": [data["W"], data["H"]],
                       "duration": data["n"] / data["fps"],
                       # 把本次使用的检测配置一并写进结果 ——
                       # 否则"同一份素材、不同阈值"产出的 JSON 长得一样，
                       # 事后无法分辨是谁跑的（调阈值时特别容易混）
                       "actions": {k: dict(v) for k, v in ACTIONS.items()},
                       "derived": DERIVED,
                       "events": events}, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
