# -*- coding: utf-8 -*-
"""量"这段素材能不能用锁头"（probe_headroom.py）
====================================================================
**为什么需要这个脚本**：`probe_lock.py` 量的是**成片**上的漂移，而且
`CASES` 是写死的文件清单；它没法回答录制完成后最该先问的那句话 ——
「这段素材的头部活动范围，锁头撑得住吗」。等出片跑完几十分钟再发现
撑不住，就白跑了。

判据（来自 README / 交接文档）：
    · 拍摄时四周留 **15%~20%** 余量 —— 这是给锁头补偿留的空间
    · 锁头的放大倍率下界 k ≈ (W/2) / (W/2 - max|hx - W/2|)，
      k 越大有效分辨率越低（k=4 → 1080 宽只剩 270px）

两种用法：
    # 1) 素材体检（录制完立刻跑）—— 直接吃视频，内部走两级人脸检测
    python probe_headroom.py output/录制_raw_0924_082604.mp4

    # 2) 锁定验收（出片后跑）—— 量"两眼中点"漂移，与锁定点口径一致
    python probe_headroom.py output/成片_人脸特写_v2.mp4 --eyes-only

    # 也可直接吃特征缓存（比重新检测快得多，流水线本来就会产出）
    python probe_headroom.py output/_pipeline/动作分析/features_xxxx.npz

⚠ **验收指标必须与锁定点同口径**：`face_crop --lock` 锁的是**两眼中点**
（眼眶骨骼位置比下颌/额头的检测抖动稳得多，见 README）。若用"人脸框中心"
去验它，量到的是「框中心 vs 两眼中点」的相对摆动被裁切放大后的结果 ——
实测同一份输出：框中心 水平 std 38.7px（判"不合格"），
两眼中点 水平 std 3.2px（判"合格"）。**指标选错，结论会整个反过来。**
"""
import argparse
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def from_cache(path):
    z = np.load(path, allow_pickle=False)
    F = {k[2:]: z[k] for k in z.files if k.startswith("F_")}
    W, H = int(z["W"]), int(z["H"])
    src = str(z["src_fp"]) if "src_fp" in z.files else "（未记录来源）"
    ok = F["fh"] > 1e-6
    # 缓存里的 cx/cy 是"人脸框中心"（整帧归一化）
    return (F["cx"][ok] * W, F["cy"][ok] * H, F["fh"][ok] * H,
            None, W, H, src, float(ok.mean()), len(F["fh"]))


def from_video(path, roi_below=0.35, force_roi=None):
    import cv2
    import face_crop as fc
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if force_roi is None:
        print("  [第 1 遍] 整图检测（快）…")
        boxes, eyes, hit, tot, _ = fc.track_faces(path, fps, W, H, False,
                                                  verbose=True)
        # ⚠ 第 1 遍的结果必须留着复用：整图这一遍本来就要跑几十秒，
        #   如果判完"够用"还再跑一遍同样的检测，等于白花一倍时间
        #   （实测 434 帧的素材多花约 50s）。
        if tot and hit / tot >= roi_below:
            roi = False
        else:
            print(f"  整图检出率 {hit/max(1,tot)*100:.1f}% < "
                  f"{roi_below*100:.0f}% —— 人脸占比太小，改用两级检测重跑…")
            roi = True
            boxes, eyes, hit, tot, _ = fc.track_faces(path, fps, W, H, True,
                                                      verbose=True)
    else:
        boxes, eyes, hit, tot, _ = fc.track_faces(path, fps, W, H, force_roi,
                                                  verbose=True)

    cx, cy, fh, ex, ey = [], [], [], [], []
    for b, e in zip(boxes, eyes):
        if b:
            cx.append(b[0] + b[2] / 2)
            cy.append(b[1] + b[3] / 2)
            fh.append(b[3])
        if e:
            ex.append(e[0])
            ey.append(e[1])
    return (np.asarray(cx), np.asarray(cy), np.asarray(fh),
            (np.asarray(ex), np.asarray(ey)) if ex else None,
            W, H, path, hit / max(1, tot), tot)


def zoom_lower_bound(vals, span):
    """锁头放大倍率下界：锚点在画面中心，画布四角反解回源图必须仍在源内。"""
    half = span / 2.0
    need = max(half - vals.min(), vals.max() - half)
    if need >= half:
        return float("inf")
    return half / max(half - need, 1e-6)


def report(tag, cx, cy, fh, eyes, W, H, src, rate, n):
    print()
    print("=" * 78)
    print(f"头部活动范围体检 —— {os.path.basename(str(src))}")
    print("=" * 78)
    print(f"素材 {W}x{H}  {n} 帧  人脸命中 {rate*100:.1f}%")
    if cx.size < 10:
        print("有效帧不足，无法评估")
        return
    print(f"人脸框高  {fh.min():.0f} ~ {fh.max():.0f}px"
          f"（占画面高 {fh.min()/H*100:.1f}% ~ {fh.max()/H*100:.1f}%，"
          f"中位 {np.median(fh)/H*100:.1f}%）")
    for lab, v, span, ax in (("X", cx, W, "宽"), ("Y", cy, H, "高")):
        rng = v.max() - v.min()
        print(f"人脸框中心 {lab}  {v.min():.0f} ~ {v.max():.0f}px   "
              f"位移 {rng:.0f}px = 画面{ax}的 {rng/span*100:.1f}%")
    dx = (cx.max() - cx.min()) / W
    dy = (cy.max() - cy.min()) / H
    print(f"相邻帧位移  中位 {np.median(np.hypot(np.diff(cx), np.diff(cy))):.1f}px"
          f" / p90 {np.percentile(np.hypot(np.diff(cx), np.diff(cy)), 90):.1f}px")

    if eyes is not None and eyes[0].size > 10:
        ex, ey = eyes
        rngx, rngy = (ex.max() - ex.min()) / W, (ey.max() - ey.min()) / H
        print(f"两眼中点位移  水平 {rngx*100:.1f}% / 垂直 {rngy*100:.1f}%")
        kx = zoom_lower_bound(ex, W)
        ky = zoom_lower_bound(ey, H)
    else:
        kx = zoom_lower_bound(cx, W)
        ky = zoom_lower_bound(cy, H)

    print()
    print("锁头可行性（相似变换路线：锚点取画面中心，四角反解求 k 下界）")
    print(f"  水平需放大 ≥ {kx:.2f}x   垂直需放大 ≥ {ky:.2f}x   "
          f"-> 综合 k ≈ {max(kx, ky):.2f}x")
    k = max(kx, ky)
    if np.isfinite(k):
        print(f"  有效分辨率 ≈ {W/k:.0f}px 宽"
              f"（原 {W}px，掉 {100 - W/k/W*100:.0f}%）")
    print(f"  留白余量判据（拍摄时四周应留 15%~20%）: "
          f"水平 {dx*100:.1f}% / 垂直 {dy*100:.1f}%  ->",
          "达标 ✓ 锁头可用" if max(dx, dy) <= 0.20 else
          "超了 ✗ 锁头会掉分辨率（或露黑边），建议重录时动作幅度收小、坐远一点")

    print()
    print("按 1080 宽归一化的标准差（与 probe_lock.py 同口径，便于与成片对比）")
    print(f"  人脸框中心  水平 {cx.std()/W*1080:.1f}px  垂直 {cy.std()/H*1920:.1f}px")
    if eyes is not None and eyes[0].size > 10:
        print(f"  两眼中点    水平 {eyes[0].std()/W*1080:.1f}px  "
              f"垂直 {eyes[1].std()/H*1920:.1f}px"
              f"   <- 与 face_crop --lock 的锁定点同口径，验收看这一行")


def main():
    ap = argparse.ArgumentParser(
        description="量素材的头部活动范围 / 锁头可行性；也能量成片的锁定漂移")
    ap.add_argument("input", help="视频或特征缓存 .npz")
    ap.add_argument("--roi-below", type=float, default=0.35,
                    help="整图检出率低于此值就改用两级检测（默认 0.35）")
    ap.add_argument("--roi", dest="roi", action="store_true", default=None,
                    help="强制两级检测")
    ap.add_argument("--no-roi", dest="roi", action="store_false",
                    help="强制整图检测")
    ap.add_argument("--eyes-only", action="store_true",
                    help="只看两眼中点（验收锁头成片时用）")
    a = ap.parse_args()

    if a.input.lower().endswith(".npz"):
        cx, cy, fh, eyes, W, H, src, rate, n = from_cache(a.input)
    else:
        cx, cy, fh, eyes, W, H, src, rate, n = from_video(
            a.input, a.roi_below, a.roi)

    if a.eyes_only:
        if eyes is None or eyes[0].size < 10:
            print("拿不到两眼中点（人脸太小或检测失败）——"
                  "先用两级检测跑一遍")
            return 1
        ex, ey = eyes
        print()
        print("=" * 78)
        print(f"锁定验收（两眼中点）—— {os.path.basename(str(src))}")
        print("=" * 78)
        print(f"  {W}x{H}  {n} 帧  命中 {rate*100:.1f}%")
        print(f"  水平 std {ex.std()/W*1080:.1f}px（范围 {(ex.max()-ex.min())/W*1080:.0f}px）")
        print(f"  垂直 std {ey.std()/H*1920:.1f}px（范围 {(ey.max()-ey.min())/H*1920:.0f}px）")
        ok = ex.std() / W * 1080 < 10 and ey.std() / H * 1920 < 10
        print(f"  判定: {'通过 ✓（<10px）' if ok else '未达标 ✗'}"
              f"   [标准差按 1080x1920 画布归一化]")
        return 0

    report(a.input, cx, cy, fh, eyes, W, H, src, rate, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
