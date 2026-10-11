# -*- coding: utf-8 -*-
"""校验「一镜一段」—— 两条独立判据。

判据 A（算术，精确）：
    段长 DUR = 模板入点步长；镜 k 的素材区间 = [k·DUR, k·DUR + 单镜时长]。
    必须 k·DUR + 单镜时长 ≤ (k+1)·DUR，即**整镜落在同一段内**。
    ⚠ plan.json 把 in_pt/out_dur 四舍五入到 4 位小数，所以段号按 round() 定，
      并留 2e-4 的舍入容差。

判据 B（实测，不看元数据）：
    在**素材文件**上按 30fps 扫过整镜区间，算相邻帧差；
    若镜内出现远高于中位的尖峰，说明画面中途换了场景（跨段）。

用法：python verify_one_seg.py [plan.json ...]
"""
import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SEG = {"slop": 60.0 / 128 * 1.35, "ref": 60.0 / 67.12}
TOL = 2e-4
SPIKE = 3.0          # 镜内帧差尖峰超过中位多少倍算"跳变"


def seg_of(ip, dur):
    """按 round 定段号（抗 plan 的 4 位舍入）。"""
    return int(round(ip / dur))


def decode_all(path):
    """顺序解码整条素材 -> [(t_sec, gray64), ...]。

    ⚠ 不要用 CAP_PROP_POS_MSEC 逐帧定位：它会跳到关键帧，产生**假跳变**。
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (64, 64)), cv2.COLOR_BGR2GRAY)
        out.append((i / fps, g.astype(np.float32)))
        i += 1
    cap.release()
    return out


def shot_stats(frames, t0, t1):
    """镜内帧差 (max, median)；以及**镜首那一刀**的帧差（作为参照）。"""
    idx = [k for k, (t, _) in enumerate(frames) if t0 - 1e-6 <= t < t1 - 1e-6]
    if len(idx) < 3:
        return None
    d = [float(np.mean(np.abs(frames[k][1] - frames[k - 1][1])))
         for k in idx[1:]]
    a = np.asarray(d)
    cut = None
    if idx[0] > 0:
        cut = float(np.mean(np.abs(frames[idx[0]][1] - frames[idx[0] - 1][1])))
    return float(a.max()), float(np.median(a)), cut


def check(plan_path):
    pl = json.load(open(plan_path, encoding="utf-8"))
    name = os.path.basename(plan_path)
    mat_rel = pl.get("material", "")
    mat = mat_rel if os.path.isabs(mat_rel) else os.path.join(BASE, mat_rel)
    key = "ref" if "复刻" in os.path.basename(mat) else "slop"
    DUR = SEG[key]
    shots = sorted([s for s in pl["shots"] if s.get("src") == 1],
                   key=lambda s: s["in_pt"])
    print(f"\n=== {name} ===")
    print(f"  素材 {os.path.basename(mat)}   段长 {DUR:.7f}s   "
          f"单镜 {shots[0]['out_dur']:.4f}s")
    frames = decode_all(mat)
    print(f"  {'镜':>3} {'入点':>9} {'段号':>4} {'余量':>9} "
          f"{'镜内峰':>8} {'镜内中位':>9} {'镜首一刀':>9}  判定")
    okA = okB = True
    for i, s in enumerate(shots):
        ip, dur = s["in_pt"], s["out_dur"]
        k = seg_of(ip, DUR)
        margin = (k + 1) * DUR - (ip + dur)
        goodA = margin >= -TOL
        r = shot_stats(frames, ip, ip + dur)
        if r is None:
            mx = md = cut = float("nan")
            goodB = True
        else:
            mx, md, cut = r
            # 镜内最大帧差若远小于"镜首那一刀"，说明镜内没有真正的切点
            goodB = (cut is None) or (mx < cut * 0.7)
        okA &= goodA
        okB &= goodB
        cs = "—" if cut is None else f"{cut:.2f}"
        print(f"  {i+1:>3} {ip:>9.4f} {k+1:>4} {margin:>9.4f} "
              f"{mx:>8.2f} {md:>9.2f} {cs:>9}  "
              f"{'✓' if goodA and goodB else '✗'} "
              f"{'' if goodA else '跨段 '}"
              f"{'' if goodB else '镜内跳变'}")
    used = sorted({seg_of(s["in_pt"], DUR) + 1 for s in shots})
    print(f"  用到的段: {used}")
    print(f"  判据A(整镜落一段) {'通过' if okA else '未通过'} / "
          f"判据B(镜内无跳变) {'通过' if okB else '未通过'}")
    return okA and okB


if __name__ == "__main__":
    args = sys.argv[1:] or ["output/卡点_竖屏_v12.plan.json",
                            "output/复刻版_v7.plan.json"]
    allok = True
    for p in args:
        allok &= check(os.path.join(BASE, p))
    print("\n" + "=" * 60)
    print("总结果:", "全部通过" if allok else "有未通过项")
    sys.exit(0 if allok else 1)
