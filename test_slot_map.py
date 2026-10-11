# -*- coding: utf-8 -*-
"""回归测试：分镜表的「源 ↔ 时间槽」映射（test_slot_map.py）
====================================================================
**为什么必须有这个测试**

2026-09-24 发现：`zoompan`（缩放脉冲）会按自己的 fps 重新生成时间戳、
**完全忽略输入 PTS**。而快剪段的时间轴在 `setpts` 里已经被压缩过（1.35x），
于是 zoompan 输出的时长是"未压缩"的 —— 整段被拉长，**其后所有分镜整体后移**。

最阴的地方是它**不报错**：
    · 总帧数照样 562、时长照样 18.75s
    · `verify_cuts` 照样报"切点压在拍上"
    · `verify_flash` 照样通过
    · `verify_contentaware` 也照样通过（它只比"实拍镜 vs 实拍原片"）
而实际成片里，"每一格放的是哪一段源"从第一个 zoom 之后就全错位了。
`--no-fx` 渲染则完全正确 —— 所以只看"关掉特效的验收片"永远发现不了。

**这个测试怎么抓它**

不用真实素材（画面内容无法自动判源），改用**纯色合成源**：
实拍=纯红、素材=纯蓝。然后逐格采样，看 BGR 通道就知道该格用的是哪个源 ——
零歧义、零主观。再与 `--dump-plan` 写出的方案逐格比对。

用法：
    python test_slot_map.py            # 跑测试（约 30s）
    python test_slot_map.py --keep     # 保留中间产物便于排查
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import edit_slop as es          # noqa: E402

FFMPEG = es.FFMPEG
PY = sys.executable


def make_source(path, color, secs):
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=c={color}:s=1080x1920:r=30",
         "-t", str(secs), "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", path], check=True)


def sample(plan, video):
    """逐格采样，返回 (帧号, 判定的源, 是否被反相) 列表。"""
    cap = cv2.VideoCapture(video)
    out = []
    for s in plan["shots"]:
        idx = int(round((s["out_start"] + s["out_dur"] * 0.6) * 30))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            out.append((s, None, False))
            continue
        b, g, r = [float(v) for v in f.reshape(-1, 3).mean(axis=0)]
        # 纯红 (R≫B) 判实拍；纯蓝 (B≫R) 判素材；负片闪会把两者对调
        inverted = s["kind"] == "neg"
        is_fw = (r > b)
        if inverted:
            is_fw = not is_fw
        out.append((s, is_fw, inverted))
    cap.release()
    return out


def run(tmp):
    red = os.path.join(tmp, "red.mp4")
    blue = os.path.join(tmp, "blue.mp4")
    make_source(red, "red", 15.1)
    make_source(blue, "blue", 30.1)

    out = os.path.join(tmp, "out.mp4")
    plan_p = os.path.join(tmp, "out.json")
    argv = sys.argv
    sys.argv = ["edit_slop.py", "--footage", red, "--material", blue,
                "--out", out, "--dump-plan", plan_p]
    try:
        es.main()
    finally:
        sys.argv = argv
    plan = json.load(open(plan_p, encoding="utf-8"))

    bad = []
    for s, got_fw, inv in sample(plan, out):
        want_fw = (s["src"] == 0)
        if got_fw is None:
            bad.append((s["i"], "读帧失败"))
        elif got_fw != want_fw:
            bad.append((s["i"], f"方案{'实拍' if want_fw else '素材'}"
                               f"/实测{'实拍' if got_fw else '素材'}"))
    return plan, bad


def main():
    keep = "--keep" in sys.argv
    tmp = tempfile.mkdtemp(prefix="slotmap_")
    print(f"临时目录 {tmp}")
    try:
        plan, bad = run(tmp)
        n = len(plan["shots"])
        print()
        print("=" * 66)
        print(f"分镜源↔槽位映射测试：{n} 个分镜")
        print("=" * 66)
        if bad:
            print(f"✗ 错位 {len(bad)} 个：")
            for i, why in bad:
                print(f"    分镜 {i}：{why}")
            print("\n提示：查 shot_chain 里各效果有没有改时序。"
                  "已知 zoompan 会重定时，必须在其后 trim 回目标时长。")
            print(f"（临时目录已保留：{tmp}）")
            return 1
        print(f"✓ 全部 {n} 个分镜的源与方案一致")
        # 顺带核对总时长
        dur = sum(s["out_dur"] for s in plan["shots"])
        cap = cv2.VideoCapture(os.path.join(tmp, "out.mp4"))
        got = cap.get(cv2.CAP_PROP_FRAME_COUNT) / 30
        cap.release()
        print(f"  总时长 方案 {dur:.3f}s / 实测 {got:.3f}s"
              f"（差 {abs(got-dur)*1000:.0f}ms）")
        return 0
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"中间产物保留在 {tmp}")


if __name__ == "__main__":
    sys.exit(main())
