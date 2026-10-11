# -*- coding: utf-8 -*-
"""`--action-plan`（指定动作）的回归测试 —— 纯逻辑，不渲染。

用法：
    python test_action_plan.py            # 约 1s

覆盖四件事：
  1. `parse_action_plan()` 的解析与**报错**（格式错必须抛错，不能静默忽略）
  2. `select_anchors()` 无 pins 时**保持历史行为**（按时间升序、每类一次）
  3. `select_anchors()` 有 pins 时**下标 k 就是槽位 k**（这是整个功能的地基）
  4. 真实 reel_map 上的冒烟（文件不在就跳过）

⚠ 第 3 条为什么必须测：`apply_content_aware` 是**按位置**把结果映射到槽位的
  （chosen[k] → 槽位 k）。一旦有人把有 pins 的分支也改成"按时间排序返回"，
  "槽位 3 = 举手"就会悄悄失效 —— 而且不报错。这类静默失效只能靠测试挡住。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_aware as ca

BASE = os.path.dirname(os.path.abspath(__file__))
REEL = os.path.join(BASE, "output", "_pipeline", "reel_map_used.json")
FAIL = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


print("=== 1) parse_action_plan ===")
check(ca.parse_action_plan("3=举手,7=转头") == {2: "举手", 6: "转头"}, "基本解析")
check(ca.parse_action_plan("3 = 举手 ， 7=转头") == {2: "举手", 6: "转头"}, "全角逗号 + 空白")
check(ca.parse_action_plan("") == {}, "空串 → {}")
check(ca.parse_action_plan(None) == {}, "None → {}")
for bad, why in [("3:举手", "缺 ="), ("abc=举手", "槽位非数字"),
                 ("0=举手", "槽位从 1 开始"), ("3=", "缺动作名")]:
    try:
        ca.parse_action_plan(bad)
        check(False, f"应报错但没报：{bad}（{why}）")
    except ValueError:
        check(True, f"正确报错：{bad}（{why}）")

print()
print("=== 2) 无 pins：保持历史行为 ===")
ACTS = ["张嘴", "嘟嘴", "转头", "举手", "挑眉", "后退"]
OD = 0.633
cand = [{"t": round(0.3 + i * 0.7, 3), "action": ACTS[i % len(ACTS)],
         "peak": 0.5 + 0.05 * (i % 3)} for i in range(12)]
base = ca.select_anchors(cand, 8, OD, ca.WEAK_ACTIONS)
check(len(base) == 8, f"选满 8 个（实得 {len(base)}）")
check([p["t"] for p in base] == sorted(p["t"] for p in base), "按时间升序")
check(len({p["action"] for p in base}) == 6, "尽量覆盖不同动作")

print()
print("=== 3) 有 pins：下标 k 就是槽位 k ===")
pins = {2: "举手", 6: "转头"}
r = ca.select_anchors(cand, 8, OD, ca.WEAK_ACTIONS, pins)
check(len(r) == 8, f"仍选满 8 个（实得 {len(r)}）")
for k, want in pins.items():
    got = r[k]["action"] if k < len(r) else None
    check(got == want, f"槽位 {k} 指定「{want}」→ 实际「{got}」")
tt = sorted(p["t"] for p in r)
check(all(tt[i + 1] - tt[i] >= OD - 1e-6 for i in range(len(tt) - 1)),
      "所有锚点时间不重叠")

print()
print("=== 4) 指定允许重复 / 不可满足 / 越界 ===")
r2 = ca.select_anchors(cand, 8, OD, ca.WEAK_ACTIONS, {1: "举手", 4: "举手"})
check(r2[1]["action"] == "举手" and r2[4]["action"] == "举手",
      "两个槽位都拿到「举手」（不受「每类一次」限制）")
r3 = ca.select_anchors(cand, 8, OD, ca.WEAK_ACTIONS, {0: "不存在的动作"})
check(len(r3) == 8 and r3[0]["action"] != "不存在的动作",
      "不可满足 → 退回自动，且不伪造")
r4 = ca.select_anchors(cand, 8, OD, ca.WEAK_ACTIONS, {99: "举手"})
check(len(r4) == 8, "越界槽位被忽略，不崩")

print()
print("=== 5) 真实 reel_map 冒烟 ===")
if not os.path.exists(REEL):
    print(f"  跳过（没有 {os.path.relpath(REEL, BASE)}）")
else:
    cm = ca.load_clipmap(REEL)
    if cm is None:
        check(False, "load_clipmap 返回 None")
    else:
        rp = cm["peaks"]
        avail = sorted({p["action"] for p in rp})
        print(f"  动作池 {avail}（{len(rp)} 个峰值）")
        real = ca.select_anchors(rp, 8, OD, ca.WEAK_ACTIONS)
        print("  无 pins：" + " / ".join(f"{p['action']}@{p['t']:.2f}" for p in real))
        real2 = ca.select_anchors(rp, 8, OD, ca.WEAK_ACTIONS, {1: "举手", 5: "转头"})
        print("  有 pins：" + " / ".join(f"{p['action']}@{p['t']:.2f}" for p in real2))
        check(len(real2) == 8, "真实数据下选满 8 个")
        check(real2[1]["action"] == "举手", "槽位 1 拿到「举手」")
        check(real2[5]["action"] == "转头", "槽位 5 拿到「转头」")

print()
print("=" * 52)
print("结果：全部通过" if not FAIL else f"结果：{len(FAIL)} 项失败")
for f in FAIL:
    print("  ✗", f)
sys.exit(1 if FAIL else 0)
