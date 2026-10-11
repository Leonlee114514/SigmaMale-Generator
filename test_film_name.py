# -*- coding: utf-8 -*-
"""成片命名规则测试（test_film_name.py）
====================================================================
命名规则看着简单，但有三处很容易改坏、而且坏了**不报错**：

  1. **token 顺序**：顺序一乱，同类文件在目录里就排不到一起
  2. **版本扫描要含 `.plan.json`**：只扫 mp4 的话，"删了 mp4 留 plan"
     会让新成片覆盖掉那份方案（而方案是验收脚本的输入）
  3. **版本扫描要含存量文件**：新规则第一次跑时，默认名早被旧成片占着；
     不算存量就会**把旧成片覆盖掉**

第 3 条尤其重要 —— 它是"存量文件一个都不动"这个承诺的技术保障。

用法：
    python test_film_name.py
"""
import os
import shutil
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import film_name as fn  # noqa: E402

FAILS = []


def eq(desc, got, want):
    if got == want:
        print(f"  ✓ {desc}")
    else:
        print(f"  ✗ {desc}\n      得到 {got!r}\n      期望 {want!r}")
        FAILS.append(desc)


def touch(*paths):
    for p in paths:
        with open(p, "w", encoding="utf-8") as f:
            f.write("x")


def main():
    print("=" * 66)
    print("[1] 示例表：film_stem 逐条比对")
    print("=" * 66)
    eq("默认", fn.film_stem("卡点", "vertical"), "卡点_竖屏")
    eq("人脸特写", fn.film_stem("卡点", "vertical", face_crop=3.0),
       "卡点_竖屏_人脸特写")
    eq("人脸特写+锁头",
       fn.film_stem("卡点", "vertical", face_crop=3.0, face_lock=True),
       "卡点_竖屏_人脸特写_锁头")
    eq("调色", fn.film_stem("卡点", "vertical", grade=True), "卡点_竖屏_调色")
    eq("极碎", fn.film_stem("卡点", "vertical", slices=2), "卡点_竖屏_极碎")
    eq("组合（顺序：构图→节奏→色彩）",
       fn.film_stem("卡点", "vertical", face_crop=3.0, face_lock=True,
                    grade=True, slices=2),
       "卡点_竖屏_人脸特写_锁头_极碎_调色")
    eq("复刻版/方形", fn.film_stem("复刻版", "square"), "复刻版_方形")
    eq("安全区", fn.film_stem("卡点", "vertical", face_crop=3.0,
                              face_safe_y="0:100"), "卡点_竖屏_人脸特写_安全区")

    print()
    print("[2] 默认值不得进名（否则每次都是噪声）")
    print("=" * 66)
    eq("slices=1 不进名", fn.film_stem("卡点", "vertical", slices=1), "卡点_竖屏")
    eq("bw=0 不进名", fn.film_stem("卡点", "vertical", bw=0.0), "卡点_竖屏")
    eq("no_fx=False 不进名", fn.film_stem("卡点", "vertical", no_fx=False),
       "卡点_竖屏")
    eq("face_crop=0 不进名（编辑器里 0 就是不裁）",
       fn.film_stem("卡点", "vertical", face_crop=0), "卡点_竖屏")
    eq("多余的键被忽略", fn.film_stem("卡点", "vertical", grain=14, rgb=6,
                                      dry_run=True), "卡点_竖屏")
    eq("降饱和只在 >0 时进名", fn.film_stem("卡点", "vertical", bw=0.35),
       "卡点_竖屏_降饱和")

    print()
    print("[3] 版本号规则（合成目录，与真实 output/ 无关）")
    print("=" * 66)
    tmp = tempfile.mkdtemp(prefix="filmname_")
    try:
        d = tmp
        eq("空目录 → 裸名", os.path.basename(fn.next_versioned(d, "卡点_竖屏")),
           "卡点_竖屏.mp4")

        touch(os.path.join(d, "卡点_竖屏.mp4"))
        eq("裸名已存在 → _v2", os.path.basename(fn.next_versioned(d, "卡点_竖屏")),
           "卡点_竖屏_v2.mp4")

        touch(os.path.join(d, "卡点_竖屏_v2.mp4"))
        eq("裸名+_v2 → _v3", os.path.basename(fn.next_versioned(d, "卡点_竖屏")),
           "卡点_竖屏_v3.mp4")

        # 删掉裸名，只剩 _v2：应续到 _v3（不是回头用裸名）
        os.remove(os.path.join(d, "卡点_竖屏.mp4"))
        os.remove(os.path.join(d, "卡点_竖屏_v3.mp4")) if os.path.exists(
            os.path.join(d, "卡点_竖屏_v3.mp4")) else None
        eq("裸名被删但 _v2 在 → 续 _v3（不回头用裸名）",
           os.path.basename(fn.next_versioned(d, "卡点_竖屏")),
           "卡点_竖屏_v3.mp4")

        # 只留 .plan.json（mp4 被删）：也算占用，否则会覆盖那份方案
        d2 = tempfile.mkdtemp(prefix="filmname2_")
        touch(os.path.join(d2, "卡点_竖屏.plan.json"))
        eq("只留 .plan.json 也算占用 → _v2",
           os.path.basename(fn.next_versioned(d2, "卡点_竖屏")),
           "卡点_竖屏_v2.mp4")
        shutil.rmtree(d2, ignore_errors=True)

        # 不同模式各自独立编号
        touch(os.path.join(d, "卡点_竖屏_人脸特写.mp4"))
        eq("别的模式占用不影响本模式",
           os.path.basename(fn.next_versioned(d, "卡点_竖屏")),
           "卡点_竖屏_v3.mp4")
        eq("本模式自己计数",
           os.path.basename(fn.next_versioned(d, "卡点_竖屏_人脸特写")),
           "卡点_竖屏_人脸特写_v2.mp4")

        # 主名里含数字也要安全（正则锚定在整个主名上）
        touch(os.path.join(d, "降饱和35.mp4"))
        eq("主名含数字：仍按整个主名匹配",
           os.path.basename(fn.next_versioned(d, "降饱和35")), "降饱和35_v2.mp4")
        eq("主名含数字：不会串到别的 stem",
           os.path.basename(fn.next_versioned(d, "降饱和")), "降饱和.mp4")

        # 非成片后缀不参与计数
        touch(os.path.join(d, "卡点_竖屏_备注.txt"),
              os.path.join(d, "卡点_竖屏_v9.mkv"))
        eq("无关后缀不参与计数",
           os.path.basename(fn.next_versioned(d, "卡点_竖屏")),
           "卡点_竖屏_v3.mp4")

        # 伴生路径约定
        eq("plan 与成片同 stem",
           os.path.basename(fn.plan_path_of(
               os.path.join(d, "卡点_竖屏_v3.mp4"))), "卡点_竖屏_v3.plan.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("[4] 存量避让（对真实 output/ 的信息性检查，不做硬断言）")
    print("=" * 66)
    out = os.path.join(BASE, "output")
    if os.path.isdir(out):
        got = os.path.basename(fn.next_versioned(out, "卡点_竖屏"))
        print(f"  普通版下一个可用名：{got}")
        if got == "卡点_竖屏.mp4":
            print("      （裸名当前空闲；若之前有旧成片被删过，这是正常的）")
        else:
            print("      ✓ 已避开存量文件，不会覆盖旧成片")
    else:
        print("  （没有 output/ 目录，跳过）")

    print()
    print("=" * 66)
    if FAILS:
        print(f"✗ 失败 {len(FAILS)} 项：")
        for d in FAILS:
            print(f"    {d}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
