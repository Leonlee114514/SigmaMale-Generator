# -*- coding: utf-8 -*-
"""成片自动命名（film_name.py）
====================================================================
**为什么单独一个模块**

成片有**两个出片入口**：

    · 网页流水线 `slop_pipeline.py` —— 它给 edit_slop/edit_ref 传 `--out`
    · `卡点成片.bat` —— 直接调 `edit_slop.py`，**不传 `--out`**，走兜底名

两条路必须产出**完全一致**的名字。命名逻辑只要有两份，迟早会不一致
（本项目的"验收的对象和交付的对象悄悄不是同一个"已经栽过三次），
所以只留这一份，两边都 import。

**名字长什么样**

扁平、模式词用 `_` 连接、空 token 跳过：

    <模板>_<画幅>[_人脸特写][_锁头][_安全区][_极碎][_调色][_降饱和][_无特效]

    卡点_竖屏.mp4
    卡点_竖屏_人脸特写.mp4
    卡点_竖屏_人脸特写_锁头.mp4
    卡点_竖屏_极碎_调色.mp4
    复刻版_方形.mp4

**为什么模式词不带数字**

版本号扫描用 `^<主名>(?:_v(\\d+))?$`。词里混数字的话，
`降饱和35` 里那个 35 到底是参数还是版本号，人和正则都难判断。
精确参数值一律记进 plan.json 的 `render` 段（见 edit_slop.py 的 dump）。

**为什么 `内容感知` 不进名**

`--clips`（按动作峰值取段）现在是**默认常开**，人人都有，没有区分度。
（早期文件 `卡点_内容感知_竖屏.mp4` 是它刚出现时用来做区分的。）
想知道某片是不是内容感知取段，读 plan.json 的 `content_aware` 字段。
"""
import os
import re

# 画幅词（沿用项目既有写法）
ASPECT_WORD = {"vertical": "竖屏", "square": "方形"}

# 模式词表：**顺序即文件名里的顺序**，不要随意调换（同类文件才排得在一起）。
#   顺序 = 构图 → 节奏 → 色彩 → 效果
# 每项 (参数键, 模式词)。触发判定见 _hit()。
TOKEN_ORDER = (
    ("face_crop", "人脸特写"),
    ("face_lock", "锁头"),
    ("face_safe_y", "安全区"),
    ("slices", "极碎"),
    ("grade", "调色"),
    ("bw", "降饱和"),
    ("no_fx", "无特效"),
)

# Windows 文件名非法字符。模式词都是项目里定的中文词，正常不会命中；
# 留着是防将来有人把模板名/画幅传成别的东西。
_ILLEGAL = '<>:"/\\|?*'


def _hit(key, v):
    """这个参数算不算"偏离默认、值得进名"。

    ⚠ 判定要和 edit_slop/edit_ref 里 `if a.xxx:` 的语义保持一致 ——
      否则会出现"名字说有、实际没生效"（比不写还糟）。
      比如 face_crop 用真值判断：0.0 在编辑器里就是不裁脸，名字里也不该写。
    """
    if v is None or v is False:
        return False
    if key == "slices":
        return int(v) >= 2          # 1 = 每拍一刀（默认），2 = 每半拍一刀
    if key == "bw":
        return float(v) > 0         # 0 = 不动（默认）
    return bool(v)


def _sanitize(s):
    out = "".join("_" if (ch in _ILLEGAL or ord(ch) < 32) else ch for ch in s)
    return out.strip().strip(".")


def film_stem(template, aspect, **params):
    """由模板名 + 画幅 + 模式参数拼出成片**主名**（不含版本号与扩展名）。

    `params` 只认 TOKEN_ORDER 里那几个键，其余一律忽略 ——
    这样调用方可以把整个配置字典直接塞进来，不用自己筛一遍。
    """
    toks = [template, ASPECT_WORD.get(aspect, aspect)]
    for key, word in TOKEN_ORDER:
        if _hit(key, params.get(key)):
            toks.append(word)
    return _sanitize("_".join(t for t in toks if t))


def plan_path_of(out_path):
    """成片的方案文件路径。同 stem 约定（webapp 也靠这个找伴生文件）。"""
    root, _ = os.path.splitext(out_path)
    return root + ".plan.json"


def newest_film(out_dir):
    """out_dir 里最新的一个**成片**；没有则返回空串。

    ⚠ 判据是"**有 `.plan.json` 伴生文件**"，不是"最新一个 mp4"。
      output/ 里除了成片还有录制素材（`录制_*.mp4`）和一堆中间产物，
      按 mtime 取最新会经常取到**刚录的素材** —— 然后验收报一堆
      "找不到切点""闪白 0 个"之类看不懂的错，根因却是选错了文件。
      成片必有方案文件，素材没有，这个判据很稳。

    以 `_` 开头的（`_pipeline/`、`_raw*`、`_facetest` 等）一律跳过。
    """
    best, best_t = "", -1.0
    try:
        names = os.listdir(out_dir)
    except OSError:
        return ""
    for name in names:
        if name.startswith("_") or not name.lower().endswith(".mp4"):
            continue
        p = os.path.join(out_dir, name)
        if not os.path.exists(plan_path_of(p)):
            continue
        try:
            t = os.path.getmtime(p)
        except OSError:
            continue
        if t > best_t:
            best, best_t = p, t
    return best


def next_versioned(out_dir, stem, ext=".mp4"):
    """返回 out_dir 下该主名**下一个可用**的完整路径（绝不覆盖已有文件）。

    版本规则：首次用裸名，其后 `_v2` / `_v3` …

    ⚠ 取 **max+1**，不是"最小空闲号"。两者在"中间某个版本被删掉"时结果不同：
      假设现有 裸名 与 _v2，此时把裸名删了 ——
        · 最小空闲号 → 又回到裸名（**号会倒退**）
        · max+1     → _v3
      必须用后者：只有单调递增，"号最大 = 最新"这个直觉才成立；
      否则用户按号找最新片会找错。而且本项目真的发生过"把 _v2 挪走"，
      最小空闲号会立刻把那个号重新用掉，跟历史版本号撞车。
      全删光时 max 视为 0 → 从裸名重新开始（合理）。

    ⚠⚠ 扫描**必须同时包含 `.mp4` 与 `.plan.json`**：
      成片与它的方案同 stem。只扫 mp4 的话，"删了 mp4 但留下 plan"
      会让新片直接覆盖掉那份 plan —— 而 plan 是验收脚本的输入。

    ⚠⚠ 也必须把**存量文件**算进来：新规则第一次跑时，
      默认名 `卡点_竖屏` 早已被旧成片占着；不算存量就会把它们覆盖掉。
    """
    pat = re.compile(re.escape(stem) + r"(?:_v(\d+))?$", re.IGNORECASE)
    used = set()
    try:
        names = os.listdir(out_dir)
    except OSError:
        names = []
    for fn in names:
        root, e = os.path.splitext(fn)
        e = e.lower()
        if e == ".json":
            if not root.lower().endswith(".plan"):
                continue
            root = root[: -len(".plan")]
        elif e != ".mp4":
            continue
        m = pat.fullmatch(root)
        if m:
            used.add(int(m.group(1)) if m.group(1) else 1)

    n = (max(used) + 1) if used else 1
    name = stem if n == 1 else f"{stem}_v{n}"
    return os.path.join(out_dir, name + ext)
