"""
Slop 预制剪辑模板库 (slop_templates.py)
=========================================
把互联网 slop 短视频的标准剪辑语法，实现为可复用的 ffmpeg 滤镜链。

调研来源：
  - CapCut 官方速度曲线文档：曲线图上半区加速至 10x，下半区减速至 0.1x
  - brainrot 剪辑拆解（opus.pro）：
      总时长 15-30s；主素材 20-40s；梗图每 3-5 秒插一次(1-3 帧)；
      背景乐 15-25% 音量；缩放每 5 秒 1 次；RGB 故障仅 1 次；
      导出 9:16 / 30fps / 字幕烧录
  - ffmpeg 官方 wiki：setpts 乘数 <1 加速、>1 减速；minterpolate 做光流补帧
  - 倒放素材要求：动作有明确起点终点、无文字、≤10s

所有滤镜链均在本机 ffmpeg 8.1.1 实测通过。
"""

FFMPEG = r"C:\Users\Lyonl\AppData\Local\Programs\FlyingMouse Format\resources\ffmpeg\ffmpeg.exe"

# ------------------------------------------------------------------ 输出规格
SPEC = {
    "w": 1080, "h": 1920, "fps": 30,
    "vcodec": "libx264", "preset": "medium", "crf": "20",
    "pix_fmt": "yuv420p",
}

W, H, FPS = SPEC["w"], SPEC["h"], SPEC["fps"]


def _no_minus(expr):
    """把裸减号 `-` 改写成 `+-`。

    ⚠⚠ 说明：**这条改写其实不是必需的**，早期对它的归因是错的，留在这里
    只是因为它无害（`PTS+-1.4` 与 `PTS-1.4` 求值相同）。

    真正的规则（probe_setpts.py 实测，2026-09-23 更正）：

      本机 ffmpeg 8.1.1 的表达式解析器**不支持裸的 `<` 和 `>` 比较运算符**，
      用了就报 `Missing ')'`。**必须改用 `lt()` / `gt()` / `gte()` 函数。**

          setpts=(T<0.5)*(PTS-0.5)    ->  Missing ')'   ✗
          setpts=(T<0.5)*(PTS+-0.5)   ->  Missing ')'   ✗  （旧文档误称可用）
          setpts=(T<0.5)*(PTS+0.5)    ->  Missing ')'   ✗  （无减号也挂！）
          setpts=(T>0.5)*PTS          ->  Missing ')'   ✗
          setpts=lt(T\\,0.5)*PTS       ->  正常           ✓

      早期之所以误判成"减号问题"，是因为当时失败的例子
      `(T<1.4)*(PTS-1.4)` **同时含 `<` 和 `-`**，改掉减号后看起来"好了"，
      其实是巧合 —— 真正的原因是那个 `<`。

    另一条独立规则：**函数调用里的逗号必须转义成 `\\,`**，
    否则滤镜图会在逗号处切开，报 `No such filter: '0.5)*PTS'`。
    """
    return expr.replace("-", "+-")


def _esc_commas(expr):
    """给表达式里所有**函数调用内部**的逗号加反斜杠转义。

    ffmpeg 的滤镜图在**进入表达式解析器之前**就会按逗号切分滤镜链，
    所以 `if(lt(T,1.4),A,B)` 会被切成三段，报 `No such filter: '1.4)'`。
    必须写成 `if(lt(T\\,1.4)\\,A\\,B)`。

    括号深度为 0 的逗号（真正的滤镜分隔符）保持原样不动。
    """
    out, depth = [], 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth > 0:
            out.append("\\,")
        else:
            out.append(ch)
    return "".join(out)


def setpts_expr(expr):
    """把裸表达式包装成可安全放进 -vf / -filter_complex 的完整 setpts 滤镜。

    统一处理上面两个坑（减号 + 逗号转义），所有变速模板都走这个入口。
    """
    return f"setpts={_esc_commas(_no_minus(expr))}"


# =================================================================== 曲线变速
def speed_ramp_piecewise(slow_at=0.35, slow_span=0.30, fast=3.0, slow=0.4,
                         duration=None):
    """分段变速（最接近剪映/CapCut 的"曲线变速"预设）

    形状：快入 -> 慢放(高潮点) -> 快出。这是"卡点变速"最常用的形态。

    参数
        slow_at    慢放起点在时间轴上的相对位置（0-1）
        slow_span  慢放持续的相对时长
        fast       快段的速度倍率（>1 = 加速）
        slow       慢段的速度倍率（<1 = 减速）

    返回**完整滤镜字符串**（含 setpts= 前缀），可直接放进 -vf。

    实测坑：ffmpeg 8.1.1 的表达式解析器对本式里的减号有 BUG，
    故统一经 setpts_expr() 做 `-` -> `+-` 改写。
    """
    a = slow_at
    b = min(slow_at + slow_span, 1.0)
    # setpts 语义：PTS/倍率 -> 加速；PTS*倍率 -> 减速
    # 用 T（秒）判断当前帧处于哪一段。需 duration 才能用绝对秒。
    if duration is None:
        duration = 1.0
    t1, t2 = a * duration, b * duration
    # 逗号已由 setpts_expr 转义，这里可以放心用 if/lt 的常规写法
    lead = t1 / fast
    expr = (f"if(lt(T,{t1:g}),PTS/{fast:g},"
            f"if(lt(T,{t2:g}),{lead:g}+(PTS-{t1:g})/{slow:g},"
            f"{lead + (t2 - t1) / slow:g}+(PTS-{t2:g})/{fast:g}))")
    return setpts_expr(expr)


def speed_ramp_smooth(amp=0.8, period=1.0):
    """平滑正弦变速：速度连续变化，无硬切感。

    速度 = 1 / (1 + amp*sin(2*PI*T/period))
    amp=0.8 时：峰速 5x（在 sin=-1 处）、谷速 0.56x（在 sin=+1 处）。

    返回**完整滤镜字符串**（含 setpts= 前缀）。
    """
    return setpts_expr(f"PTS*(1.0+{amp:g}*sin(2*PI*T/{period:g}))")


def speed_const(rate):
    """恒定变速。rate>1 加速，rate<1 减速。

    返回**完整滤镜字符串**（含 setpts= 前缀）。
    """
    return setpts_expr(f"PTS*{1.0 / rate:g}")


# =================================================================== 慢动作丝滑
def with_motion_interpolation(fps_out=FPS):
    """光流插帧，消除慢放的卡顿感。必须接在 setpts 之后"""
    return (f"minterpolate='mi_mode=mci:mc_mode=aobmc:"
            f"vsbmc=1:me_mode=bidir:search_param=32:fps={fps_out}'")


# =================================================================== 倒放 / 循环
def boomerang_filters(reverse_audio=False):
    """Boomerang（正放+倒放来回）效果的滤镜链。

    实现：split 成两路，第二路 reverse 后 concat。
    结果是 正放 -> 倒放 的循环，产生"来回弹"的观感。
    """
    return ("split=2[a][b];"
            "[b]reverse[bv];"
            "[a][bv]concat=n=2:v=1:a=0")


def reverse_only():
    """纯倒放。注意：reverse 需把整段解码进内存，长视频很吃内存"""
    return "reverse"


# =================================================================== 正方形构图
def square_canvas(size=1080, bg_blur=True):
    """横屏素材 -> 正方形画布（"正方"效果）。

    bg_blur=True 时用自身模糊放大填满背景，避免黑边。
    这与项目里竖屏导出的做法同源。
    """
    if bg_blur:
        return (f"split=2[bg][fg];"
                f"[bg]scale={size}:{size}:force_original_aspect_ratio=increase,"
                f"crop={size}:{size},gblur=sigma=28[bgb];"
                f"[fg]scale={size}:-2[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2")
    return (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black")


# =================================================================== 镜像
def mirror(kind="hflip"):
    """镜像。hflip 左右翻转（最常用），vflip 上下颠倒"""
    return "hflip" if kind == "hflip" else "vflip"


def mirror_concat():
    """镜像拼接：正常 + 镜像 并排或串联，视觉上翻倍"""
    return "split=2[a][b];[b]hflip[bv];[a][bv]hstack"


def split_mirror(size=None):
    """分屏镜像：左半原图 + 右半镜像，拼成对称画面。

    与 mirror_concat() 是同一个构造，区别只在用途/命名：
    mirror_concat 强调"翻倍"，本函数强调"对称分屏"。
    实现：split -> 一路 hflip -> hstack 并排。
    ⚠ hstack 要求两路尺寸相同，所以 hflip 那一路不能再改尺寸。
    """
    base = "split=2[a][b];[b]hflip[bv];[a][bv]hstack"
    return f"{base},scale={size[0]}:{size[1]}" if size else base


def zoom_pulse(amp=0.35, period=0.46875, decay=10.0, size=(1080, 1920), fps=30):
    """节拍点突然放大再收回（关键帧缩放）。

    ⚠ 必须用 zoompan，**不能用 crop 表达式**。
      实测 `crop=w='iw/(1+...)':h='ih/(1+...)'` 会报
      `Error when evaluating the expression` —— crop 的 w/h 不支持
      随时间的表达式（x/y 支持，w/h 不支持）。这是很容易踩的一点：
      看起来"crop 能用表达式"（x/y 确实能），就以为 w/h 也行。

    ⚠ `size` 必须给**明确尺寸**。实测传 `iwxih` 会报
      `Unable to parse "s" option value "iwxih" as image size`
      —— zoompan 的 s 不接受表达式，只接受 WxH 字面量。
    """
    if not size or not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("zoom_pulse 需要明确尺寸，例如 size=(1080, 1920)")
    return (f"zoompan=z='1+{amp:g}*exp(-mod(time,{period:.6f})*{decay:g})'"
            f":d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={int(size[0])}x{int(size[1])}:fps={fps}")


def negate_flash(period=0.46875, dur=0.06, offset=0.0):
    """负片闪：节拍点短暂色彩反转（negate 的 enable 时间线选项）。

    enable 支持 t 表达式，所以可以精确门控在每拍开头 dur 秒。
    """
    return (f"negate=enable='lt(mod(t+{offset:.6f},{period:.6f}),{dur:g})'")


def motion_blur(frames=4, weights=None, enable=None):
    """动态模糊（时间维度混合 = 拖影），用于切换瞬间。

    ⚠ "模糊过渡"必须用 tmix 走**时间维度**，不能用 gblur 走空间维度：
      实测 `gblur=sigma='18*exp(-mod(t,...)*12)'` 报
      `Undefined constant or missing '(' in 't,0.46875)*12)'`
      —— gblur 的 sigma 不接受 t 表达式。
      而且从效果上说，切换瞬间要的是"拖影"（运动模糊），
      空间高斯模糊只会让画面变糊，不是同一个观感。
    """
    w = weights or " ".join(str(i + 1) for i in range(frames))
    e = f":enable='{enable}'" if enable else ""
    return f"tmix=frames={frames}:weights='{w}'{e}"


def bw(amount=1.0):
    """降饱和 / 黑白。amount=1 全黑白，0.5 半饱和。"""
    return f"hue=s={max(0.0, 1.0 - amount):g}"


# =================================================================== 抖动 / 故障
def shake(intensity=8, freq=0.9, out_size=None):
    """画面抖动（punchline 时刻用，克制使用）

    实现：每帧从画面里按正弦轨迹取一个偏移的裁剪窗口，再放大回原尺寸。
    因为先裁小（iw-2i）再放大回去，会附带约 i/(iw-i) 的轻微放大，
    这正是抖动常见的伴随效果。

    参数
        intensity  偏移像素（相对源分辨率）
        freq       抖动频率（周期/帧）。30fps 下建议 0.7-1.5；
                   太高（如 25）会超过帧率形成闪烁，看起来像噪点而非抖动。
        out_size   放大回的目标尺寸 (w, h)。

    ⚠ 陷阱 1：**绝不能把输出尺寸写死成 1080x1920**。本函数的裁剪尺寸是
              基于**源**尺寸算的，若源是 720x1280 而回放大写死 1080x1920，
              中间放大环节本身没问题，但如果写成 `crop=1064:1904`（即按
              1080x1920 减出来的），源只有 720x1280 就会直接报
              `Invalid too big or non positive size` -> 0 字节输出（静默失败）。
              故这里用 `out_size=None` 表示"保持源尺寸"。
    ⚠ 陷阱 2：crop 的 x/y 表达式**不能加单引号**。ffmpeg 滤镜参数解析器不剥
              引号，会把引号当普通字符 -> 表达式非法 -> 同样 0 字节输出。
    """
    i = int(intensity)
    chain = (f"crop=iw-{2 * i}:ih-{2 * i}:"
             f"{i}+{i}*sin(n*{freq:g}):"
             f"{i}+{i}*cos(n*{freq * 0.78:g})")
    if out_size:
        chain += f",scale={out_size[0]}:{out_size[1]}"
    return chain + ",setsar=1"


def rgb_split(shift=6, size=None):
    """RGB 通道分离 / 色散故障（仅用于最重要的 punchline 一次）

    实现：复制三路 -> 各留单通道（其余置 0）-> 红蓝分别水平错位
    -> 链式两次 blend=addition 相加还原。

    参数
        shift  红/蓝通道的水平错位像素
        size   裁回的目标尺寸 (w, h)，默认取源尺寸

    ⚠ 陷阱 1：**裁回的尺寸不能写死成 1080x1920**。源若是 720x1280，
              `crop=1080:1920` 会直接报尺寸非法 -> 0 字节输出（静默失败）。
              故这里用 crop 的 `iw`/`ih` 相对表达式。
    ⚠ 陷阱 2：blend 只有 2 个输入，合 3 路必须串两次，不能并列写 3 路，
              否则报 `More input link labels specified for filter 'blend'`。
    ⚠ 陷阱 3：all_mode=screen 在相位不同的三通道上会过曝，用 addition 更准。
    """
    s = int(shift)
    # 用真实源尺寸裁回；size=None 时用相对表达式，避免依赖外部已知分辨率
    crop_back = (f"crop={size[0]}:{size[1]}:0:0" if size
                 else f"crop=iw-{2 * s}:ih:0:0")
    return (f"split=3[r][g][b];"
            f"[r]lutrgb=g=0:b=0,pad=iw+{2 * s}:ih:{2 * s}:0:black,"
            f"{crop_back}[rr];"
            f"[g]lutrgb=r=0:b=0[gg];"
            f"[b]lutrgb=r=0:g=0,pad=iw+{2 * s}:ih:0:0:black,"
            f"{crop_back}[bb];"
            f"[rr][gg]blend=all_mode=addition[t1];"
            f"[t1][bb]blend=all_mode=addition")


def flash(intensity=0.6):
    """闪白（节拍点用）。用 eq 提亮配合时间表达式近似"""
    return f"eq=brightness='{intensity}*pow(2, -mod(n,30)/3)':eval=frame"


# =================================================================== 色彩 / 质感
def sigma_grade():
    """sigma 风格调色：压暗 + 暖金 + 提对比"""
    return ("curves=preset=strong_contrast,"
            "eq=saturation=0.85:contrast=1.12:brightness=-0.04,"
            "colorbalance=rs=0.06:gs=0.02:bs=-0.08")


def chad_grade():
    """gigachad 高对比黑白"""
    return ("hue=s=0,"
            "eq=contrast=1.55:brightness=0.02,"
            "unsharp=5:5:0.8:5:5:0.0")


def vhs_grade():
    """VHS/低画质 slop 质感"""
    return ("eq=saturation=1.35:contrast=1.1,"
            "tblend=all_mode=average,"  # 拖影
            "noise=alls=12:allf=t+u")


# =================================================================== 竖屏
def to_vertical():
    """9:16 竖屏 + 模糊背景填充（slop 标准规格）"""
    return (f"split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},gblur=sigma=28[bgb];"
            f"[fg]scale={W}:-2[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2")


# =================================================================== 节奏参考
BEAT_GUIDE = {
    "总时长": "15-30 秒",
    "原始素材": "20-40 秒，再修剪调速",
    "梗图插入": "每 3-5 秒一次，每次 1-3 帧",
    "关键词缩放": "每 5 秒 1 次",
    "RGB 故障": "整片仅 1 次，放在最重要 punchline",
    "画面抖动": "仅高潮时刻",
    "背景乐音量": "15-25%",
    "变速范围": "上游加速至 10x，下游减速至 0.1x",
    "曲线锚点": "首尾各 1 个 + 强调点 1 个",
    "导出": "9:16 / 30fps / 字幕烧录",
}

if __name__ == "__main__":
    print("Slop 预制剪辑模板库")
    print("=" * 60)
    print("\n[节奏参考]")
    for k, v in BEAT_GUIDE.items():
        print(f"  {k:12} {v}")
    print("\n[可用模板]")
    for name in ["speed_ramp_piecewise", "speed_ramp_smooth", "speed_const",
                 "with_motion_interpolation", "boomerang_filters", "reverse_only",
                 "square_canvas", "mirror", "mirror_concat", "shake", "rgb_split",
                 "flash", "sigma_grade", "chad_grade", "vhs_grade", "to_vertical"]:
        print(f"  {name}")
