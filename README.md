# 西格玛男人生成器 · Sigma Male Generator

对着摄像头做动作、实时合成 sigma / gigachad 风格画面；录完的素材配固定 BGM，
按**节拍分析**做卡点变速，导出可直接发平台的短视频。

- **两条成片路线** —— 卡点版（128BPM 自创编排）与复刻版（照搬参考片结构）
- **内容感知选段** —— 实拍位的入点压在动作峰值上，而不是顺序取
- **硬指标验收** —— 闪白 / 硬切 / 内容匹配 / 一镜一段都有脚本量化，不看观感

> 📖 设计细节、踩坑记录与实测数据（2000+ 行）见 **[docs/详细文档.md](docs/详细文档.md)**

## 两条成片路线

| | **卡点版** | **复刻版** |
|---|---|---|
| 脚本 | `edit_slop.py` | `edit_ref.py` |
| 结构 | 实拍 ↔ 交错素材交替快剪，22 拍快剪段 | 照搬参考片 `2022年的剪辑 belike.mp4`：9 拍铺垫 + 15 镜 × 1 拍 + 2 拍结尾 |
| 拍网格 | **128 BPM** | **67.12 BPM**（参考片实测裁定） |
| 逐镜特效 | 微波炉 / 镜像 / 缩放脉冲 / 负片闪 + 原样 | 无（硬切到底，只有黑白镜） |
| 全局特效 | 调色 / 色差 / VHS 颗粒 / 闪白 / 抖动 / 切换模糊 | 调色 / 色差 / 颗粒 / 结尾淡出 |
| 默认画幅 | 竖屏 1080×1920 | 方形 1080×1080 |

另外两个小工具：`bgm_pipeline.py`（单素材变速到 N 个强拍 + 配乐）、
`slop_apply.py`（只套画面模板、不配乐）。

## 快速开始

```bash
cd "C:/Users/Lyonl/Desktop/SigmaMaleGenerator"
PY="C:/Users/Lyonl/.workbuddy-ai/binaries/python/envs/default/Scripts/python.exe"

# 1) 录制实拍素材（对镜头做动作，按 R 开始 / 停止）
"$PY" sigma_realtime.py

# 2) 拼交错素材 —— 两个模板的入点步长不同，必须各拼一份（见下节）
"$PY" make_material_mix.py --template slop     # 卡点版
"$PY" make_material_mix.py --template ref      # 复刻版

# 3) 出片
"$PY" edit_slop.py --material output/交错素材_贝特曼混剪.mp4 --aspect vertical
"$PY" edit_ref.py  --material output/交错素材_贝特曼混剪_复刻.mp4

# 4) 验收（硬指标，不看观感）
"$PY" verify_contentaware.py output/卡点_竖屏_v18.mp4 --plan output/卡点_竖屏_v18.plan.json
"$PY" verify_flash.py        output/卡点_竖屏_v18.mp4
"$PY" verify_one_seg.py      output/卡点_竖屏_v18.plan.json
"$PY" verify_ref.py          output/复刻版_v13.mp4
```

**一条命令跑完**（推荐，验收作门禁 —— 不过就非零退出）：

```bash
"$PY" slop_pipeline.py --template both --material <交错素材.mp4> --face-crop --grade
```

也可以双击 `启动.bat`（录制）/ `卡点成片.bat` / `套模板.bat` / `启动网页.bat`。

## 常用参数

| 脚本 | 参数 | 说明 |
|---|---|---|
| `edit_slop` / `edit_ref` | `--material` | 交错素材 mp4（**必填**） |
| | `--footage` | 实拍素材；有 `--clips` 时可省 |
| | `--clips` | `clip_extract.py` 产出的 `reel_map.json`，接入**内容感知取段** |
| | `--aspect` | `vertical`（竖屏 1080×1920）/ `square`（方形 1080×1080） |
| | `--dump-plan` | 导出 `plan.json`（验收脚本的输入） |
| `edit_slop` | `--action-plan "4=举手,6=后退"` | 指定某个**槽位**用哪个动作（编号看 `--dry-run` 打印的槽位表） |
| | `--slices` | 快剪段切片密度 |
| | `--grain` / `--rgb` / `--shake` | VHS 颗粒 / 色差 / 抖动（0 = 关） |
| `edit_ref` | `--snap-scope` | 硬切吸附范围 |
| | `--fade` / `--volume` | 结尾淡出秒数 / BGM 音量 |
| `make_material_mix` | `--template {slop,ref}` | 按模板生成对应段长的素材 |
| `slop_pipeline` | `--template {ref,slop,both}` | 出哪条路线 |
| | `--face-crop [ZOOM]` / `--face-lock` | 人脸特写 / 加锁头运镜 |
| | `--grade` | 先调色再出片 |
| | `--dry-run` / `--no-verify-gate` | 只打印命令 / 验收不过也返回 0（**仅排查用**） |

## 素材怎么拼（`make_material_mix.py`）

素材是**从头顺序消耗**的（每个素材镜推进一个固定步长），所以「段长」必须**精确等于**
该模板的入点步长：

| 模板 | 入点步长 = 段长 | 产物 |
|---|---|---|
| 卡点版 | `0.6328125s`（= 60/128 × 1.35） | `交错素材_贝特曼混剪.mp4` |
| 复刻版 | `0.8939213s`（= 60/67.12） | `交错素材_贝特曼混剪_复刻.mp4` |

两者不可通约 → **必须各拼一份**。段长错了会出现「一镜跨两段」（画面中途跳）或
「一镜被切成两段」。脚本按**帧号**裁剪（不按秒，避免浮点边界多裁 1 帧），
并把段帧数按入点网格分配（复刻版是 26/27 帧混合）。

四个源各取所长：④ 招牌（面具）· ③ 干净原片 · ② 场景 · ① 表情。
**挑镜头有三条硬约束**（每条都对应踩过的坑，详见详细文档）：

1. **整窗落在同一个镜头内** —— 否则成片里「一镜中途换场景」
2. **避开烧死的字幕 / 水印** —— 源片**右上角**有（③ 中文字幕、② 周大福水印），
   顶部和底部都要查
3. **避开失焦 / 主体不清的帧**

## 验收脚本

| 脚本 | 查什么 | 判据 |
|---|---|---|
| `verify_contentaware.py` | 内容感知选段对不对 | 对照式内容匹配（成片镜 vs 入点 / 入点−1.2s） |
| `verify_ref.py` | 复刻版硬切 / 黑白镜 / 淡出 / Drop 对齐 | 硬切落在拍上、淡出到黑、Drop 偏差 ≤150ms |
| `verify_flash.py` | 闪白是否压在拍上 | 闪白事件间距 vs 拍周期 |
| `verify_cuts.py` | 硬切是否压在拍上 | 帧间差切点 vs 理论拍点 |
| `verify_one_seg.py` | **一镜一段** | 判据 A 整镜落一段；判据 B 镜内帧差峰值须远小于「镜首那一刀」 |

## 测试

```bash
"$PY" test_slot_map.py      # ★ 纯色合成源：逐格核对"这格用哪个源"（约 30s）
"$PY" test_action_plan.py   # --action-plan 解析与落位
"$PY" test_film_name.py     # 成片命名规则
"$PY" test_beat.py test_beat_sync.py test_audio.py test_bgm_pipeline.py
cd slop_templates && "$PY" test_templates.py   # 19 条 ffmpeg 滤镜链逐条实跑
```

> ⚠ 算法测试全绿**不代表**成片正确。`test_slot_map.py` 是补上的，因为其余测试
> 漏掉了整整一类错：逐镜滤镜改了段落时序、导致「每格放的是哪个源」整体错位。
> **加任何逐镜效果后都要跑它。**

## 目录结构

```
SigmaMaleGenerator/
├── sigma_realtime.py       # 主程序：实时合成 + 录制（无声素材）
├── edit_slop.py            # ★ 卡点剪辑器（128BPM 自创编排）
├── edit_ref.py             # ★ 参考片完全复刻器（67.12BPM）
├── make_material_mix.py    # ★ 交错素材拼装（按模板段长，一镜一段）
├── content_aware.py        # ★ 内容感知取段助手（两个剪辑器共用）
├── film_name.py            # ★ 成片命名规则（共用）
├── slop_pipeline.py        # ★ 全链编排：检测→切片→成片→验收（验收作门禁）
├── action_detect.py        # 动作检测：人脸 478 点 + 表情 52 系数 + 姿态 33 点
├── clip_extract.py         # 摘片段 + 去重分区 + 素材带 + 映射表
├── face_crop.py            # 人脸特写 / 锁头运镜
├── lockhead.py             # 锁头运镜（独立实现）
├── verify_*.py             # 验收脚本（见上表）
├── test_*.py               # 单元 / 集成测试
├── webapp.py + web/        # ★ 本地前端：预览流 + 状态/产物接口
├── slop_templates/         # slop 画面模板库（滤镜链 + 预览）
├── bgm/                    # BGM 与节拍缓存 json（音频文件需自行放入）
├── models/                 # MediaPipe 模型（人脸 478 / 姿态 33）
├── docs/详细文档.md         # ★ 详细设计文档
└── output/                 # 输出目录（渲染产物，未纳入版本控制）
```

程序用 `BASE = os.path.dirname(os.path.abspath(__file__))` 动态定位资源，
**整个文件夹可以随意移动或改名**。

## 依赖

- Python 3.13 + `mediapipe` + `opencv-contrib-python` + `numpy`
- **ffmpeg**（本机用 FlyingMouse Format 内置的那份；**没有独立的 `ffprobe`**，
  读元数据统一走 `ffmpeg -i` 的 stderr，且它的退出码总是 1）
- `librosa`（可选，交叉验证节拍用）

## 已知限制

- 单人检测（`num_faces=1` / `num_poses=1`）
- 录制时帧数据全部留在内存（15fps 720p 约 33MB/秒），长时间录制占内存
- 卡点版的切换点（第 64 拍）与 128BPM 是**针对当前这首 BGM 实测写死的**，
  换歌需重跑 `analyze_sections.py` + `refine_phase.py` 更新常量
- 强拍假设 4/4 拍；3/4 拍的曲子需要调
- 尚无「梗图插入」层（slop 语法里每 3-5 秒插 1-3 帧梗图）

## 说明

本仓库**不含**渲染产物、BGM 音频与第三方影片素材（见 `.gitignore`）——
那些需要自行准备。代码本身用于个人视频剪辑实验。
