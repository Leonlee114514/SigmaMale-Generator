@echo off
chcp 65001 >nul
title 卡点成片 - 实拍 + 交错素材
cd /d "%~dp0"

set PY="C:\Users\Lyonl\.workbuddy-ai\binaries\python\envs\default\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   卡点成片 - 实拍 + 交错素材
echo ============================================================
echo.
echo   按 128BPM 拍网格自动剪辑：
echo     0-7.97s   正常速度  实拍
echo     7.97s     DROP 切换点（0.25x 慢放冲击）
echo     之后      1.35x 快剪 22 刀，实拍 ^<-^> 素材 交替
echo.
echo   特效：闪白（Drop 后每拍 0.05s）/ 轻微抖动（仅 Drop）
echo         色差 RGB 分离 / VHS 颗粒 / 结尾淡出
echo.

REM ---------- BGM ----------
if not exist "bgm" mkdir bgm
set HAVE_BGM=0
for %%E in (mp3 m4a wav aac opus flac) do (
    if exist "bgm\*.%%E" set HAVE_BGM=1
)
if "%HAVE_BGM%"=="0" (
    echo [提示] bgm 目录里还没有音频。请放入 gigachad theme：
    echo          %~dp0bgm\
    echo.
    pause
    exit /b 1
)

REM ---------- 实拍素材 ----------
echo 【1/5】实拍素材（抬头/喝水等，建议 15 秒以上）
echo.
echo   把文件拖到本窗口后回车，或直接输入路径
echo   直接回车 = 用 output 里最新的 mp4
echo.
set /p FOOT=实拍素材: 
if "%FOOT%"=="" set FOOT=

REM ---------- 交错素材 ----------
echo.
echo 【2/5】交错素材（如"2022年的剪辑 belike.mp4"）
echo   直接回车 = 用桌面上的参考片
echo.
set /p MAT=交错素材: 
if "%MAT%"=="" set MAT=%USERPROFILE%\Desktop\2022年的剪辑 belike.mp4

REM ---------- 画幅 ----------
echo.
echo 【3/5】画幅
echo   v = 1080x1920 竖屏（默认，平台标准）
echo   s = 1080x1080 方形（参考片就是方形）
echo.
set /p ASP=画幅 [v]: 
if "%ASP%"=="" set ASP=v
set ASPECT=vertical
if /i "%ASP%"=="s" set ASPECT=square
if /i "%ASP%"=="square" set ASPECT=square

REM ---------- 人脸特写 ----------
echo.
echo 【4/5】人脸特写 / 锁头运镜
echo   把人脸追踪放大裁切后再剪辑。默认是**证件照构图**（含肩）。
echo.
echo   0 = 不用（直接用原素材）
echo   2 = 人脸特写（裁切跟随，头会轻微移动）
echo   3 = 人脸特写 + 锁头（头固定在画面同一位置，背景绕着头转）
echo.
echo   前提：素材必须**不含烧死的 HUD**（人脸框/文字），后期擦不掉。
echo         请用「启动.bat - 自动录制」产出。
echo.
echo   画质提示：请把上面的"实拍素材"指向 **原始分辨率版**
echo             （新素材_原始.mp4），不要用竖屏合成版 ——
echo             合成版有一道 scale=1080:-2 会先缩小画面，人脸像素少 29%%。
echo.
set /p FC=选择 [0]: 
if "%FC%"=="" set FC=0
set FCOPT=
if "%FC%"=="2" set FCOPT=--face-crop 3.0
if "%FC%"=="3" set FCOPT=--face-crop 3.0 --face-lock

REM ---------- 切片密度 ----------
echo.
echo 【5/5】切片密度
echo   1 = 每拍一刀（0.469s，跟着底鼓，稳）
echo   2 = 每半拍一刀（0.234s，切得极碎）
echo.
set /p SL=切片密度 [1]: 
if "%SL%"=="" set SL=1

echo.
echo ------------------------------------------------------------
if "%FOOT%"=="" (
    %PY% edit_slop.py --material "%MAT%" --aspect %ASPECT% --slices %SL% %FCOPT%
) else (
    %PY% edit_slop.py --footage "%FOOT%" --material "%MAT%" --aspect %ASPECT% --slices %SL% %FCOPT%
)
echo ------------------------------------------------------------

echo.
echo 其他可调项（直接加在命令行后面）：
echo     --no-fx           关掉分镜效果（微波炉/镜像/分屏/缩放/负片）
echo     --bw 0.35         降饱和/黑白程度 0-1
echo     --cut-blur 0      关掉切换动态模糊（默认 4 帧）
echo     --flash 0         关掉闪白
echo     --shake 0         关掉抖动
echo.
echo 提示：验证成片是否真的卡在拍上，可以跑：
echo     python verify_flash.py 成片.mp4   （闪白是否压点）
echo     python verify_cuts.py  成片.mp4   （硬切是否压点）
echo     python analyze_bright.py 成片.mp4 （白闪占比是否超标）
echo.
pause
