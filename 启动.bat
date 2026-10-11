@echo off
chcp 65001 >nul
title Sigma Male Generator
cd /d "%~dp0"

set PY="C:\Users\Lyonl\.workbuddy-ai\binaries\python\envs\default\Scripts\python.exe"

echo ========================================
echo   Sigma Male Generator
echo ========================================
echo.
echo   1 = 手动录制（实时预览，R 开始/停止，M 换模式）
echo   2 = 自动录制（倒数后自动录 N 秒，无人值守）
echo.
echo   两种都能选"无 HUD"。做人脸特写必须用无 HUD 版 ——
echo   人脸框和文字是**烧进像素**的，后期擦不掉。
echo.
set /p MODE=选择 [1]: 
if "%MODE%"=="" set MODE=1

if "%MODE%"=="2" goto AUTO

echo.
echo   录制模式：
echo     1 = 普通（画面含人脸框 / 文字 / 状态栏）
echo     2 = 无 HUD（画面干净，适合后期做人脸特写）
echo.
set /p HUD=选择 [1]: 
if "%HUD%"=="" set HUD=1

echo.
if "%HUD%"=="2" (
    echo [无 HUD] 屏幕显示仍有完整 HUD，但录制下来的画面不含任何叠加。
    echo.
    %PY% sigma_realtime.py --bare
) else (
    %PY% sigma_realtime.py
)
goto END

:AUTO
echo.
echo   录制秒数（建议 25，人脸特写需要 15s 以上）
set /p SECS=秒数 [25]: 
if "%SECS%"=="" set SECS=25
echo.
echo   倒数秒数（留出走到镜头前的时间）
set /p CD=倒数 [5]: 
if "%CD%"=="" set CD=5
echo.
echo   会同时产出两份：
echo     竖屏合成版（剪辑用）
echo     原始分辨率版（人脸特写用，画质好 29%%）
echo.
%PY% record_auto.py --secs %SECS% --countdown %CD% --out "output\新素材_无HUD.mp4" --raw-out "output\新素材_原始.mp4"

:END
echo.
echo 程序已退出。
pause
