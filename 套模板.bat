@echo off
chcp 65001 >nul
title Slop Apply - 一键套剪辑模板
cd /d "%~dp0"

echo ========================================
echo   Slop Apply - 给录像套剪辑模板
echo ========================================
echo.
echo   预设说明:
echo     sigma  曲线变速 + 金色调色 + 竖屏  (默认)
echo     chad   倒放 + 黑白高对比 + 正方形
echo     chaos  全特效叠加 (变速+抖动+RGB+VHS)
echo     raw    只变速 + 竖屏, 不调色
echo.
echo   直接回车 = 用 sigma 预设处理 output 里最新的 mp4
echo.

set /p PRESET=预设 [sigma/chad/chaos/raw]: 
if "%PRESET%"=="" set PRESET=sigma

echo.
echo 使用预设: %PRESET%
echo.

set PY="C:\Users\Lyonl\.workbuddy-ai\binaries\python\envs\default\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

%PY% slop_apply.py --preset %PRESET%

echo.
pause
