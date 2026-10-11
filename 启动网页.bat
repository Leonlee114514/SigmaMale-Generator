@echo off
setlocal
title Sigma Male Generator - Web UI
cd /d "%~dp0"

rem ---------------------------------------------------------------
rem Keep this file 100% ASCII. Do NOT add Chinese text here.
rem Measured on this machine: a .bat that mixes "chcp 65001" with
rem UTF-8 Chinese echo lines desyncs cmd's batch reader. Lines get
rem shredded into bogus commands -- the "set PY=" line was split
rem mid-token and the server never started. All-Chinese-in-Python
rem instead: webapp.py prints the Chinese banner itself (Python
rem uses WriteConsoleW, so it renders on any console codepage).
rem ---------------------------------------------------------------

set "PY=C:\Users\Lyonl\.workbuddy-ai\binaries\python\envs\default\Scripts\python.exe"

if not exist "%PY%" goto NOPY
if not exist "webapp.py" goto NOSCRIPT

"%PY%" webapp.py --cam 0 %*

echo.
echo [web server stopped]
pause
exit /b 0

:NOPY
echo.
echo ERROR: python interpreter not found:
echo   %PY%
echo.
echo Fix the PY line in this .bat file.
pause
exit /b 1

:NOSCRIPT
echo.
echo ERROR: webapp.py not found in:
echo   %CD%
echo.
pause
exit /b 1
