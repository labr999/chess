@echo off
chcp 65001 >nul
title Xiangqi Video Scan v1.1
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
echo Checking packages (opencv, numpy, yt-dlp, deno)...
%PY% -m pip install -q -U -r requirements_video.txt
%PY% video_scan.py %*
if exist video_results start "" "video_results"
echo.
pause
