@echo off
chcp 65001 >nul
title Xiangqi Video Scan v1.4.0
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
set SRC=https://raw.githubusercontent.com/labr999/chess/main/xiangqi-video-scan
echo Getting latest program files from GitHub...
for %%F in (video_scan.py xq_core.py board_model.py requirements_video.txt) do (
  curl -fsSL -o "%%F.new" "%SRC%/%%F" && move /y "%%F.new" "%%F" >nul
  if exist "%%F.new" del "%%F.new"
)
if not exist video_scan.py (
  echo [ERROR] video_scan.py not found and download failed. Check internet, then run again.
  pause
  exit /b
)
echo Checking packages (opencv, numpy, yt-dlp, deno)...
%PY% -m pip install -q -U -r requirements_video.txt
%PY% video_scan.py %*
if exist video_results start "" "video_results"
echo.
pause
