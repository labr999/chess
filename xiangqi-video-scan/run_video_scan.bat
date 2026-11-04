@echo off
chcp 65001 >nul
title 象棋影片自動覆盤 v1.0
cd /d "%~dp0"
echo ========================================================
echo   象棋影片自動覆盤：YouTube 清單 → 辨識棋步 → 開局分類 → 先後手勝
echo ========================================================
set PY=python
where python >nul 2>nul || set PY=py

%PY% -c "import cv2, numpy, yt_dlp" >nul 2>nul
if errorlevel 1 (
    echo 第一次使用，安裝需要的套件（opencv、numpy、yt-dlp）...
    %PY% -m pip install -r requirements_video.txt
)
echo 更新 yt-dlp（YouTube 常改版，舊版會抓不到）...
%PY% -m pip install -U -q "yt-dlp[default]"

where deno >nul 2>nul
if errorlevel 1 (
    echo.
    echo [提示] 沒有找到 deno。新版 yt-dlp 下載 YouTube 需要它，
    echo        請另開視窗執行：winget install DenoLand.Deno  裝好後重新執行本檔。
    echo.
)

set URL=
set /p URL=請貼上 YouTube 播放清單或影片網址：
set BR=
set /p BR=私人清單才需要：借用哪個瀏覽器的登入（firefox/edge/chrome，公開或不公開清單直接按 Enter）：

if "%BR%"=="" (
    %PY% video_scan.py "%URL%"
) else (
    %PY% video_scan.py "%URL%" --browser %BR%
)
if exist video_results start "" "video_results"
echo.
pause
