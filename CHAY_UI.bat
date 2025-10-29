@echo off
chcp 65001 >nul
echo ============================================================
echo 🚀 KHỞI ĐỘNG GIAO DIỆN WEB - PHÁT HIỆN VI PHẠM MŨ BẢO HIỂM
echo ============================================================
echo.
echo 📍 Giao diện sẽ mở tại: http://127.0.0.1:7860
echo.
echo ⚠️  Nhấn Ctrl+C để dừng server
echo ============================================================
echo.

cd /d "%~dp0"
py -3.13 quick_start_ui.py

pause
