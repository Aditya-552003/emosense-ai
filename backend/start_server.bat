@echo off
echo.
echo  =========================================
echo   EmoSense AI  --  Backend Startup
echo  =========================================
echo.

:: Kill anything still on port 8080
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8080 " ^| findstr "LISTENING"') do (
    echo  Freeing port 8080 (PID %%a) ...
    taskkill /PID %%a /F >nul 2>&1
)

:: Small wait so the port is fully released
ping -n 2 127.0.0.1 >nul

echo  Starting server on http://127.0.0.1:8080
echo  Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
uvicorn app:create_app --factory --host 127.0.0.1 --port 8080
