@echo off
cd /d "%~dp0"
if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" -Data "%~1"
)
echo.
pause
