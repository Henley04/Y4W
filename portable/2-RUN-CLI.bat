@echo off
chcp 65001 >nul
setlocal
set "PY=%~dp0python\python.exe"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
if not exist "%PY%" (
  echo [ERROR] python not found: %PY%
  echo Please extract ALL split volumes into this folder first.
  pause
  exit /b 1
)
echo ============================================================
echo   Y4W 一键 CLI 推理
echo ============================================================
echo  Tips: paste any extra y4w generate args after this script,
echo        e.g. drag a .bat shortcut or run from cmd:
echo        2-RUN-CLI.bat --max-new-tokens 32 --seed 7
echo.
set /p "PROMPT=Enter prompt (default: demo prompt): "
if "%PROMPT%"=="" set "PROMPT=a gentle lo-fi music loop with warm piano"
echo.
"%PY%" -m y4w generate --backend auto --prompt "%PROMPT%" --max-new-tokens 16 %*
echo.
pause
