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
echo   Y4W 环境自检
echo ============================================================
echo.
echo --- Python / PyTorch / CUDA ---
"%PY%" -c "import sys; print('Python :', sys.version.split()[0]); import torch; print('Torch  :', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU    :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.
echo --- Y4W backends ---
"%PY%" -m y4w --list-backends
echo.
echo --- Y4W full info (JSON) ---
"%PY%" -m y4w --info
echo.
pause
