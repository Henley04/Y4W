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
echo   Y4W 一键推理服务  ^(Gradio WebUI + REST API^)
echo ============================================================
echo.
echo  Starting server on http://127.0.0.1:8000
echo  WebUI   : http://127.0.0.1:8000/
echo  API docs: http://127.0.0.1:8000/docs
echo  Backend : auto ^(CUDA -^> TensorRT-RTX -^> Vulkan -^> CPU^)
echo.
echo  Press Ctrl+C in this window to stop the server.
echo.
"%PY%" -m y4w serve --host 127.0.0.1 --port 8000 %*
echo.
echo Server stopped.
pause
