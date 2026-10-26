@echo off
rem moe-cache control page (Windows): double-click this file, or run it in a terminal. Options: --ui-port N  --llama-server PATH  --no-browser
rem Opens http://127.0.0.1:8070 (this machine only). Python 3.8+ is needed; MOE_CACHE_PYTHON picks a specific one.
set "PY=%MOE_CACHE_PYTHON%"
if "%PY%"=="" (where python >nul 2>nul && set "PY=python" || set "PY=py")
"%PY%" "%~dp0moe_cache.py" ui %*
if errorlevel 1 pause
