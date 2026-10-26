@echo off
rem moe-cache control page for Windows: double-click this file (or a Start-menu shortcut to it). Opens http://127.0.0.1:8070 (this machine only).
rem   moe-cache-ui.bat [--ui-port N] [--llama-server PATH] [--no-browser]     options go to "moe_cache.py ui"
rem   moe-cache-ui.bat --shortcut                                            adds "moe-cache" to your Start menu (and nothing else)
rem Finds: Python (MOE_CACHE_PYTHON, else the py launcher, else a real python.exe), the plugin DLL (MOE_CACHE_LIB, else ggml-moe-cache-api3.dll /
rem ggml-moe-cache.dll next to this file or one folder up (the kit), else build\win-api3 or build\win), llama-server (LLAMA_SERVER, else on the PATH,
rem else C:\llama\llama-server.exe). Set any of these variables yourself to override.
setlocal
set "HERE=%~dp0"

if /i "%~1"=="--shortcut" (
  powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Programs')+'\moe-cache.lnk'); $s.TargetPath='%~f0'; $s.WorkingDirectory='%HERE%'; $s.Description='moe-cache control page'; $s.Save()"
  if errorlevel 1 (echo Could not create the shortcut. & pause & exit /b 1)
  echo Added "moe-cache" to the Start menu.
  exit /b 0
)

set "PY=%MOE_CACHE_PYTHON%"
if not defined PY (
  py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  rem the python.exe that Windows ships without Python only opens the Store: run it once to be sure it is real
  python -c "import sys" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.8 or newer is needed: install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then start this again.
  pause
  exit /b 1
)

if not defined MOE_CACHE_LIB (
  for %%F in ("%HERE%ggml-moe-cache-api3.dll" "%HERE%..\ggml-moe-cache-api3.dll" "%HERE%ggml-moe-cache.dll" "%HERE%..\ggml-moe-cache.dll" "%HERE%..\build\win-api3\ggml-moe-cache.dll" "%HERE%..\build\win\ggml-moe-cache.dll") do (
    if not defined MOE_CACHE_LIB if exist "%%~fF" set "MOE_CACHE_LIB=%%~fF"
  )
)
if defined MOE_CACHE_LIB (echo plugin: "%MOE_CACHE_LIB%") else (echo plugin: not found next to this file or in build\win*: the launcher looks itself; set MOE_CACHE_LIB to the DLL)

if not defined LLAMA_SERVER (
  for /f "delims=" %%L in ('where llama-server.exe 2^>nul') do if not defined LLAMA_SERVER set "LLAMA_SERVER=%%L"
)
if not defined LLAMA_SERVER if exist "C:\llama\llama-server.exe" set "LLAMA_SERVER=C:\llama\llama-server.exe"
if defined LLAMA_SERVER (echo llama-server: "%LLAMA_SERVER%") else (echo llama-server: not found: unpack the llama.cpp release to C:\llama or set LLAMA_SERVER)

%PY% "%HERE%moe_cache.py" ui %*
if errorlevel 1 pause
