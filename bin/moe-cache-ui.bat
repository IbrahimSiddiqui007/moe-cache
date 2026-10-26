@echo off
rem moe-cache control page for Windows: double-click this file (or a Start-menu shortcut to it). Opens the page at http://127.0.0.1:8070 (this
rem machine and this user only: the address carries a per-run token).
rem   moe-cache-ui.bat [--ui-port N] [--llama-server PATH] [--no-browser]     options go to "moe_cache.py ui"
rem   moe-cache-ui.bat --shortcut                                            adds "moe-cache" to your Start menu (and nothing else)
rem Finds: Python (MOE_CACHE_PYTHON, else the py launcher, else a real python.exe), the plugin DLL (MOE_CACHE_LIB, else ggml-moe-cache-api3.dll /
rem ggml-moe-cache.dll next to this file or one folder up (the kit), else build\win-api3 or build\win), llama-server (LLAMA_SERVER, else on the PATH,
rem else C:\llama\llama-server.exe). Set any of these variables yourself to override. Programs are started by their full path, from this folder.
setlocal
cd /d "%~dp0" || (echo Cannot change to the folder of this file. & pause & exit /b 1)
set "HERE=%~dp0"
set "SYS32=%SystemRoot%\System32"
set "WHERE=%SYS32%\where.exe"
rem (SystemRoot has no spaces: where.exe is called unquoted, as for /f mangles a command line that starts with a quote)

if /i "%~1"=="--shortcut" goto shortcut

set "PY=%MOE_CACHE_PYTHON%"
if defined PY goto have_python
rem the py launcher, by its full path
for /f "delims=" %%P in ('%WHERE% py.exe 2^>nul') do if not defined PY (
  "%%~fP" -3 -c "import sys" >nul 2>nul && set "PY="%%~fP" -3"
)
if defined PY goto have_python
rem a python.exe that really runs: the one Windows ships without Python only opens the Store
for /f "delims=" %%P in ('%WHERE% python.exe 2^>nul') do if not defined PY (
  "%%~fP" -c "import sys" >nul 2>nul && set "PY="%%~fP""
)
if defined PY goto have_python
echo Python 3.8 or newer is needed: install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then start this again.
pause
exit /b 1

:have_python
if defined MOE_CACHE_LIB goto have_lib
for %%F in ("%HERE%ggml-moe-cache-api3.dll" "%HERE%..\ggml-moe-cache-api3.dll" "%HERE%ggml-moe-cache.dll" "%HERE%..\ggml-moe-cache.dll" "%HERE%..\build\win-api3\ggml-moe-cache.dll" "%HERE%..\build\win\ggml-moe-cache.dll") do (
  if not defined MOE_CACHE_LIB if exist "%%~fF" set "MOE_CACHE_LIB=%%~fF"
)
:have_lib
if defined MOE_CACHE_LIB (echo plugin: "%MOE_CACHE_LIB%") else (echo plugin: not found next to this file or in build\win*: the launcher looks itself; set MOE_CACHE_LIB to the DLL)

if defined LLAMA_SERVER goto have_ls
for /f "delims=" %%L in ('%WHERE% llama-server.exe 2^>nul') do if not defined LLAMA_SERVER set "LLAMA_SERVER=%%~fL"
if not defined LLAMA_SERVER if exist "C:\llama\llama-server.exe" set "LLAMA_SERVER=C:\llama\llama-server.exe"
:have_ls
if defined LLAMA_SERVER (echo llama-server: "%LLAMA_SERVER%") else (echo llama-server: not found: unpack the llama.cpp release to C:\llama or set LLAMA_SERVER)

%PY% "%HERE%moe_cache.py" ui %*
if errorlevel 1 pause
exit /b

:shortcut
rem single quotes in the paths are doubled for PowerShell's '...' strings
set "SELF=%~f0"
set "SELF=%SELF:'=''%"
set "DIR=%HERE:'=''%"
"%SYS32%\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Programs')+'\moe-cache.lnk'); $s.TargetPath='%SELF%'; $s.WorkingDirectory='%DIR%'; $s.Description='moe-cache control page'; $s.Save()"
if errorlevel 1 (echo Could not create the shortcut. & pause & exit /b 1)
echo Added "moe-cache" to the Start menu.
exit /b 0
