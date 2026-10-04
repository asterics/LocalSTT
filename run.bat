@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "VENV=.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\.deps_installed"
set "HF_HOME=%~dp0models"

rem --- find a Python 3 interpreter ---
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [ERROR] Python 3 was not found. Install Python 3.10 - 3.12 from https://www.python.org/downloads/
    echo         and enable "Add python.exe to PATH" during setup.
    goto :fail
)

rem --- create virtual environment on first run ---
if not exist "%VPY%" (
    echo [1/2] Creating virtual environment in %VENV% ...
    %PY% -m venv "%VENV%"
    if errorlevel 1 goto :fail
)

rem --- install dependencies on first run ---
if not exist "%MARKER%" (
    echo [2/2] Installing dependencies - this happens only once ...
    "%VPY%" -m pip install --upgrade pip
    if errorlevel 1 goto :fail
    "%VPY%" -m pip install faster-whisper PyAudioWPatch soxr numpy
    if errorlevel 1 goto :fail
    echo installed> "%MARKER%"
)

rem --- run the demo, passing through any arguments ---
rem     e.g.  run.bat --model medium --out protokoll.txt
"%VPY%" live_transcribe.py %*
if errorlevel 1 goto :fail

endlocal
exit /b 0

:fail
echo.
echo Something went wrong - see the messages above.
pause
endlocal
exit /b 1
