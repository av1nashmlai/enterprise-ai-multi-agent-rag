@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Enterprise AI Knowledge and Decision Support System
echo ======================================================================
echo  Enterprise AI Knowledge ^& Decision Support System (Edureka / Illinois Tech)
echo  One-click launcher for Windows 10 / 11
echo ======================================================================
echo.

REM ---------------------------------------------------------------------
REM  Already set up? Skip straight to launch.
REM ---------------------------------------------------------------------
if exist "venv\Scripts\streamlit.exe" goto launch

REM ---------------------------------------------------------------------
REM  [1/4] Find a compatible x64 Python (3.11 preferred, 3.10 / 3.12 accepted)
REM        Note: Must be AMD64 (x64), not ARM64 or 32-bit, because pyarrow
REM        and chromadb only publish win_amd64 pre-built wheels on PyPI.
REM        Windows 11 ARM64 runs x64 Python seamlessly via built-in emulation.
REM ---------------------------------------------------------------------
echo [1/4] Looking for Python 3.10 - 3.12 (64-bit x64) ...
set "PY="
for %%V in (3.11 3.12 3.10) do (
    if not defined PY (
        py -V:%%V -c "import sys; exit(0 if 'AMD64' in sys.version else 1)" >nul 2>&1 && set "PY=py -V:%%V"
    )
    if not defined PY (
        py -%%V -c "import sys; exit(0 if 'AMD64' in sys.version else 1)" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) and 'AMD64' in sys.version else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    for %%D in (Python311 Python312 Python310) do (
        if not defined PY if exist "%LocalAppData%\Programs\Python\%%D\python.exe" (
            "%LocalAppData%\Programs\Python\%%D\python.exe" -c "import sys; exit(0 if 'AMD64' in sys.version else 1)" >nul 2>&1 && set "PY=%LocalAppData%\Programs\Python\%%D\python.exe"
        )
    )
)

if not defined PY (
    echo.
    echo   Compatible 64-bit ^(x64^) Python 3.10 - 3.12 was not found on this PC.
    echo   ^(Note: ARM64/32-bit Python is skipped because pyarrow/chromadb require x64 wheels.^)
    echo   Attempting automatic install of Python 3.11 ^(x64^) via winget ...
    winget install -e --id Python.Python.3.11 --architecture x64 --force --accept-package-agreements --accept-source-agreements
    if errorlevel 1 goto nopython
    if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
        set "PY=%LocalAppData%\Programs\Python\Python311\python.exe"
    ) else (
        py -V:3.11 -c "import sys; exit(0 if 'AMD64' in sys.version else 1)" >nul 2>&1 && set "PY=py -V:3.11"
    )
    if not defined PY (
        echo.
        echo   Python 3.11 ^(x64^) installed. Please CLOSE this window and double-click
        echo   run_enterprise_ai.bat again so Windows picks up the new Python.
        pause
        exit /b 0
    )
)
echo       Using: %PY%
if exist "%PY%" (
    "%PY%" --version
) else (
    %PY% --version
)

REM ---------------------------------------------------------------------
REM  [2/4] Create virtual environment
REM ---------------------------------------------------------------------
echo.
echo [2/4] Creating virtual environment (venv) ...
if exist "%PY%" (
    "%PY%" -m venv venv
) else (
    %PY% -m venv venv
)
if errorlevel 1 goto fail

REM ---------------------------------------------------------------------
REM  [3/4] Install dependencies (first run only, ~3-6 minutes)
REM ---------------------------------------------------------------------
echo.
echo [3/4] Installing dependencies - this takes a few minutes on first run ...
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install --prefer-binary -r requirements.txt
if errorlevel 1 goto fail
"venv\Scripts\python.exe" -c "import streamlit, langgraph, chromadb, langchain_openai" 
if errorlevel 1 goto fail
echo       Dependencies installed successfully.

:launch
REM ---------------------------------------------------------------------
REM  [4/4] Launch the app (browser opens automatically)
REM ---------------------------------------------------------------------
if not exist ".env" if exist ".env.example" copy ".env.example" ".env" >nul
echo.
echo [4/4] Launching Streamlit at http://localhost:8501 ...
echo       Keep this window OPEN while using the app. Press Ctrl+C to stop.
echo.
"venv\Scripts\python.exe" -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false
pause
exit /b 0

:nopython
echo.
echo ----------------------------------------------------------------------
echo  Could not install Python automatically.
echo  Please install Python 3.11 ^(Windows installer 64-bit / amd64, NOT ARM64^) from:
echo     https://www.python.org/downloads/release/python-3119/
echo  IMPORTANT: tick "Add python.exe to PATH" on the first installer screen.
echo  Then double-click run_enterprise_ai.bat again.
echo ----------------------------------------------------------------------
pause
exit /b 1

:fail
echo.
echo ----------------------------------------------------------------------
echo  Setup failed. Removing the partial venv so the next run starts clean.
echo  Common fixes:
echo   - Check the internet connection (pip downloads packages).
echo   - Copy this folder from the USB stick to the Desktop and run it there.
echo   - Make sure the folder path has no special characters.
echo ----------------------------------------------------------------------
rmdir /s /q venv >nul 2>&1
pause
exit /b 1
