@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Please run run_enterprise_ai.bat once first to set up the environment.
    pause
    exit /b 1
)
echo Running automated unit tests (pytest) ...
"venv\Scripts\python.exe" -m pytest tests/test_pipeline.py -v
pause
