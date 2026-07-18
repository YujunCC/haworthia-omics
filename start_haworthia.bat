@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_haworthia.py
) else (
  python run_haworthia.py
)
if errorlevel 1 pause
