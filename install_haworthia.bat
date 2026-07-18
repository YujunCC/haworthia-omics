@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.12 was not found. Install Python 3.12 and select "Add Python to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the local Python environment...
  python -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing Haworthia OMICS dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Installation completed.
echo Background segmentation models are separate third-party files.
echo Review their upstream terms before downloading.
choice /C YN /M "Download the segmentation weights from the upstream rembg release now"
if errorlevel 2 goto :skip_models
".venv\Scripts\python.exe" scripts\download_segmentation_models.py
if errorlevel 1 echo Segmentation model download failed. You can retry it later.

:skip_models
echo.
echo Double-click start_haworthia.bat to open the application.
pause
exit /b 0

:failed
echo.
echo Installation failed. Review the error above before trying again.
pause
exit /b 1
