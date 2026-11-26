@echo off
setlocal
cd /d "%~dp0"
if not exist work\.venv\Scripts\python.exe (
  echo Сначала запусти setup.bat для установки приложения.
  pause
  exit /b 1
)
set PYTHONPATH=%~dp0src
work\.venv\Scripts\python.exe -m face_tracker
