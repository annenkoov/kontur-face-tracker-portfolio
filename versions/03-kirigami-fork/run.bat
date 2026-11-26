@echo off
setlocal
cd /d "%~dp0"
if exist work\craft-root\bin\python.exe (
  setlocal EnableDelayedExpansion
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_kirigami.ps1
  exit /b !errorlevel!
)
if not exist work\.venv\Scripts\python.exe (
  echo Не найден runtime. Запусти setup.bat для legacy Tkinter-версии или настрой Craft runtime.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0src"
work\.venv\Scripts\python.exe -m face_tracker.legacy_app
exit /b %errorlevel%
