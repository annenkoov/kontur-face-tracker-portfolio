@echo off
setlocal
cd /d "%~dp0"
python -m venv work\.venv
if errorlevel 1 (
  echo Не удалось создать Python-окружение. Проверь, что установлен Python 3.13 x64.
  pause
  exit /b 1
)
work\.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Не удалось установить зависимости. Проверь подключение к интернету.
  pause
  exit /b 1
)
echo Настройка завершена. Теперь запусти run.bat.
pause
