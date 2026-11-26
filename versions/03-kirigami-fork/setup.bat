@echo off
setlocal
cd /d "%~dp0"
if exist work\craft-root\bin\python.exe (
  echo Устанавливаются ML-зависимости в Craft Python без замены Qt runtime.
  work\craft-root\bin\python.exe -I -m pip install -r requirements-craft-ml.txt
  if errorlevel 1 (
    echo Не удалось установить ML-зависимости.
    pause
    exit /b 1
  )
  work\craft-root\bin\python.exe -I -m pip install --no-deps hsemotion-onnx==0.3.1
  if errorlevel 1 (
    echo Не удалось установить hsemotion-onnx.
    pause
    exit /b 1
  )
  echo Craft ML зависимости готовы. Py-Feat остается необязательной установкой.
  pause
  exit /b 0
)
echo Устанавливается legacy Tkinter-версия в work\.venv.
echo Для версии Kirigami сначала подготовь Craft runtime через tools\craft_setup.ps1.
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
