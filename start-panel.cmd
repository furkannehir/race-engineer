@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Create the virtual environment and install .[desktop] first. See docs\control-panel.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import PySide6, qtawesome" >nul 2>&1
if errorlevel 1 (
    echo Desktop dependencies are missing. Run: .venv\Scripts\python.exe -m pip install -e ".[desktop]"
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m race_engineer.ui
