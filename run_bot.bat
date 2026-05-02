@echo off
cd /d "%~dp0"

echo Checking for virtual environment...
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing/Updating dependencies...
pip install -r requirements.txt

echo Starting Telegram Bot...
python main.py

pause
