@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

echo Installing/updating dependencies...
pip install -r requirements.txt

echo Starting Eye Project Web App...
echo Open this URL in browser: http://127.0.0.1:5000
start http://127.0.0.1:5000
python webapp\app.py
