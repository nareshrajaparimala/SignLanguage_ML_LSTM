@echo off
echo ===================================================
echo   Sign Language Recognition - Windows Auto-Runner
echo ===================================================

:: Check for Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not added to PATH!
    echo Please install Python 3.10 or higher from python.org and check "Add Python to PATH".
    pause
    exit /b
)

:: Check for Node.js
echo [2/4] Checking Node.js installation...
node --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed or not added to PATH!
    echo Please install Node.js from nodejs.org.
    pause
    exit /b
)

echo.
echo [3/4] Setting up Python Backend...
cd ProjectPhase2\sign_language_recognition\camera_backend
IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo Installing/Updating Python dependencies...
pip install -r requirements_advanced.txt
pip install pyserial
cd ..\..\..\..

echo.
echo [4/4] Setting up React Frontend...
cd ProjectPhase2\sign_language_recognition\frontend
echo Installing Node dependencies...
call npm install
cd ..\..\..\..

echo.
echo ===================================================
echo   Setup Complete! Starting Servers...
echo   - Backend terminal will open in a new window
echo   - Frontend will open in your default web browser
echo ===================================================

:: Start backend in a new window
start cmd /k "cd ProjectPhase2\sign_language_recognition\camera_backend && call venv\Scripts\activate.bat && python advanced_mediapipe_server.py"

:: Start frontend in this window
cd ProjectPhase2\sign_language_recognition\frontend
call npm run dev

pause
