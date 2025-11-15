@echo off
REM Quick deployment script for SuhasMartha/Parkinson-Detector (Windows)

echo 🚀 Ultimate Parkinson's Detector - Deployment Script
echo ==================================================

REM Check Python
echo ✓ Checking Python...
python --version

REM Create venv if not exists
if not exist "venv" (
    echo ✓ Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo ✓ Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo ✓ Installing dependencies...
pip install -r requirements_suhas.txt

REM Verify models
echo ✓ Verifying model files...
if exist "models\mri_model.h5" (
    echo   ✅ mri_model.h5
) else (
    echo   ❌ mri_model.h5 MISSING
)
if exist "models\drawing_model.h5" (
    echo   ✅ drawing_model.h5
) else (
    echo   ❌ drawing_model.h5 MISSING
)
if exist "models\speech_model.pkl" (
    echo   ✅ speech_model.pkl
) else (
    echo   ❌ speech_model.pkl MISSING
)
if exist "models\gait_model.pkl" (
    echo   ✅ gait_model.pkl
) else (
    echo   ❌ gait_model.pkl MISSING
)

echo.
echo ==================================================
echo ✅ Setup complete!
echo.
echo To run the app:
echo   streamlit run app.py
echo.
echo App will open at: http://localhost:8501
echo ==================================================
pause
