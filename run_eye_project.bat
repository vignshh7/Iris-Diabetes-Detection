@echo off
chcp 65001 >nul
cls
color 0F
title Eye Project - Main Menu

rem Ensure virtual environment exists
if exist ".venv\pyvenv.cfg" (
    findstr /c:"version = 3.10." /c:"version = 3.11." /c:"version = 3.12." ".venv\pyvenv.cfg" >nul
    if errorlevel 1 (
        set "VENV_VERSION="
        for /f "tokens=2 delims==" %%A in ('findstr /b "version =" ".venv\pyvenv.cfg"') do set "VENV_VERSION=%%A"
        if defined VENV_VERSION (
            echo Existing .venv uses Python %VENV_VERSION%, which is not supported.
        ) else (
            echo Existing .venv uses an unsupported Python version.
        )
        echo Recreating .venv with Python 3.12 or 3.11...
        rmdir /s /q .venv
    )
)

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating .venv...
    set "VENV_CREATED="
    set "INTERPRETER_FOUND="

    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
        set "INTERPRETER_FOUND=1"
        "%LocalAppData%\Programs\Python\Python312\python.exe" -m venv .venv
        if not errorlevel 1 set "VENV_CREATED=1"
    )
    if not defined VENV_CREATED if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
        set "INTERPRETER_FOUND=1"
        "%LocalAppData%\Programs\Python\Python311\python.exe" -m venv .venv
        if not errorlevel 1 set "VENV_CREATED=1"
    )
    if not defined VENV_CREATED if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
        set "INTERPRETER_FOUND=1"
        "%LocalAppData%\Programs\Python\Python310\python.exe" -m venv .venv
        if not errorlevel 1 set "VENV_CREATED=1"
    )

    if not defined VENV_CREATED (
        where py >nul 2>nul
        if not errorlevel 1 (
            set "INTERPRETER_FOUND=1"
            py -3.12 -m venv .venv
            if not errorlevel 1 set "VENV_CREATED=1"
            if not defined VENV_CREATED py -3.11 -m venv .venv
            if not errorlevel 1 set "VENV_CREATED=1"
            if not defined VENV_CREATED py -3.10 -m venv .venv
            if not errorlevel 1 set "VENV_CREATED=1"
        )
    )

    if not defined VENV_CREATED (
        if defined INTERPRETER_FOUND (
            echo ❌ Python was found, but the virtual environment could not be created.
            echo Close any terminals or programs using .venv, delete the .venv folder, then run this script again.
        ) else (
            echo ❌ No supported Python interpreter was found.
            echo Please install Python 3.12 or 3.11, then run this script again.
        )
        pause
        exit /b 1
    )

    if errorlevel 1 (
        echo ❌ Failed to create .venv.
        echo Please install Python and make sure it is available on PATH, then run this script again.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Virtual environment setup is incomplete: .venv\Scripts\activate.bat was not created.
    pause
    exit /b 1
)

:main_menu
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                      👁️  EYE PROJECT SUITE  👁️                  ║
echo ║                    Diabetic Retinopathy Detection            ║
echo ╠══════════════════════════════════════════════════════════════╣
echo ║                                                              ║
echo ║  Please select an option:                                    ║
echo ║                                                              ║
echo ║  [1] 🔄 Process New Dataset        (Clean ^& Number Files)    ║
echo ║  [2] 🤖 Train CNN Models           (5-Fold Cross Validation) ║
echo ║  [3] 🎯 Test Set Evaluation       (Comprehensive Metrics)   ║
echo ║  [4] 📊 Predict Real Data          (Batch Processing)        ║
echo ║  [5] 💬 Interactive Prediction     (Manual Ground Truth)     ║
echo ║  [6] ❌ Exit                                                 ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto process_dataset
if "%choice%"=="2" goto train_cnn
if "%choice%"=="3" goto evaluate_metrics
if "%choice%"=="4" goto predict_realdata
if "%choice%"=="5" goto predict_interactive
if "%choice%"=="6" goto exit_program
goto invalid_choice

:process_dataset
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    🔄 PROCESSING NEW DATASET                 ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Starting dataset processing...
echo This will:
echo  • Clear existing dataset files
echo  • Process images from dataset_backup
echo  • Apply sequential numbering (Control: 1-N, Diabetic: N+1-end)
echo  • Handle orphaned files
echo.
pause
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Checking and installing requirements...
pip install -r requirements.txt --progress-bar on
echo Running dataset processor...
python src\process_new_dataset.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Error occurred during dataset processing!
    pause
    goto main_menu
)
echo.
echo ✅ Dataset processing completed successfully!
pause
goto main_menu

:train_cnn
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    🤖 TRAINING CNN MODELS                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Starting CNN training...
echo This will:
echo  • Create train/test/validation splits (80/20 with 20%% val from train)
echo  • Train 5-fold cross-validation models
echo  • Save best models for each fold
echo  • Generate data_split_info.json
echo.
pause
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Checking and installing requirements...
pip install -r requirements.txt --progress-bar on
echo Running CNN training...
python src\cnntrain.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Error occurred during training!
    pause
    goto main_menu
)
echo.
echo ✅ CNN training completed successfully!
pause
goto main_menu

:evaluate_metrics
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                  🎯 TEST SET EVALUATION                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Starting test set evaluation...
echo This will:
echo  • Load test patients from data_split_info.json
echo  • Evaluate with ensemble of 5 trained models
echo  • Calculate comprehensive metrics (Accuracy, Precision, Recall, F1)
echo  • Generate confusion matrix and classification report
echo  • Save results to evaluation_results.csv
echo.
pause
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Checking and installing requirements...
pip install -r requirements.txt --progress-bar on
echo Running test set evaluation...
python src\metrices.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Error occurred during evaluation!
    pause
    goto main_menu
)
echo.
echo ✅ Test set evaluation completed successfully!
pause
goto main_menu

:predict_realdata
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   📊 PREDICT REAL DATA                       ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Starting real data prediction...
echo This will:
echo  • Process all images in realdata folder
echo  • Run ensemble prediction with all 5 models
echo  • Generate prediction results CSV
echo  • Show confidence scores and probabilities
echo.
pause
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Checking and installing requirements...
pip install -r requirements.txt --progress-bar on
echo Running real data prediction...
python src\predict_realdata.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Error occurred during prediction!
    pause
    goto main_menu
)
echo.
echo ✅ Real data prediction completed successfully!
pause
goto main_menu

:predict_interactive
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                 💬 INTERACTIVE PREDICTION                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Starting interactive prediction...
echo This will:
echo  • Process images in realdata folder one by one
echo  • Ask for ground truth for each patient pair
echo  • Calculate accuracy and confusion matrix
echo  • Generate detailed results CSV
echo.
pause
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Checking and installing requirements...
pip install -r requirements.txt --progress-bar on
echo Running interactive prediction...
python src\predict_realdata_interactive.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Error occurred during interactive prediction!
    pause
    goto main_menu
)
echo.
echo ✅ Interactive prediction completed successfully!
pause
goto main_menu

:invalid_choice
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                        ❌ INVALID CHOICE                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Please enter a number between 1 and 5.
echo.
pause
goto main_menu

:exit_program
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                         👋 GOODBYE!                         ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Thank you for using the Eye Project Suite!
echo.
timeout /t 2 /nobreak >nul
exit