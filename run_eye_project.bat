@echo off
chcp 65001 >nul
cls
color 0F
title Eye Project - Main Menu

rem Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found at .venv\
    echo Please create a virtual environment first:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate
    echo    pip install -r requirements.txt
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