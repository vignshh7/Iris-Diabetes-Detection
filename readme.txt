# DIABETES DETECTION FROM IRIS IMAGES - QUICK START GUIDE

## PROJECT OVERVIEW
This project implements a two-phase deep learning system for detecting diabetes from iris images using advanced computer vision techniques.

## EXECUTION PHASES

### PHASE 1: IRIS SEGMENTATION MODEL
STEP 1: Train iris segmentation model using manual binary masks
        Command: python maskstrain.py
        Output: best_iris_model_3class.pth

STEP 2: Generate iris masks for all image directories
        Command: python maskspredict.py
        Output: Binary masks in test_results_masks/

### PHASE 2: ROI EXTRACTION AND CLASSIFICATION MODEL
STEP 3: Generate pancreatic region masks from iris masks
        Command: python pancreaticmasks.py
        Output: Pancreatic masks in dataset/pancreas_masks_for_training/

STEP 4: Train classification model using 5-fold cross-validation
        Command: python cnntrain.py
        Output: 5 model files (best_f1_model_fold_*.pth)

STEP 5: Run diabetes classification predictions
        Command: python cnnpredict.py
        Output: prediction_results.csv

## PERFORMANCE ANALYSIS
For comprehensive performance evaluation:
        Command: python performance_analysis_generator.py
        Command: python detailed_sample_generator.py
        Command: python comprehensive_metrics_generator.py

## CURRENT PERFORMANCE RESULTS
- Accuracy: 92.2%
- Sensitivity: 94.7%
- Specificity: 88.5%
- F1-Score: 93.5%
- AUC-ROC: 94.9%

## QUICK EXECUTION (Using Pre-trained Models)
If models are already trained, run predictions directly:
        python cnnpredict.py

## ENVIRONMENT SETUP
PS C:\Users\Lenovo> cd Desktop/eye_project
PS C:\Users\Lenovo\Desktop\eye_project> .\.venv\Scripts\Activate
pip install -r requirements.txt

## OUTPUT STRUCTURE
- performance_analysis/: Comprehensive performance metrics and visualizations
- test_results_masks/: Generated iris segmentation masks
- evaluation_results.csv: Detailed prediction results with probabilities
- Generated model files: Pre-trained models for immediate use

For detailed documentation, see README.md 