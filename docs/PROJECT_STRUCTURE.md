# 🏗️ PROJECT STRUCTURE

## 📁 Organized Directory Structure

```
📦 eye_project/
├── 📁 src/                          # Source Code
│   ├── 🐍 maskstrain.py            # Phase 1: Iris segmentation training
│   ├── 🐍 maskspredict.py          # Phase 1: Iris mask generation
│   ├── 🐍 pancreaticmasks.py       # Phase 2: ROI extraction
│   ├── 🐍 cnntrain.py              # Phase 2: Classification training
│   ├── 🐍 cnnpredict.py            # Phase 2: Prediction
│   ├── 🐍 metrices.py              # Performance evaluation
│   └── 🐍 evaluate.py              # Model evaluation
│
├── 📁 models/                       # Trained Models
│   ├── 🏷️ best_iris_model_2class.pth     # Iris segmentation (2-class)
│   ├── 🏷️ best_iris_model_3class.pth     # Iris segmentation (3-class)
│   ├── 🏷️ best_f1_model_fold_1.pth       # Classification fold 1
│   ├── 🏷️ best_f1_model_fold_2.pth       # Classification fold 2
│   ├── 🏷️ best_f1_model_fold_3.pth       # Classification fold 3
│   ├── 🏷️ best_f1_model_fold_4.pth       # Classification fold 4
│   └── 🏷️ best_f1_model_fold_5.pth       # Classification fold 5
│
├── 📁 dataset/                      # Training & Test Data
│   ├── 📁 control/                  # Control subject images
│   ├── 📁 diabetic/                 # Diabetic subject images
│   ├── 📁 testing/                  # Test images set 1
│   ├── 📁 testing1/                 # Test images set 2
│   ├── 📁 masks/                    # Manual masks for training
│   └── 📁 pancreas_masks_for_training/
│       ├── 📁 control/              # Generated pancreatic masks (control)
│       └── 📁 diabetic/             # Generated pancreatic masks (diabetic)
│
├── 📁 test_results_masks/           # Generated Iris Masks
│   ├── 📁 control/                  # Control iris masks
│   ├── 📁 diabetic/                 # Diabetic iris masks
│   ├── 📁 testing/                  # Test iris masks set 1
│   └── 📁 testing1/                 # Test iris masks set 2
│
├── 📁 test_results/                 # Classification Results
│   ├── 📁 control/                  # Control classification results
│   └── 📁 diabetic/                 # Diabetic classification results
│
├── 📁 results/                      # Performance Results
│   ├── 📄 evaluation_results.csv    # Detailed evaluation results
│   ├── 📄 prediction_results.csv    # Prediction results
│   └── 📄 cross_validation_chart.json  # Cross-validation metrics
│
├── 📁 performance_analysis/         # Performance Analysis
│   ├── 📁 confusion_matrices/       # Confusion matrix plots
│   ├── 📁 sample_results/           # Sample result visualizations
│   │   ├── 📁 images/               # 10 sample result images
│   │   └── 📄 simple_visualizations_index.html
│   └── 📁 metrics/                  # Performance metrics
│
├── 📁 docs/                         # Documentation
│   ├── 📄 PROJECT_SUMMARY.md        # Project overview
│   ├── 📄 SIMPLIFIED_SAMPLES_SUMMARY.md  # Sample results info
│   └── 📄 FINAL_MASKS_SUMMARY.md    # Mask visualization info
│
├── 📁 temp/                         # Temporary Files
│   ├── 🗑️ Old scripts and utilities
│   ├── 🗑️ Generated figures
│   └── 🗑️ Cache files
│
├── 📁 .venv/                        # Virtual Environment
├── 📁 .dist/                        # Distribution files
│
├── 📄 README.md                     # Main documentation
├── 📄 readme.txt                    # Quick start guide
├── 📄 requirements.txt              # Dependencies
└── 📄 annotations.csv               # Image annotations
```

## 🚀 Quick Start

### Option 1: Use Pre-trained Models (Recommended)
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run predictions with existing models
python src/cnnpredict.py
```

### Option 2: Full Training Pipeline
```bash
# Phase 1: Iris Segmentation
python src/maskstrain.py     # Train segmentation model
python src/maskspredict.py   # Generate iris masks

# Phase 2: Classification  
python src/pancreaticmasks.py  # Generate ROI masks
python src/cnntrain.py        # Train classification model
python src/cnnpredict.py      # Run predictions
```

### View Results
- **Performance Analysis**: Open `performance_analysis/sample_results/simple_visualizations_index.html`
- **Detailed Results**: Check `results/evaluation_results.csv`
- **Sample Visualizations**: View `performance_analysis/sample_results/images/`

## 📊 Current Performance
- **Accuracy**: 92.2%
- **Sensitivity**: 94.7%
- **Specificity**: 88.5%
- **F1-Score**: 93.5%
- **AUC-ROC**: 94.9%

## 🔧 File Descriptions

### Core Scripts (`src/`)
- **maskstrain.py**: Trains U-Net for iris segmentation
- **maskspredict.py**: Generates iris masks using trained model
- **pancreaticmasks.py**: Extracts pancreatic ROI from iris masks
- **cnntrain.py**: Trains CNN classifier with 5-fold cross-validation
- **cnnpredict.py**: Performs diabetes classification predictions
- **metrices.py**: Evaluates model performance
- **evaluate.py**: Additional evaluation utilities

### Models (`models/`)
- **Iris Segmentation**: U-Net models for iris boundary detection
- **Classification**: Ensemble of 5 CNN models for diabetes detection

### Results (`results/`)
- **evaluation_results.csv**: Patient-wise results with probabilities
- **prediction_results.csv**: Prediction outputs
- **cross_validation_chart.json**: 5-fold CV performance metrics

## 🧹 Cleaned Organization
- ✅ Core functionality in `src/`
- ✅ Models organized in `models/`
- ✅ Results centralized in `results/`
- ✅ Documentation in `docs/`
- ✅ Temporary files moved to `temp/`
- ✅ Clean project root directory