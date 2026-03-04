# 👁️ Eye Project: Diabetes Detection from Iris Images

> Deep learning ensemble system for diabetes detection using paired iris images with 5-fold cross-validation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Dataset Statistics](#-dataset-statistics)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This project implements an automated pipeline for diabetes detection from iris images using deep learning. The system uses a multi-channel CNN ensemble with spatial attention mechanisms to classify patients as diabetic or control based on paired left-right iris images.

### Key Highlights
- **5-Fold Cross-Validation** ensemble for robust predictions
- **Multi-channel processing** (RGB, Grayscale, HSV, LAB + mask attention)
- **Sequential dataset numbering** system for organized data management
- **Automated mask generation** for iris and pancreatic ROI extraction
- **Patient-level data splitting** to prevent data leakage
- **Easy-to-use batch menu** for all operations

---

## ✨ Features

- ✅ **Automated Pipeline**: One-click batch menu for entire workflow
- ✅ **Robust Validation**: Patient-level stratified splits (60% train, 20% val, 20% test)
- ✅ **Ensemble Learning**: 5-fold cross-validation with optimal threshold tuning
- ✅ **Multi-channel Input**: RGB + Grayscale + HSV + LAB + spatial attention
- ✅ **Performance Tracking**: Comprehensive metrics and visualizations
- ✅ **Explainability**: Grad-CAM++ heatmaps for model interpretation
- ✅ **Production Ready**: Handles new data without ground truth labels

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Patients** | 127 (52 control, 75 diabetic) |
| **Images per Patient** | 2 (left eye, right eye) |
| **Image Format** | JPEG/JPG |
| **Resolution** | 128×128 (auto-resized) |
| **Input Channels** | 40 (20 per eye) |
| **Train Split** | 75 patients (60%) |
| **Validation Split** | 26 patients (20%) |
| **Test Split** | 26 patients (20%) |

---

## 📁 Project Structure

```
eye_project/
├── run_eye_project.bat           # ⭐ Main menu system
├── config.py                      # Configuration parameters
├── requirements.txt               # Python dependencies
├── data_split_info.json          # Train/val/test patient splits
│
├── dataset/                       # Training dataset
│   ├── data/
│   │   ├── control/              # Control patients (1-52)
│   │   └── diabetic/             # Diabetic patients (53-127)
│   ├── masks/                    # Auto-generated iris masks
│   └── pancreatic_masks/         # Auto-generated ROI masks
│
├── dataset_backup/                # Original images backup
│   └── data/
│       ├── control/
│       └── diabetic/
│
├── realdata/                      # Test/new data
│   ├── images/                   # Test images location
│   ├── masks/                    # Auto-generated test masks
│   └── pancreatic_masks/         # Auto-generated test ROI masks
│
├── models/                        # Trained model checkpoints
│   ├── best_iris_model_3class.pth        # Iris segmentation
│   ├── best_f1_model_fold_1.pth          # Classification fold 1
│   ├── best_f1_model_fold_2.pth          # Classification fold 2
│   ├── best_f1_model_fold_3.pth          # Classification fold 3
│   ├── best_f1_model_fold_4.pth          # Classification fold 4
│   └── best_f1_model_fold_5.pth          # Classification fold 5
│
└── src/                           # Source code
    ├── process_new_dataset.py            # Dataset processor
    ├── cnntrain.py                       # Training script
    ├── cnnpredict.py                     # Prediction utilities
    ├── metrices.py                       # Model evaluation
    ├── predict_realdata.py               # Batch prediction
    ├── predict_realdata_interactive.py   # Interactive prediction
    ├── generate_masks.py                 # Mask generation
    ├── data_manager.py                   # Data splitting
    ├── visualize_results.py              # Result visualization
    ├── gradcam_generate.py               # Grad-CAM generation
    └── gradcam_montage.py                # Figure generation
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies (auto-installed by bat file)
pip install -r requirements.txt
```

### 2. Main Menu System

**Simply double-click `run_eye_project.bat`** to access all functions:

```
╔══════════════════════════════════════════════════════════════╗
║                   👁️  EYE PROJECT SUITE  👁️                   ║
║                 Diabetic Retinopathy Detection               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [1] 🔄 Process New Dataset      (Clean & Number Files)     ║
║  [2] 🤖 Train CNN Models         (5-Fold Cross Validation)  ║
║  [3] 🎯 Test Set Evaluation      (Comprehensive Metrics)    ║
║  [4] 📊 Predict Real Data        (Batch Processing)         ║
║  [5] 💬 Interactive Prediction   (Manual Ground Truth)      ║
║  [6] ❌ Exit                                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### 3. Complete Workflow

#### Step 1: Process Dataset (Option 1)
Place raw images in `dataset_backup/data/control/` and `dataset_backup/data/diabetic/`:
- Naming format: `XL.*.jpg` and `XR.*.jpg` (X = any number)
- Script applies sequential numbering: Control 1-52, Diabetic 53-127
- Handles orphaned files (images without L/R pairs)

#### Step 2: Train Models (Option 2)
- Auto-generates iris and pancreatic masks
- Creates train/val/test splits (60/20/20)
- Trains 5 models using K-fold cross-validation
- Saves models to `models/` directory
- Generates `data_split_info.json` for reproducibility

#### Step 3: Evaluate on Test Set (Option 3)
- Uses held-out test patients from `data_split_info.json`
- Ensemble prediction across 5 folds
- Comprehensive metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Saves results to `evaluation_results.csv`

#### Step 4: Predict on New Data (Option 4)
- Place new images in `realdata/images/`
- Auto-generates masks and runs ensemble prediction
- Outputs to `realdata_predictions.csv` (no ground truth required)

#### Step 5: Interactive Prediction (Option 5)
- Process images with manual ground truth input
- Calculates accuracy and confusion matrix
- Saves to `realdata_pair_predictions.csv`

---

## 🏗️ System Architecture

### Multi-Channel Input Processing

```
Input: Paired Iris Images (Left + Right)
    ↓
Per Eye Processing:
├── RGB channels (3)
├── Grayscale channel (1)
├── HSV channels (3)
├── LAB channels (3)
└── Mask attention (spatial)
    ↓
Total: 20 channels/eye × 2 eyes = 40 channels
```

### Model Architecture

```
Input (40 channels)
    ↓
Conv Block 1: 40→64 (3×3, GroupNorm, ReLU, MaxPool)
    ↓
Conv Block 2: 64→128 (3×3, GroupNorm, ReLU, SE, MaxPool)
    ↓
Conv Block 3: 128→256 (3×3, GroupNorm, ReLU, SE, MaxPool)
    ↓
Conv Block 4: 256→512 (3×3, GroupNorm, ReLU, SE, MaxPool)
    ↓
Global Average Pooling → FC (512→256) → Output (256→1)
```

**Key Components:**
- **SE Blocks**: Squeeze-and-Excitation for channel attention
- **GroupNorm**: Stable training with small batch sizes
- **Focal Loss**: Handles class imbalance
- **Early Stopping**: Prevents overfitting (patience=8)

### Training Strategy

```
Dataset → 5-Fold Cross-Validation
    ↓
Per Fold:
├── Train on 4 folds (with validation)
├── Optimal threshold tuning
└── Save best F1 model
    ↓
Ensemble: Average predictions across 5 folds
```

### Data Flow Pipeline

```
Phase 1: Dataset Setup
dataset_backup/ → process_new_dataset.py → dataset/data/ (sequential)

Phase 2: Training
dataset/data/ → generate_masks.py → masks/
                      ↓
                 cnntrain.py (5-fold CV)
                      ↓
                 models/ (5 checkpoints)

Phase 3: Evaluation
data_split_info.json → metrices.py → evaluation_results.csv

Phase 4: Prediction
realdata/images/ → predict_realdata.py → realdata_predictions.csv
```

---

## 📚 Usage Guide

### Command Line Usage

If you prefer manual command-line execution:

```bash
# Process dataset
python src/process_new_dataset.py

# Train models
python src/cnntrain.py

# Evaluate on test set
python src/metrices.py

# Predict on new data (batch)
python src/predict_realdata.py

# Interactive prediction
python src/predict_realdata_interactive.py

# Generate Grad-CAM heatmaps
python src/gradcam_generate.py --left <img> --right <img> --patient-class control

# Create paper figures
python src/gradcam_montage.py --patient-class control --n 10
```

### Script Details

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|  
| `process_new_dataset.py` | Dataset processor | `dataset_backup/` | `dataset/data/` |
| `cnntrain.py` | Model training | `dataset/data/` | `models/*.pth` |
| `metrices.py` | Test evaluation | `data_split_info.json` | `evaluation_results.csv` |
| `predict_realdata.py` | Batch prediction | `realdata/images/` | `realdata_predictions.csv` |
| `predict_realdata_interactive.py` | Interactive predict | `realdata/images/` | `realdata_pair_predictions.csv` |
| `generate_masks.py` | Mask generation | Image folders | Mask folders |
| `visualize_results.py` | Result visualization | Prediction CSVs | Figures (local) |
| `gradcam_generate.py` | Explainability | Image pairs | Heatmaps (local) |

---

## ⚙️ Configuration

Edit `config.py` to customize parameters:

```python
# Core settings
IMAGE_SIZE = 128                           # Input resolution
N_FOLDS = 5                                # Cross-validation folds
BATCH_SIZE = 8                             # Training batch size
LEARNING_RATE = 0.0001                     # Adam optimizer LR
EARLY_STOPPING_PATIENCE = 8                # Early stopping patience

# Multi-channel processing
CHANNELS = ['rgb', 'gray', 'hsv', 'lab', 'mask']

# Model architecture
SE_REDUCTION = 16                          # SE block reduction ratio
DROPOUT_RATE = 0.5                         # Dropout probability
```

---

## 🔧 Troubleshooting

### Common Issues

**Q: Missing masks during training**  
A: Run Option 1 (Process Dataset) first. Option 2 auto-generates masks.

**Q: No images found in realdata**  
A: Place test images directly in `realdata/images/` (no subfolders). Use format `XL.*.jpg` and `XR.*.jpg`.

**Q: CUDA out of memory**  
A: Reduce `BATCH_SIZE` in `config.py` or set `device='cpu'` in training scripts.

**Q: Data leakage detected**  
A: Always use `data_split_info.json` for evaluation. Never train and test on same patients.

**Q: Orphaned files found**  
A: Each patient needs both left (L) and right (R) images. Check local backup for unpaired images.

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~80% (26 held-out patients) |
| **Ensemble Benefit** | +5-7% over single model |
| **Inference Speed** | <1 sec/patient (GPU) |
| **AUC-ROC** | ~0.85 |

---

## 🔒 Git Structure

- **`.gitkeep` files** preserve empty folder structure
- **`.gitignore`** excludes images (`.jpg`, `.jpeg`, `.png`) and outputs
- **Tracked folders**: `dataset/`, `dataset_backup/`, `realdata/`, `models/`, `src/`
- **Not tracked**: Generated outputs (`test_results_analysis/`, `temp/`, etc.)
- Images remain local-only (not pushed to repository)

---

## 📜 License

See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Keep data splits reproducible (don't modify `data_split_info.json` manually)
2. Maintain sequential numbering convention
3. Test changes with full pipeline (process → train → evaluate)
4. Document architecture or hyperparameter changes

---

## 📞 Support

For issues:
1. Check local output folders for error logs
2. Verify folder structure matches documentation
3. Ensure all requirements are installed
4. Review `config.py` for parameter conflicts

---

**Last Updated**: March 2026  
**Python Version**: 3.8+  
**PyTorch Version**: 2.0+  
**Status**: Production Ready ✅

---

*Made by Vignesh*