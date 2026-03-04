
## 👁️ Eye Project: Diabetes Detection from Iris Images

### 🎯 Overview
This project predicts diabetes from paired iris images using deep learning ensemble methods. It features:
- **Automated mask generation** for iris and pancreatic ROI extraction
- **Sequential dataset numbering** system (Control: 1-N, Diabetic: N+1-end)
- **Ensemble CNN classification** with 5-fold cross-validation
- **Robust data splits** with no data leakage (train/validation/test)
- **Easy-to-use batch menu** for all operations
- **Comprehensive evaluation** metrics and visualizations

### 📊 Dataset Statistics
- **Total Patients**: 127 (52 control, 75 diabetic)
- **Images per Patient**: 2 (left eye, right eye)
- **Image Format**: JPEG/JPG
- **Resolution**: Variable (auto-resized to 128×128 for training)
- **Channels**: Multi-channel processing (RGB, Grayscale, HSV, LAB, Mask attention)
- **Data Splits**: 60% train, 20% validation, 20% test (stratified by patient)

### 📁 Project Folder Structure
```
eye_project/
├── run_eye_project.bat       # ⭐ Main menu for all operations
├── config.py                  # Centralized configuration
├── requirements.txt           # Python dependencies
├── data_split_info.json       # Train/val/test patient splits
├── README.md                  # This documentation
│
├── dataset/                   # Main training dataset
│   ├── data/                  # Patient images (sequential numbering)
│   │   ├── control/           # Control patients (1-52)
│   │   └── diabetic/          # Diabetic patients (53-127)
│   ├── masks/                 # Auto-generated iris masks
│   │   ├── control/
│   │   └── diabetic/
│   └── pancreatic_masks/      # Auto-generated pancreatic ROI masks
│       ├── control/
│       └── diabetic/
│
├── dataset_backup/            # Backup of original images
│   └── data/
│       ├── control/
│       └── diabetic/
│
├── realdata/                  # Test data for evaluation
│   ├── images/                # Place test images here
│   ├── masks/                 # Auto-generated test masks
│   └── pancreatic_masks/      # Auto-generated test ROI masks
│
├── models/                    # Trained model checkpoints
│   ├── best_iris_model_3class.pth    # Iris segmentation model
│   ├── best_f1_model_fold_1.pth      # Classification model fold 1
│   ├── best_f1_model_fold_2.pth      # Classification model fold 2
│   ├── best_f1_model_fold_3.pth      # Classification model fold 3
│   ├── best_f1_model_fold_4.pth      # Classification model fold 4
│   └── best_f1_model_fold_5.pth      # Classification model fold 5
│
└── src/                       # Source code directory
    ├── process_new_dataset.py    # Dataset processor with sequential numbering
    ├── cnntrain.py               # CNN classification training
    ├── cnnpredict.py             # CNN prediction utilities
    ├── predict_realdata.py       # Batch prediction on test data
    ├── predict_realdata_interactive.py  # Interactive prediction with manual GT
    ├── metrices.py               # Comprehensive model evaluation
    ├── generate_masks.py         # Unified mask generation
    ├── data_manager.py           # Data splitting and management
    ├── visualize_results.py      # Result visualization
    ├── gradcam_generate.py       # Grad-CAM heatmap generation
    └── gradcam_montage.py        # Paper-ready figure generation
```

### 🚀 Quick Start Guide

#### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies (done automatically by bat file)
pip install -r requirements.txt
```

#### 2. Using the Main Menu System
Simply double-click **`run_eye_project.bat`** to access all functions:

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

#### 3. Complete Workflow

**Step 1: Process Dataset (Option 1)**
- Place raw images in `dataset_backup/data/control/` and `dataset_backup/data/diabetic/`
- Naming format: `XL.*.jpg` and `XR.*.jpg` (where X is any number)
- Run Option 1 to:
  - Clear existing dataset
  - Apply sequential numbering (Control: 1-N, Diabetic: N+1-end)
  - Handle orphaned files (files without L/R pairs)
  - Preserve folder structure

**Step 2: Train Models (Option 2)**
- Automatically generates iris and pancreatic masks
- Creates train/validation/test splits (60/20/20)
- Trains 5 models using K-fold cross-validation
- Saves models to `models/` directory
- Generates `data_split_info.json` for reproducibility

**Step 3: Evaluate on Test Set (Option 3)**
- Uses held-out test patients from `data_split_info.json`
- Evaluates with ensemble of 5 trained models
- Calculates comprehensive metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix, ROC-AUC
  - Per-patient predictions with confidence
- Saves results to `evaluation_results.csv`

**Step 4: Predict on New Data (Option 4)**
- Place new images in `realdata/images/`
- Automatically generates masks
- Runs ensemble prediction
- Outputs to `realdata_predictions.csv`
- No ground truth required

**Step 5: Interactive Prediction (Option 5)**
- Process images in `realdata/images/`
- Manually input ground truth for each pair
- Calculates accuracy, confusion matrix
- Saves to `realdata_pair_predictions.csv`

### 🔧 Manual Usage (Command Line)

If you prefer command-line usage instead of the menu system:

#### Process New Dataset
```bash
python src/process_new_dataset.py
```
Clears `dataset/data/`, renumbers from `dataset_backup/`, handles orphaned files.

#### Train CNN Models
```bash
python src/cnntrain.py
```
Automatically generates masks, trains 5-fold ensemble, saves models to `models/`.

#### Test Set Evaluation
```bash
python src/metrices.py
```
Evaluates ensemble on held-out test set from `data_split_info.json`.

#### Predict Real Data (Batch)
```bash
python src/predict_realdata.py
```
Processes all images in `realdata/images/`, outputs `realdata_predictions.csv`.

#### Interactive Prediction
```bash
python src/predict_realdata_interactive.py
```
Processes images with manual ground truth input, outputs `realdata_pair_predictions.csv`.

### 📜 Script Details

#### Core Classification Scripts

**`src/cnntrain.py`** - Classification Model Training
- Train 5-fold ensemble CNN for diabetes classification
- Multi-channel input: RGB + Grayscale + HSV + LAB + mask attention
- Architecture: Custom CNN with Squeeze-and-Excitation blocks and GroupNorm
- Early stopping (patience=8), optimal threshold per fold
- Output: 5 model checkpoints in `models/`

**`src/cnnpredict.py`** - Classification Prediction Utilities
- Helper functions for ensemble prediction
- Used by `predict_realdata.py` and `metrices.py`
- Applies optimal thresholds from training

**`src/metrices.py`** - Comprehensive Test Evaluation
- Evaluates held-out test patients (from `data_split_info.json`)
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
- Output: `evaluation_results.csv` with per-patient predictions

**`src/predict_realdata.py`** - Batch Prediction on New Data
- Auto-generates masks for images in `realdata/images/`
- Ensemble prediction across 5 folds
- No ground truth required
- Output: `realdata_predictions.csv` with confidence scores

**`src/predict_realdata_interactive.py`** - Interactive Prediction
- Manual ground truth input for each pair
- Calculate accuracy and confusion matrix
- Output: `realdata_pair_predictions.csv` with performance metrics

#### Mask Generation Scripts

**`src/generate_masks.py`** - Unified Mask Generation
- Generates iris and pancreatic ROI masks
- Clears old masks before generation (prevents orphaned files)
- Creates `.gitkeep` files for git structure preservation
- Used by training and prediction scripts

#### Data Management Scripts

**`src/data_manager.py`** - Data Splitting and Management
- Patient-level stratified splitting (no data leakage)
- Reproducible splits with fixed seeds
- K-fold cross-validation generation
- Loads/saves `data_split_info.json`
- Ground truth format: `Control_X` / `Diabetic_X` patient IDs

**`src/process_new_dataset.py`** - Dataset Processor
- Sequential numbering: Control 1-N, Diabetic N+1-end
- Handles orphaned files (images without L/R pairs)
- Clears `dataset/data/` before processing
- Creates backups of unpaired files locally

#### Visualization and Explainability

**`src/visualize_results.py`** - Comprehensive Test Analysis
- Confusion matrix with detailed statistics
- ROC curve analysis
- Probability distribution plots
- Sample prediction visualizations
- Output: Generates timestamped results locally (not tracked in git)

**`src/gradcam_generate.py`** - Grad-CAM / Grad-CAM++ Visual Explanations
- Generate Grad-CAM heatmaps for explainability
- Same preprocessing as classification (multi-channel + mask attention)
- Example: `python src/gradcam_generate.py --left <image> --right <image> --patient-class control --method gradcampp`
- Output: Heatmap overlays generated locally (not tracked in git)

**`src/gradcam_montage.py`** - Paper-Ready Figures
- Generate publication-quality figures for multiple patients
- 3-column layout: Original | Mask | Grad-CAM
- Example: `python src/gradcam_montage.py --patient-class control --n 10`

### 🔄 System Architecture

#### Multi-Channel Input Processing
```
Input Image Pair (Left + Right Eye)
    ↓
Preprocessing (per eye):
├── RGB channels (3)
├── Grayscale channel (1)
├── HSV channels (3)
├── LAB channels (3)
└── Mask attention (spatial weighting)
    ↓
Total: 20 channels per eye × 2 eyes = 40-channel input
```

#### Model Architecture
```
Input (40 channels) → Conv Blocks with SE Attention → Global Pooling → FC Layers → Binary Output
├── Conv Block 1: 40→64 (3×3, GroupNorm, ReLU, MaxPool)
├── Conv Block 2: 64→128 (3×3, GroupNorm, ReLU, SE, MaxPool)
├── Conv Block 3: 128→256 (3×3, GroupNorm, ReLU, SE, MaxPool)
├── Conv Block 4: 256→512 (3×3, GroupNorm, ReLU, SE, MaxPool)
├── Global Average Pooling
├── FC Layer: 512→256 (Dropout 0.5)
└── Output Layer: 256→1 (Sigmoid)
```

#### Training Strategy
- **5-Fold Cross-Validation**: Ensures robust model performance
- **Early Stopping**: Prevents overfitting (patience=8 epochs)
- **Optimal Threshold Finding**: Per-fold threshold tuning for best F1-score
- **Ensemble Prediction**: Average predictions across 5 folds

#### Data Flow
```
Phase 1: Dataset Setup
dataset_backup/ → process_new_dataset.py → dataset/data/ (sequential numbering)

Phase 2: Training
dataset/data/ → generate_masks.py → dataset/masks/ + dataset/pancreatic_masks/
                                   ↓
                              cnntrain.py (5-fold CV)
                                   ↓
                              models/ (5 checkpoints + data_split_info.json)

Phase 3: Evaluation
data_split_info.json → metrices.py → evaluation_results.csv

Phase 4: New Data Prediction
realdata/images/ → predict_realdata.py → realdata_predictions.csv
```

### ⚙️ Configuration

#### `config.py` - Key Parameters
```python
IMAGE_SIZE = 128                           # Input image resolution
CHANNELS = ['rgb', 'gray', 'hsv', 'lab', 'mask']  # Multi-channel processing
N_FOLDS = 5                                # Cross-validation folds
BATCH_SIZE = 8                             # Training batch size
LEARNING_RATE = 0.0001                     # Adam optimizer LR
EARLY_STOPPING_PATIENCE = 8                # Early stopping patience
```

### 📊 Expected Performance
- **Test Set Accuracy**: ~80% (on 26 held-out patients)
- **Ensemble Benefit**: ~5-7% improvement over single model
- **Inference Speed**: <1 second per patient pair (GPU)

### 🔍 Troubleshooting

#### Issue: "Missing masks" during training
**Solution**: Run Option 1 (Process Dataset) first to ensure proper file structure, then Option 2 will auto-generate masks.

#### Issue: "No images found in realdata"
**Solution**: Place test images directly in `realdata/images/` (no subfolders). Use format `XL.*.jpg` and `XR.*.jpg`.

#### Issue: "CUDA out of memory"
**Solution**: Reduce `BATCH_SIZE` in `config.py` or run on CPU by setting `device='cpu'`.

#### Issue: "Data leakage detected"
**Solution**: Ensure you're using `data_split_info.json` for evaluation. Never train and test on the same patients.

#### Issue: "Orphaned files found"
**Solution**: Check local backup for unpaired images. Each patient needs both left (L) and right (R) images.

### 📝 Git Structure Preservation
- The project uses `.gitkeep` files to preserve empty folder structure in git
- `.gitignore` excludes all image files (`.jpg`, `.jpeg`, `.png`) and generated outputs
- Folder structure tracked for: `dataset/`, `dataset_backup/`, `realdata/`, `models/`, `src/`
- Not tracked: `test_results_analysis/`, `temp/`, `orphaned_backup/`, `figures_286pairs/`, `gradcam_outputs/`, `simulated_results_1000pairs/` (generated locally)
- Images and analysis results remain local-only (not pushed to repository)

### 🧪 Testing Workflow
1. **Initial Setup**: Run Option 1 to process and number dataset
2. **Training**: Run Option 2 to train 5-fold ensemble
3. **Validation**: Run Option 3 to evaluate on held-out test set
4. **New Data**: Place images in `realdata/images/`, run Option 4 for predictions
5. **Manual Verification**: Run Option 5 to interactively verify predictions

### 📜 Citation
If you use this code in your research, please ensure you:
- Document the 5-fold cross-validation methodology
- Report ensemble performance separately from individual folds
- Maintain proper train/test separation using `data_split_info.json`
- Reference the multi-channel feature extraction approach

### 📄 License
See [LICENSE](LICENSE) file for details.

### 🤝 Contributing
1. Keep data splits reproducible (never modify `data_split_info.json` manually)
2. Maintain sequential numbering convention for new datasets
3. Test changes with full pipeline (process → train → evaluate)
4. Document any architecture or hyperparameter changes

### 📞 Support
For issues or questions:
1. Check `test_results_analysis/` for detailed error logs
2. Verify folder structure matches documentation
3. Ensure Python environment has all requirements
4. Review `config.py` for parameter conflicts

---

**Last Updated**: February 2025  
**Python Version**: 3.8+  
**PyTorch Version**: 2.0+  
**Status**: Production Ready ✅
    │   ├── LAB channels (3)
    │   └── Spatial mask attention (applied, not concatenated)
    │
    ├── Model Architecture:
    │   ├── Custom CNN with SE-blocks
    │   ├── GroupNorm for batch size stability
    │   ├── Three convolutional blocks
    │   └── Binary classification output
    │
    ├── Training Process:
    │   ├── Focal loss for class imbalance
    │   ├── AdamW optimizer (lr=1e-4)
    │   ├── Early stopping (patience=8)
    │   ├── Validation-based threshold optimization
    │   └── Save best model per fold
    │
    └── Output: best_f1_model_fold_X.pth
```

#### Step 7: Ensemble Prediction
```
Test Images → Ensemble of 5 Models → Final Predictions
├── Load all 5 trained models
├── Process left-right eye pairs
├── Multi-channel feature extraction with spatial attention
├── Average predictions across models
├── Apply fold-specific optimal thresholds
└── Generate final classification with confidence scores
```

#### Step 8: Performance Evaluation
```
Predictions + Ground Truth → Comprehensive Analysis
├── Calculate performance metrics (Accuracy, F1, AUC, etc.)
├── Generate confusion matrix and ROC curves
├── Analyze prediction confidence distributions
├── Create visualizations with prediction overlays
└── Save results for clinical validation
```

### Data Flow Architecture

```
Input: Raw Eye Images (JPG)
    ↓
[Phase 1: Iris Segmentation]
    ├── U-Net Training → Iris Masks
    └── ROI Extraction → Pancreatic Masks
    ↓
[Phase 2: Classification]
    ├── Multi-channel Processing:
    │   ├── RGB → 3 channels
    │   ├── Gray → 1 channel
    │   ├── HSV → 3 channels
    │   ├── LAB → 3 channels
    │   └── Mask → Spatial attention
    │
    ├── Paired Eye Processing:
    │   └── Left + Right → 20 total channels
    │
    ├── 5-Fold Cross-Validation:
    │   ├── Fold 1 → Model 1
    │   ├── Fold 2 → Model 2
    │   ├── Fold 3 → Model 3
    │   ├── Fold 4 → Model 4
    │   └── Fold 5 → Model 5
    │
    └── Ensemble Prediction:
        ├── Average 5 model outputs
        ├── Apply optimal thresholds
        └── Generate final classification
    ↓
Output: Diabetic/Control Classification + Confidence Score
```

---
Made by Vignesh