
## Eye Project: Diabetes Detection from Iris Images

### Overview
This project predicts diabetes from paired iris images using deep learning. It includes:
- Automated mask generation
- Ensemble CNN classification
- Robust data splits (no leakage)
- Easy prediction for new hospital data

### Project Folder Structure
```
eye_project/
├── dataset/                # Main dataset directory
│   ├── data/               # Raw image data
│   │   ├── control/        # Control (healthy) subject images
│   │   └── diabetic/       # Diabetic subject images
│   ├── masks/              # Manual annotations for training
│   │   ├── control/        # Control iris masks
│   │   └── diabetic/       # Diabetic iris masks
│   └── pancreatic_masks/   # Generated ROI masks
│       ├── control/        # Control pancreatic region masks
│       └── diabetic/       # Diabetic pancreatic region masks
├── models/                 # Trained model checkpoints
│   ├── best_iris_model_3class.pth
│   ├── best_f1_model_fold_1.pth
│   ├── best_f1_model_fold_2.pth
│   ├── best_f1_model_fold_3.pth
│   ├── best_f1_model_fold_4.pth
│   └── best_f1_model_fold_5.pth
├── src/                    # Source code directory
│   ├── cnntrain.py         # Classification training script
│   ├── cnnpredict.py       # Classification prediction script
│   ├── maskstrain.py       # Iris segmentation training
│   ├── maskspredict.py     # Iris mask generation
│   ├── generate_masks.py   # Pancreatic mask generation
│   ├── metrices.py         # Model evaluation
│   ├── evaluate.py         # Performance analysis
│   ├── data_manager.py     # Data splitting and management
│   └── visualize_results.py# Result visualization
├── test_results_analysis/  # Generated test analysis results
│   ├── images/             # Visualization plots (confusion matrix, ROC curve, etc.)
│   ├── metrics/            # Performance metrics and JSON reports
│   └── csv/                # Test predictions and evaluation results
├── realdata/               # For hospital/test images and predictions
│   ├── images/             # Place new image pairs here (flat, no subfolders)
│   ├── masks/              # Auto-generated masks for realdata images
│   ├── pancreatic_masks/   # Auto-generated pancreatic masks for realdata images
│   └── realdata_predictions.csv # Output predictions for realdata images
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── data_split_info.json    # Train/val/test splits
└── README.md               # This documentation
```

### Overview
This project predicts diabetes from paired iris images using deep learning. It includes:
- Automated mask generation
- Ensemble CNN classification
- Robust data splits (no leakage)
- Easy prediction for new hospital data

### Folder Structure
- `dataset/` — Contains all data and masks (see below)
- `models/` — Trained model weights (per fold)
- `src/` — All code (training, prediction, mask generation, etc.)
- `realdata/` — For hospital/test images (put images in `realdata/images/`)

### How to Train
1. Place your images in `dataset/data/control/` and `dataset/data/diabetic/`.
2. Run: `python src/cnntrain.py`
  - This auto-generates masks and trains 5-fold ensemble models.
  - Models are saved in `models/`.

### How to Predict on New Data
1. Put new image pairs in `realdata/images/` (no subfolders).
2. Run: `python src/predict_realdata.py`
  - Generates masks and predicts for all pairs.
  - Results saved as `realdata_predictions.csv`.

### Key Implementation Details
- Uses multiple color channels (RGB, gray, HSV, LAB) for each eye.
- Mask is used as spatial attention (not as a direct input channel).
- All splits and results are reproducible.

### Requirements
- Python 3.8+
- See `requirements.txt` for dependencies.

### Notes
- Do NOT upload actual images to GitHub. Only keep folder structure and code.
- For any new data, just add images and rerun the scripts—everything else is automated.
- **Total Patients**: 128 (52 control, 76 diabetic)
- **Images per Patient**: 2 (left eye, right eye)
- **Image Format**: JPEG
- **Resolution**: Variable (auto-resized to 128×128 for training)
- **Channels**: RGB color images
- **Annotations**: Manual iris segmentation masks for training

## 🗂️ File Usage and Purpose

### Core Training Files

#### `src/cnntrain.py` - Classification Model Training
- **Purpose**: Train diabetic classification models using 5-fold cross-validation
- **Input**: Multi-channel eye images (RGB + Gray + HSV + LAB + spatial mask attention)
- **Architecture**: Custom CNN with Squeeze-and-Excitation blocks and GroupNorm
- **Output**: 5 trained model checkpoints (one per fold)
- **Key Features**:
  - Early stopping with patience=8
  - Optimal threshold finding per fold
  - Spatial attention masking
  - Reproducible training with fixed seeds

#### `src/cnnpredict.py` - Classification Prediction
- **Purpose**: Generate predictions on test data using ensemble of trained models
- **Input**: Eye image pairs from test set
- **Process**: Load 5-fold models, ensemble predictions, apply optimal thresholds
- **Output**: CSV file with patient predictions and probabilities

#### `src/gradcam_generate.py` - Grad-CAM / Grad-CAM++ Visual Explanations
- **Purpose**: Generate Grad-CAM heatmaps to visualize which iris regions the classifier focuses on (paper-ready overlays)
- **Uses**: The *exact same preprocessing* as `src/cnnpredict.py` (multi-channel + pancreatic mask spatial attention)
- **Input**: A left-eye image + right-eye image (same patient)
- **Output**: PNG heatmaps/overlays in `gradcam_outputs/` + a JSON metadata file with probabilities/logits

**Example (recommended for research figures: average across 5 folds + per-eye isolation):**
```bash
python src/gradcam_generate.py \
  --models-dir models \
  --left dataset/data/control/10L.IMG20250525165821.jpg \
  --right dataset/data/control/10R.IMG20250525165746.jpg \
  --patient-class control \
  --method gradcampp \
  --target pred \
  --eye all \
  --overlay-size original \
  --outdir gradcam_outputs
```

**Notes**
- Use `--eye left` / `--eye right` to isolate which eye contributes most (the model concatenates left/right as channels, so isolation improves interpretability).
- If your masks are missing, the pipeline will fall back to a blank mask (outputs still generated, but attention may be less meaningful).

#### `src/gradcam_montage.py` - Paper Figures for 10 Pairs (White BG, Up/Down)
- **Purpose**: Generate *paper-ready* figures for multiple patients, where each figure has 3 columns:
  1) Original image (top=left eye, bottom=right eye)
  2) Iris masks (top=left mask, bottom=right mask)
  3) Grad-CAM (top=left-only CAM overlay, bottom=right-only CAM overlay)
- **Output**: Saves **10 separate PNG figures** into `gradcam_outputs/` + one JSON file listing the exact pairs used (reproducibility).

**Example (10 control pairs):**
```bash
python src/gradcam_montage.py \
  --patient-class control \
  --models-dir models \
  --n 10 \
  --method gradcampp \
  --target pred \
  --outdir gradcam_outputs
```

#### `src/maskstrain.py` - Iris Segmentation Training
- **Purpose**: Train U-Net model for iris segmentation
- **Architecture**: U-Net with MobileNetV2 encoder
- **Input**: Eye images with manual iris annotations
- **Output**: Trained segmentation model (`best_iris_model_3class.pth`)

#### `src/maskspredict.py` - Iris Mask Generation
- **Purpose**: Generate iris segmentation masks for all images
- **Input**: Raw eye images from control/diabetic directories
- **Process**: Apply trained segmentation model
- **Output**: Binary iris masks saved to appropriate directories

### Data Management Files

#### `src/data_manager.py` - Data Splitting and Management
- **Purpose**: Handle train/validation/test splits with no data leakage
- **Features**:
  - Stratified patient-level splitting (not image-level)
  - Reproducible splits with fixed random seeds
  - K-fold cross-validation generation
  - Split information saving/loading

#### `src/generate_masks.py` - Pancreatic Region Mask Generation
- **Purpose**: Create pancreatic region masks from iris segmentations
- **Process**:
  - Analyze iris masks to find center and radius
  - Generate annular pancreatic region (40%-85% of iris radius)
  - Account for left/right eye anatomical differences
- **Output**: ROI masks for training and inference

### Evaluation Files

#### `src/metrices.py` - Model Evaluation
- **Purpose**: Calculate comprehensive performance metrics on test set
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Sensitivity, Specificity
- **Output**: Detailed performance report and confusion matrix

#### `src/visualize_results.py` - Comprehensive Test Analysis
- **Purpose**: Generate complete test analysis with visualizations and metrics
- **Features**:
  - Confusion matrix with detailed statistics
  - ROC curve analysis
  - Probability distribution plots
  - Sample prediction visualizations
  - Comprehensive HTML reports
- **Output**: Creates timestamped results in `test_results_analysis/` with subfolders:
  - `images/`: All plots and visualizations
  - `metrics/`: JSON performance reports
  - `csv/`: Prediction results and evaluations

### Configuration Files

#### `config.py` - Centralized Configuration
- **Purpose**: Single source of truth for all paths and parameters
- **Contains**:
  - Directory paths for data, models, results
  - Training hyperparameters
  - Model architecture settings
  - Device configuration (CPU/GPU)

#### `data_split_info.json` - Split Information
- **Purpose**: Store train/validation/test patient splits
- **Format**: JSON with patient IDs for each split
- **Ensures**: Reproducible data splits across runs

#### `requirements.txt` - Dependencies
- **Purpose**: Specify exact Python package versions
- **Key Packages**: PyTorch, OpenCV, Albumentations, scikit-learn, torchmetrics

## 🔄 System Workflow and Flow

### Phase 1: Iris Segmentation Pipeline

#### Step 1: Data Preparation
```
Raw Eye Images → Manual Annotations → Training Dataset
├── Load eye images from dataset/data/control/ and dataset/data/diabetic/
├── Load corresponding manual annotations from annotations.csv
├── Split data into train/validation sets (85%/15%)
└── Apply data augmentation (rotation, flip, brightness/contrast)
```

#### Step 2: Segmentation Model Training
```
Training Images + Annotations → U-Net Training → Trained Model
├── Initialize U-Net with MobileNetV2 encoder
├── Train with Dice loss for binary segmentation
├── Monitor validation loss with early stopping
└── Save best model as best_iris_model_3class.pth
```

#### Step 3: Iris Mask Generation
```
All Images → Trained Segmentation Model → Iris Masks
├── Load trained segmentation model
├── Process all images in dataset directories
├── Generate binary iris masks
```

### Phase 2: Classification Pipeline

#### Step 4: ROI Extraction
```
Iris Masks → Geometric Analysis → Pancreatic Region Masks
├── Analyze iris masks to find center and radius
├── Generate annular pancreatic region masks
├── Inner radius: 40% of iris radius
├── Outer radius: 85% of iris radius
└── Save ROI masks to dataset/pancreatic_masks/
```

#### Step 5: Data Splitting (Academic Rigor)
```
Patient Data → Stratified Splitting → Train/Val/Test Sets
├── Patient-level stratified splitting (not image-level)
├── Train: 60%, Validation: 20%, Test: 20%
├── Ensure no patient appears in multiple splits
├── Save split information for reproducibility
└── Generate K-fold splits from train+validation data only
```

#### Step 6: Classification Training (5-Fold CV)
```
For each fold (1-5):
    ├── Multi-channel Feature Extraction:
    │   ├── RGB channels (3)
    │   ├── Grayscale channel (1)
    │   ├── HSV channels (3)
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