# Diabetes Detection from Iris Images

## Project Overview
This system detects diabetes through non-invasive iris image analysis using computer vision and deep learning. The project employs a two-stage approach: iris segmentation followed by pancreatic region classification to identify diabetic patterns in eye images.

## 📁 Dataset Structure

```
eye_project/
├── 📁 dataset/                           # Main dataset directory
│   ├── 📁 data/                          # Raw image data
│   │   ├── 📁 control/                   # Control (healthy) subject images
│   │   │   ├── patient_1_left.jpg
│   │   │   ├── patient_1_right.jpg
│   │   │   └── ... (52 patients × 2 eyes)
│   │   └── 📁 diabetic/                  # Diabetic subject images
│   │       ├── patient_53_left.jpg
│   │       ├── patient_53_right.jpg
│   │       └── ... (76 patients × 2 eyes)
│   │
│   ├── 📁 masks/                         # Manual annotations for training
│   │   ├── 📁 control/                   # Control iris masks
│   │   └── 📁 diabetic/                  # Diabetic iris masks
│   │
│   └── 📁 pancreatic_masks/              # Generated ROI masks
│       ├── 📁 control/                   # Control pancreatic region masks
│       └── 📁 diabetic/                  # Diabetic pancreatic region masks
│
├── 📁 models/                            # Trained model checkpoints
│   ├── best_iris_model_3class.pth        # Iris segmentation model
│   ├── best_f1_model_fold_1.pth          # Classification model fold 1
│   ├── best_f1_model_fold_2.pth          # Classification model fold 2
│   ├── best_f1_model_fold_3.pth          # Classification model fold 3
│   ├── best_f1_model_fold_4.pth          # Classification model fold 4
│   └── best_f1_model_fold_5.pth          # Classification model fold 5
│
├── 📁 src/                               # Source code directory
│   ├── cnntrain.py                       # Classification training script
│   ├── cnnpredict.py                     # Classification prediction script
│   ├── maskstrain.py                     # Iris segmentation training
│   ├── maskspredict.py                   # Iris mask generation
│   ├── generate_masks.py                 # Pancreatic mask generation
│   ├── metrices.py                       # Model evaluation
│   ├── evaluate.py                       # Performance analysis
│   ├── data_manager.py                   # Data splitting and management
│   └── visualize_results.py              # Result visualization
│
├── 📁 results/                           # Output results
│   ├── cross_validation_results.json     # CV performance metrics
│   ├── prediction_results.csv            # Model predictions
│   └── evaluation_results.csv            # Test set evaluation
│
├── 📁 performance_analysis/              # Performance analytics
│   ├── 📁 confusion_matrices/            # Confusion matrix plots
│   ├── 📁 metrics/                       # Performance metrics
│   └── 📁 sample_results/                # Sample predictions
│
├── config.py                            # Centralized configuration
├── requirements.txt                     # Python dependencies
├── data_split_info.json                # Train/val/test splits
└── README.md                           # This documentation
```

### Dataset Characteristics
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

#### `src/evaluate.py` - Performance Analysis
- **Purpose**: Generate detailed performance analysis and visualizations
- **Features**:
  - ROC curve analysis
  - Probability distribution plots
  - Cross-validation results visualization
  - Sample prediction analysis

#### `src/visualize_results.py` - Result Visualization
- **Purpose**: Create visual outputs showing predictions with original images
- **Features**:
  - Side-by-side original and segmented images
  - Prediction overlays with confidence scores
  - Color-preserved visualization with thin borders

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
└── Save masks to test_results_masks/ directories
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

### Key Workflow Principles

1. **Academic Rigor**: No test set contamination - test data never seen during training
2. **Reproducibility**: Fixed random seeds and saved split information
3. **Medical Standard**: Patient-level splitting prevents data leakage
4. **Robust Training**: Early stopping prevents overfitting on small dataset
5. **Optimal Performance**: Validation-based threshold optimization per fold
6. **Ensemble Approach**: 5-model ensemble for improved stability
7. **Spatial Attention**: Mask-guided learning focuses on pancreatic regions

This workflow ensures scientifically sound results suitable for medical AI validation and potential clinical deployment.