# 👁️ Diabetes Detection from Iris Images

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)

> **Advanced Computer Vision System for Non-Invasive Diabetes Detection through Iris Analysis**

## Overview

This project implements a comprehensive deep learning system for detecting diabetes mellitus from iris images using advanced computer vision techniques. The system employs a two-phase approach combining iris segmentation and diabetic pattern classification to achieve robust and accurate diabetes detection.

## 📋 Table of Contents

- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Implementation Phases](#implementation-phases)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Instructions](#usage-instructions)
- [Performance Analysis](#performance-analysis)
- [Results](#results)
- [Technical Specifications](#technical-specifications)
- [Contributing](#contributing)

## 🏗️ Project Architecture

### Two-Phase Approach

**Phase 1: Iris Segmentation Model**
- **Objective**: Precise segmentation of iris regions from eye images
- **Model**: U-Net with MobileNetV2 encoder
- **Input**: RGB eye images (256×256)
- **Output**: Binary masks highlighting iris regions
- **Training**: Manual binary masks with data augmentation

**Phase 2: Diabetic Pattern Classification**
- **Objective**: Classify diabetic vs. control subjects based on pancreatic region patterns
- **Model**: Custom CNN with Squeeze-and-Excitation blocks
- **Input**: Multi-channel feature representations (RGB, Grayscale, HSV, LAB, Mask)
- **Output**: Binary classification (Diabetic/Control)
- **Training**: 5-fold cross-validation with ensemble learning

## 📊 Dataset

### Image Specifications
- **Format**: JPEG images
- **Resolution**: Variable (resized to 256×256 for segmentation, 128×128 for classification)
- **Channels**: RGB
- **Eye Types**: Left and Right eye pairs
- **Classes**: Control and Diabetic subjects

### Dataset Composition
- **Training Images**: Located in `dataset/` directory
- **Control Subjects**: `dataset/control/`
- **Diabetic Subjects**: `dataset/diabetic/`
- **Test Images**: `dataset/testing/` and `dataset/testing1/`
- **Manual Masks**: `dataset/masks/` (for iris segmentation training)

### Annotations
- **Format**: CSV file with VGG Image Annotator format
- **File**: `annotations.csv`
- **Content**: Spatial coordinates for iris regions
- **Total Annotations**: 117 annotated images

## 🔄 Implementation Phases

### Phase 1: Iris Segmentation

#### Step 1: Model Training (`maskstrain.py`)
```python
# Key Features:
- U-Net architecture with MobileNetV2 encoder
- Dice loss for binary segmentation
- Data augmentation (rotation, flip, brightness/contrast)
- Train/validation split: 85%/15%
- Model saved as: best_iris_model_3class.pth
```

#### Step 2: Mask Generation (`maskspredict.py`)
```python
# Inference Process:
- Load trained segmentation model
- Process control and diabetic image directories
- Generate binary masks for iris regions
- Save masks in test_results_masks/
```

### Phase 2: ROI Extraction and Classification

#### Step 3: Pancreatic Mask Generation (`pancreaticmasks.py`)
```python
# ROI Extraction:
- Use iris masks to identify iris center and radius
- Generate pancreatic region masks (annular region)
- Inner radius: 40% of iris radius
- Outer radius: 85% of iris radius
- Account for left/right eye differences
```

#### Step 4: Classification Training (`cnntrain.py`)
```python
# Training Features:
- 5-fold stratified cross-validation
- Multi-channel input (RGB + Gray + HSV + LAB + Mask)
- Custom CNN with SE blocks
- Focal loss for class imbalance
- Ensemble of 5 models (one per fold)
```

#### Step 5: Prediction (`cnnpredict.py`)
```python
# Prediction Process:
- Load ensemble of trained models
- Process left-right eye pairs
- Multi-channel feature extraction
- Ensemble averaging for final prediction
```

## 🧠 Model Architecture

### Iris Segmentation Model (U-Net)
```
Encoder: MobileNetV2 (ImageNet pretrained)
├── Decoder: Upsampling blocks
├── Skip connections for feature preservation
├── Output: 1 channel (binary mask)
└── Loss: Dice Loss
```

### Classification Model (Custom CNN)
```
Input: Multi-channel features (22 channels total)
├── Conv Block 1: 32 filters + SE Block + MaxPool
├── Conv Block 2: 64 filters + SE Block + MaxPool  
├── Conv Block 3: 128 filters + SE Block + MaxPool
├── Global Average Pooling
├── FC Layer: 256 neurons + Dropout
└── Output: 1 neuron (binary classification)
```

### Squeeze-and-Excitation (SE) Block
```
Input Features → Global Average Pooling
├── FC1: Reduction (r=4)
├── ReLU
├── FC2: Expansion
├── Sigmoid
└── Channel-wise multiplication with input
```

## 📁 Project Structure

```
eye_project/
├── 📁 dataset/
│   ├── 📁 control/              # Control subject images
│   ├── 📁 diabetic/             # Diabetic subject images
│   ├── 📁 testing/              # Test images set 1
│   ├── 📁 testing1/             # Test images set 2
│   ├── 📁 masks/                # Manual masks for training
│   ├── 📁 pancreas_masks_for_training/
│   │   ├── 📁 control/          # Pancreatic masks for control
│   │   └── 📁 diabetic/         # Pancreatic masks for diabetic
│   └── 📁 pancreas_masks_for_testing/
│       ├── 📁 testing/          # Test pancreatic masks
│       └── 📁 testing1/         # Test pancreatic masks
│
├── 📁 test_results_masks/       # Generated iris masks
│   ├── 📁 control/
│   ├── 📁 diabetic/
│   ├── 📁 testing/
│   └── 📁 testing1/
│
├── 📁 performance_analysis/     # Performance evaluation results
│   ├── 📁 confusion_matrices/   # Confusion matrix visualizations
│   ├── 📁 sample_results/       # Sample prediction results
│   └── 📁 metrics/              # Performance metrics and plots
│
├── 📁 generated_figures/        # Generated visualizations
├── 📁 gradcam_outputs/          # Grad-CAM heatmaps
├── 📁 heatmaps_output/          # Additional heatmaps
├── 📁 test_data/                # Test data
└── 📁 test_results/             # Test results
    ├── 📁 control/
    └── 📁 diabetic/

# Core Python Scripts
├── 🐍 maskstrain.py             # Phase 1: Iris segmentation training
├── 🐍 maskspredict.py           # Phase 1: Iris mask generation
├── 🐍 pancreaticmasks.py        # Phase 2: Pancreatic ROI extraction
├── 🐍 cnntrain.py               # Phase 2: Classification training
├── 🐍 cnnpredict.py             # Phase 2: Diabetic classification
├── 🐍 metrices.py               # Model evaluation metrics
├── 🐍 evaluate.py               # Model evaluation script
├── 🐍 images.py                 # Visualization generation
├── 🐍 rgbtograycrop.py          # Image preprocessing utilities
├── 🐍 test.py                   # Testing utilities
├── 🐍 performance_analysis_generator.py  # Performance analysis

# Model Files
├── 🏷️ best_iris_model_2class.pth      # 2-class iris segmentation model
├── 🏷️ best_iris_model_3class.pth      # 3-class iris segmentation model
├── 🏷️ best_f1_model_fold_1.pth        # Classification model fold 1
├── 🏷️ best_f1_model_fold_2.pth        # Classification model fold 2
├── 🏷️ best_f1_model_fold_3.pth        # Classification model fold 3
├── 🏷️ best_f1_model_fold_4.pth        # Classification model fold 4
└── 🏷️ best_f1_model_fold_5.pth        # Classification model fold 5

# Data Files
├── 📄 annotations.csv           # Image annotations
├── 📄 evaluation_results.csv    # Model evaluation results
├── 📄 prediction_results.csv    # Prediction results
├── 📄 cross_validation_chart.json  # Cross-validation results
├── 📄 requirements.txt          # Python dependencies
└── 📄 README.md                 # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Minimum 8GB RAM
- 10GB storage space

### Step 1: Clone Repository
```bash
git clone https://github.com/vignshh7/Iris-Diabetes-Detection.git
cd Iris-Diabetes-Detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
albumentations>=1.3.0
segmentation-models-pytorch>=0.3.3
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
torchmetrics>=1.0.0
grad-cam>=1.4.8
```

## 🚀 Usage Instructions

### Option 1: Complete Pipeline (Recommended)
If you want to run the entire pipeline from scratch:

#### Phase 1: Iris Segmentation
```bash
# Step 1: Train iris segmentation model
python maskstrain.py

# Step 2: Generate iris masks for all images
python maskspredict.py
```

#### Phase 2: Classification
```bash
# Step 3: Generate pancreatic region masks
python pancreaticmasks.py

# Step 4: Train classification model
python cnntrain.py

# Step 5: Run predictions
python cnnpredict.py
```

### Option 2: Using Pre-trained Models (Quick Start)
If you want to use the existing trained models:

```bash
# Generate predictions using pre-trained models
python cnnpredict.py

# Or run evaluation
python metrices.py
```

### Option 3: Performance Analysis
```bash
# Generate comprehensive performance analysis
python performance_analysis_generator.py
```

## 📈 Performance Analysis

The project includes comprehensive performance analysis tools that generate:

### 1. Confusion Matrix
- Visual representation of classification results
- True Positives, False Positives, True Negatives, False Negatives
- Accuracy, Sensitivity, Specificity metrics

### 2. ROC Curve Analysis
- Receiver Operating Characteristic curve
- Area Under Curve (AUC) calculation
- Optimal threshold determination

### 3. Probability Distribution Analysis
- Histogram of prediction probabilities
- Box plots by class
- Violin plots for distribution shape
- Cumulative distribution functions

### 4. Sample Results Table
- 10 representative samples with results
- Correct and incorrect predictions
- Confidence levels (Low/Medium/High)
- Patient IDs and image pairs

### 5. Cross-Validation Results
- 5-fold cross-validation performance
- Box plots of metric distributions
- Mean and standard deviation reporting

## 🎯 Results

### Model Performance Metrics

#### Cross-Validation Results (5-Fold)
| Metric | Mean ± Std | Range |
|--------|------------|-------|
| **Accuracy** | 92.4% ± 0.3% | 92.0% - 92.8% |
| **Sensitivity** | 93.1% ± 0.5% | 92.5% - 93.7% |
| **Specificity** | 91.7% ± 0.2% | 91.5% - 92.0% |
| **F1-Score** | 92.5% ± 0.3% | 92.1% - 92.9% |

#### Overall Test Performance
- **Total Samples**: 148 patients
- **Control Subjects**: 65 patients  
- **Diabetic Subjects**: 83 patients
- **Overall Accuracy**: 92.6%
- **AUC-ROC**: 0.965

### Key Findings
1. **High Sensitivity**: 93.1% - Excellent detection of diabetic cases
2. **High Specificity**: 91.7% - Low false positive rate for control subjects
3. **Balanced Performance**: Consistent results across both classes
4. **Robust Model**: Low standard deviation across folds indicates stability

## 🔧 Technical Specifications

### Hardware Requirements
- **GPU**: NVIDIA GTX 1060 or better (6GB+ VRAM)
- **CPU**: Intel i5 or AMD Ryzen 5 equivalent
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### Software Environment
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **CUDA**: 11.0+ (for GPU acceleration)

### Model Specifications
- **Segmentation Model Size**: ~25MB
- **Classification Model Size**: ~15MB per fold (75MB total)
- **Training Time**: ~2-4 hours per phase (GPU)
- **Inference Time**: ~0.5 seconds per image pair

### Input/Output Specifications
- **Input Image Format**: JPEG, PNG
- **Input Resolution**: Variable (auto-resized)
- **Output Format**: CSV results, PNG visualizations
- **Batch Processing**: Supported for large datasets

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{diabetes_iris_detection_2025,
  title={Diabetes Detection from Iris Images using Deep Learning},
  author={Vignesh Venaktesan},
  year={2025},
  note={GitHub repository},
  url={https://github.com/vignshh7/Iris-Diabetes-Detection}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config files
   - Use CPU mode by setting `DEVICE = "cpu"`

2. **Missing Dependencies**
   - Ensure all packages in requirements.txt are installed
   - Use virtual environment to avoid conflicts

3. **Path Errors**
   - Verify dataset directory structure matches documentation
   - Use absolute paths if relative paths fail

4. **Model Loading Errors**
   - Ensure model files are not corrupted
   - Check device compatibility (CPU/GPU)

### Support

For issues and questions:
- 🐛 [GitHub Issues](https://github.com/vignshh7/Iris-Diabetes-Detection/issues)
- 📖 [Documentation](https://github.com/vignshh7/Iris-Diabetes-Detection/tree/main/docs)
- 💬 Contact via GitHub profile

## 📞 Contact

- **Author**: Vignesh Venaktesan
- **GitHub**: [@vignshh7](https://github.com/vignshh7)
- **Repository**: [Iris-Diabetes-Detection](https://github.com/vignshh7/Iris-Diabetes-Detection)
- **Date**: October 2025

---

**Note**: This project is for research purposes. Clinical validation required before medical use.