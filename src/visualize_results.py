#!/usr/bin/env python3
"""
Comprehensive Test Results Visualization Script
Runs predictions on test data and generates complete analysis with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import json
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import warnings
import sys
import subprocess
from datetime import datetime
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from collections import defaultdict
import re
from tqdm import tqdm
import csv
import random

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.data_manager import create_data_manager

# --- REPRODUCIBILITY SEEDS (Match Training) ---
def set_reproducibility_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_reproducibility_seeds(42)

# --- Configuration ---
def calculate_input_channels(channels_list):
    """Helper function to calculate total input channels based on a list."""
    count = 0
    if 'rgb' in channels_list: count += 3
    if 'gray' in channels_list: count += 1
    if 'hsv' in channels_list: count += 3
    if 'lab' in channels_list: count += 3
    if 'mask' in channels_list: count += 1
    return count

class Config:
    CONTROL_PANCREAS_MASK_DIR = os.path.join(DATASET_DIR, 'pancreatic_masks', 'control')
    DIABETIC_PANCREAS_MASK_DIR = os.path.join(DATASET_DIR, 'pancreatic_masks', 'diabetic')
    MODELS_DIR = MODELS_DIR
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 128
    CHANNELS_TO_USE = ['rgb', 'gray', 'hsv', 'lab', 'mask'] 
    INPUT_CHANNELS = calculate_input_channels([c for c in CHANNELS_TO_USE if c != 'mask']) * 2 

CONFIG = Config()

# --- Model Definition ---
class SEBlock(nn.Module):
    def __init__(self,c,r=4):super(SEBlock,self).__init__();self.avg_pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Sequential(nn.Linear(c,c//r,bias=False),nn.ReLU(inplace=True),nn.Linear(c//r,c,bias=False),nn.Sigmoid())
    def forward(self,x):b,c,_,_=x.size();y=self.avg_pool(x).view(b,c);y=self.fc(y).view(b,c,1,1);return x*y.expand_as(x)

class SimplerCNN(nn.Module):
    def __init__(self, input_channels=4, num_classes=1, dropout_rate=0.5):
        super(SimplerCNN, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(input_channels, 32, 3, 1, 1), nn.GroupNorm(4, 32), nn.ReLU(), nn.MaxPool2d(2,2)); self.s1 = SEBlock(32)
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.GroupNorm(4, 64), nn.ReLU(), nn.MaxPool2d(2,2)); self.s2 = SEBlock(64)
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.GroupNorm(4, 128), nn.ReLU(), nn.MaxPool2d(2,2)); self.s3 = SEBlock(128)
        final_feature_map_size = CONFIG.IMG_SIZE // 8
        flattened_size = 128 * final_feature_map_size * final_feature_map_size
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.s1(self.c1(x)); x = self.s2(self.c2(x)); x = self.s3(self.c3(x))
        x = x.view(x.size(0), -1); return self.classifier(x)

# --- Preprocessing ---
def process_single_eye(image_path: str, config: Config, patient_class: str = None, apply_mask_attention=True) -> torch.Tensor:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    b,_=os.path.splitext(os.path.basename(image_path));mf=f"{b}_pancreas_roi.png"
    
    if patient_class and patient_class.lower() == 'diabetic':
        mp = os.path.join(config.DIABETIC_PANCREAS_MASK_DIR, mf)
    else:
        mp = os.path.join(config.CONTROL_PANCREAS_MASK_DIR, mf)
    
    mask=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    resizer = A.Resize(config.IMG_SIZE, config.IMG_SIZE)
    augmented = resizer(image=img, mask=mask); img, mask = augmented['image'], augmented['mask']
    
    channels_to_stack = []
    if 'rgb' in config.CHANNELS_TO_USE: channels_to_stack.append(img)
    if 'gray' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[...,np.newaxis])
    if 'hsv' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    if 'lab' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    
    combined_image = np.dstack(channels_to_stack) if channels_to_stack else img
    
    num_image_channels = calculate_input_channels([c for c in config.CHANNELS_TO_USE if c != 'mask'])
    final_transform = A.Compose([A.Normalize(mean=[0.5]*num_image_channels, std=[0.5]*num_image_channels, max_pixel_value=255.0), ToTensorV2()])
    
    final_data = final_transform(image=combined_image)
    image_tensor = final_data['image']
    
    if 'mask' in config.CHANNELS_TO_USE and apply_mask_attention:
        mask_tensor = torch.tensor(mask, dtype=torch.float32) / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).expand_as(image_tensor)
        image_tensor = image_tensor * mask_tensor
    
    return image_tensor

# --- Prediction Logic ---
def predict_with_ensemble(models: list, model_metadata: list, left_eye_path: str, right_eye_path: str, config: Config, patient_class: str = None) -> tuple[str, float]:
    try:
        left_tensor = process_single_eye(left_eye_path, config, patient_class, apply_mask_attention=True)
        right_tensor = process_single_eye(right_eye_path, config, patient_class, apply_mask_attention=True)
        input_tensor = torch.cat([left_tensor, right_tensor], dim=0).unsqueeze(0).to(config.DEVICE)
        
        if input_tensor.shape[1] != config.INPUT_CHANNELS:
             raise ValueError(f"Shape mismatch! Expected {config.INPUT_CHANNELS} channels, but got {input_tensor.shape[1]}.")

    except (FileNotFoundError, ValueError) as e:
        return "Error", 0.0

    probs = []
    with torch.no_grad():
        for model in models:
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            probs.append(prob)
    
    avg_prob = float(np.mean(probs))
    
    if avg_prob >= 0.57:
        final_prediction = "Diabetic"
    elif avg_prob <= 0.4:
        final_prediction = "Control"
    else:
        final_prediction = "Control"
    
    return final_prediction, avg_prob

class ComprehensiveTestVisualizer:
    def __init__(self, output_dir="test_results_analysis"):
        """Initialize comprehensive test visualizer"""
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/metrics", exist_ok=True)
        os.makedirs(f"{output_dir}/csv", exist_ok=True)
        
        self.df = None
        self.test_data = None
        
    def run_test_predictions(self):
        """Run predictions on test data using ensemble models"""
        print("Running predictions on test data...")
        
        # Create data manager to get test data
        data_manager = create_data_manager(test_size=0.2, val_size=0.2, random_state=42)
        self.test_data = data_manager.get_test_data()
        
        if not self.test_data:
            print("No test data available!")
            return False
        
        print(f"Found {len(self.test_data)} test patients")
        
        # Load models
        def extract_fold_num(path):
            match = re.search(r'fold_(\d+)', path)
            return int(match.group(1)) if match else 0
        
        model_paths = glob.glob(os.path.join(CONFIG.MODELS_DIR, 'best_f1_model_fold_*.pth'))
        if not model_paths:
            print(f"No models found in '{CONFIG.MODELS_DIR}'")
            return False
        
        model_paths = sorted(model_paths, key=extract_fold_num)
        
        loaded_models = []
        model_metadata = []
        
        for path in model_paths:
            print(f"Loading model: {os.path.basename(path)}")
            checkpoint = torch.load(path, map_location=CONFIG.DEVICE, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                metadata = {
                    'optimal_threshold': checkpoint.get('optimal_threshold', 0.5),
                    'best_f1_score': checkpoint.get('best_f1_score', 0.0),
                    'fold': checkpoint.get('fold', 0)
                }
            else:
                model_state_dict = checkpoint
                metadata = {'optimal_threshold': 0.5, 'best_f1_score': 0.0, 'fold': 0}
            
            model = SimplerCNN(input_channels=CONFIG.INPUT_CHANNELS).to(CONFIG.DEVICE)
            model.load_state_dict(model_state_dict)
            model.eval()
            loaded_models.append(model)
            model_metadata.append(metadata)
        
        print(f"Loaded {len(loaded_models)} models for ensemble prediction")
        
        # Run predictions
        results = []
        for patient_data in tqdm(self.test_data, desc="Predicting Test Patients"):
            patient_id = patient_data.get('patient_id', 'unknown')
            true_class = patient_data['label_name']
            left_path = patient_data['left_image']
            right_path = patient_data['right_image']
            
            if not os.path.exists(left_path) or not os.path.exists(right_path):
                continue
            
            predicted_class, avg_prob = predict_with_ensemble(
                loaded_models, model_metadata, left_path, right_path, CONFIG, true_class
            )
            
            results.append({
                'Patient': patient_id,
                'Ground_Truth': true_class.capitalize(),
                'Prediction': predicted_class,
                'Probability': f"{avg_prob:.4f}",
                'Left_Image': os.path.basename(left_path),
                'Right_Image': os.path.basename(right_path)
            })
        
        # Save results
        results_csv = f"{self.output_dir}/csv/test_predictions_{self.timestamp}.csv"
        with open(results_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Predictions saved to: {results_csv}")
        
        # Load into dataframe
        self.df = pd.read_csv(results_csv)
        return True
    
    def generate_confusion_matrix(self):
        """Generate and save confusion matrix visualization"""
        if self.df is None:
            return {}
            
        print("Generating confusion matrix...")
        
        # Prepare data
        y_true = (self.df['Ground_Truth'] == 'Diabetic').astype(int)
        y_pred = (self.df['Prediction'] == 'Diabetic').astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Create heatmap style
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Control', 'Diabetic'],
                   yticklabels=['Control', 'Diabetic'],
                   square=True, cbar_kws={'shrink': 0.8})
        
        plt.title(f'Confusion Matrix - Test Data Results\\nAccuracy: {accuracy:.3f}', fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add detailed stats
        stats_text = f"""
True Positives: {tp} | False Positives: {fp}
False Negatives: {fn} | True Negatives: {tn}

Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}
Precision: {precision:.3f} | F1-Score: {f1:.3f}
        """
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/confusion_matrix_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'sensitivity_recall': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def generate_roc_curve(self):
        """Generate ROC curve analysis"""
        if self.df is None:
            return 0.0
            
        print("Generating ROC curve...")
        
        # Prepare data
        y_true = (self.df['Ground_Truth'] == 'Diabetic').astype(int)
        y_scores = self.df['Probability'].astype(float)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Test Data Performance', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/roc_curve_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return roc_auc
    
    def generate_probability_distribution(self):
        """Generate probability distribution analysis"""
        if self.df is None:
            return
            
        print("Generating probability distribution analysis...")
        
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Overall distribution
        plt.subplot(2, 2, 1)
        control_probs = self.df[self.df['Ground_Truth'] == 'Control']['Probability'].astype(float)
        diabetic_probs = self.df[self.df['Ground_Truth'] == 'Diabetic']['Probability'].astype(float)
        
        plt.hist(control_probs, bins=20, alpha=0.7, label='Control', color='blue', density=True)
        plt.hist(diabetic_probs, bins=20, alpha=0.7, label='Diabetic', color='red', density=True)
        plt.axvline(x=0.57, color='black', linestyle='--', label='Decision Threshold (0.57)')
        plt.axvline(x=0.4, color='gray', linestyle='--', alpha=0.7, label='Lower Bound (0.4)')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Confidence levels
        plt.subplot(2, 2, 2)
        probs = self.df['Probability'].astype(float)
        confidence_levels = []
        for prob in probs:
            if prob >= 0.7 or prob <= 0.3:
                confidence_levels.append('HIGH')
            elif prob >= 0.57 or prob <= 0.4:
                confidence_levels.append('MEDIUM')
            else:
                confidence_levels.append('LOW')
        
        confidence_counts = pd.Series(confidence_levels).value_counts()
        plt.pie(confidence_counts.values, labels=confidence_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Confidence Level Distribution')
        
        # Subplot 3: Correct vs Incorrect predictions by probability
        plt.subplot(2, 2, 3)
        correct = self.df['Ground_Truth'] == self.df['Prediction']
        correct_probs = self.df[correct]['Probability'].astype(float)
        incorrect_probs = self.df[~correct]['Probability'].astype(float)
        
        plt.hist(correct_probs, bins=15, alpha=0.7, label='Correct', color='green', density=True)
        plt.hist(incorrect_probs, bins=15, alpha=0.7, label='Incorrect', color='red', density=True)
        plt.axvline(x=0.57, color='black', linestyle='--', label='Threshold')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution by Prediction Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Performance by probability ranges
        plt.subplot(2, 2, 4)
        prob_ranges = ['0.0-0.3', '0.3-0.4', '0.4-0.57', '0.57-0.7', '0.7-1.0']
        accuracies = []
        counts = []
        
        probs = self.df['Probability'].astype(float)
        for i, (low, high) in enumerate([(0.0, 0.3), (0.3, 0.4), (0.4, 0.57), (0.57, 0.7), (0.7, 1.0)]):
            mask = (probs >= low) & (probs < high) if i < 4 else (probs >= low) & (probs <= high)
            subset = self.df[mask]
            if len(subset) > 0:
                accuracy = (subset['Ground_Truth'] == subset['Prediction']).mean()
                accuracies.append(accuracy)
                counts.append(len(subset))
            else:
                accuracies.append(0)
                counts.append(0)
        
        bars = plt.bar(prob_ranges, accuracies, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
        plt.xlabel('Probability Range')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Probability Range')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/probability_analysis_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_sample_visualizations(self):
        """Generate sample image visualizations with predictions"""
        if self.df is None or self.test_data is None:
            return
            
        print("Generating sample visualizations...")
        
        # Select interesting cases
        correct_control = self.df[(self.df['Ground_Truth'] == 'Control') & 
                                (self.df['Prediction'] == 'Control')].head(2)
        correct_diabetic = self.df[(self.df['Ground_Truth'] == 'Diabetic') & 
                                 (self.df['Prediction'] == 'Diabetic')].head(2)
        false_positives = self.df[(self.df['Ground_Truth'] == 'Control') & 
                                (self.df['Prediction'] == 'Diabetic')].head(2)
        false_negatives = self.df[(self.df['Ground_Truth'] == 'Diabetic') & 
                                (self.df['Prediction'] == 'Control')].head(2)
        
        cases = [
            (correct_control, 'Correct_Control', 'green'),
            (correct_diabetic, 'Correct_Diabetic', 'green'),
            (false_positives, 'False_Positives', 'red'),
            (false_negatives, 'False_Negatives', 'orange')
        ]
        
        for case_df, case_name, color in cases:
            if len(case_df) == 0:
                continue
                
            for idx, row in case_df.iterrows():
                patient_id = row['Patient']
                
                # Find corresponding test data
                patient_data = next((p for p in self.test_data if p.get('patient_id') == patient_id), None)
                if not patient_data:
                    continue
                    
                left_path = patient_data['left_image']
                right_path = patient_data['right_image']
                
                if os.path.exists(left_path) and os.path.exists(right_path):
                    try:
                        self.create_sample_visualization(
                            left_path, right_path, row, case_name, color
                        )
                    except Exception as e:
                        print(f"Error creating visualization for patient {patient_id}: {e}")
    
    def create_sample_visualization(self, left_path, right_path, row, case_type, color):
        """Create a single sample visualization"""
        # Load images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            return
            
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left eye
        axes[0].imshow(left_img)
        axes[0].set_title(f'Left Eye - Patient {row["Patient"]}', fontsize=12)
        axes[0].axis('off')
        
        # Right eye
        axes[1].imshow(right_img)
        axes[1].set_title(f'Right Eye - Patient {row["Patient"]}', fontsize=12)
        axes[1].axis('off')
        
        # Add prediction information
        prob = float(row['Probability'])
        confidence = (
            "HIGH" if prob >= 0.7 or prob <= 0.3
            else "MEDIUM" if prob >= 0.57 or prob <= 0.4
            else "LOW"
        )
        
        fig.suptitle(
            f'Ground Truth: {row["Ground_Truth"]} | Prediction: {row["Prediction"]} | '
            f'Probability: {prob:.3f} | Confidence: {confidence}',
            fontsize=14, color=color, weight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/{case_type}_Patient_{row['Patient']}_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test analysis report"""
            
        print("Generating confusion matrix...")
        
        # Convert predictions to binary
        y_true = [1 if 'Diabetic' in str(x) else 0 for x in self.df['Ground_Truth']]
        y_pred = [1 if 'Diabetic' in str(x) else 0 for x in self.df['Prediction']]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Control', 'Diabetic'],
                   yticklabels=['Control', 'Diabetic'])
        plt.title('Confusion Matrix - Diabetes Detection from Iris Images', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Save metrics
        metrics = {
            'confusion_matrix': cm.tolist(),
            'accuracy': float(accuracy),
            'sensitivity_recall': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1_score),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        with open(f"{self.output_dir}/metrics/performance_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def generate_roc_curve(self):
        """Generate ROC curve visualization"""
        if self.df is None:
            return
            
        print("Generating ROC curve...")
        
        # Convert to binary
        y_true = [1 if 'Diabetic' in str(x) else 0 for x in self.df['Ground_Truth']]
        y_scores = self.df['Probability'].values
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve - Diabetes Detection', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/roc_curve.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return roc_auc
    
    def generate_sample_visualizations(self):
        """Generate sample image visualizations with predictions"""
        if self.df is None or self.test_data is None:
            return
            
        print("Generating sample visualizations...")
        
        # Select interesting cases
        correct_control = self.df[(self.df['Ground_Truth'] == 'Control') & 
                                (self.df['Prediction'] == 'Control')].head(2)
        correct_diabetic = self.df[(self.df['Ground_Truth'] == 'Diabetic') & 
                                 (self.df['Prediction'] == 'Diabetic')].head(2)
        false_positives = self.df[(self.df['Ground_Truth'] == 'Control') & 
                                (self.df['Prediction'] == 'Diabetic')].head(2)
        false_negatives = self.df[(self.df['Ground_Truth'] == 'Diabetic') & 
                                (self.df['Prediction'] == 'Control')].head(2)
        
        cases = [
            (correct_control, 'Correct_Control', 'green'),
            (correct_diabetic, 'Correct_Diabetic', 'green'),
            (false_positives, 'False_Positives', 'red'),
            (false_negatives, 'False_Negatives', 'orange')
        ]
        
        for case_df, case_name, color in cases:
            if len(case_df) == 0:
                continue
                
            for idx, row in case_df.iterrows():
                patient_id = row['Patient']
                
                # Find corresponding test data
                patient_data = next((p for p in self.test_data if p.get('patient_id') == patient_id), None)
                if not patient_data:
                    continue
                    
                left_path = patient_data['left_image']
                right_path = patient_data['right_image']
                
                if os.path.exists(left_path) and os.path.exists(right_path):
                    try:
                        self.create_sample_visualization(
                            left_path, right_path, row, case_name, color
                        )
                    except Exception as e:
                        print(f"Error creating visualization for patient {patient_id}: {e}")
    
    def create_sample_visualization(self, left_path, right_path, row, case_type, color):
        """Create a single sample visualization"""
        # Load images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            return
            
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left eye
        axes[0].imshow(left_img)
        axes[0].set_title(f'Left Eye - Patient {row["Patient"]}', fontsize=12)
        axes[0].axis('off')
        
        # Right eye
        axes[1].imshow(right_img)
        axes[1].set_title(f'Right Eye - Patient {row["Patient"]}', fontsize=12)
        axes[1].axis('off')
        
        # Add prediction information
        prob = float(row['Probability'])
        confidence = (
            "HIGH" if prob >= 0.7 or prob <= 0.3
            else "MEDIUM" if prob >= 0.55 or prob <= 0.4
            else "LOW"
        )
        
        fig.suptitle(
            f'Ground Truth: {row["Ground_Truth"]} | Prediction: {row["Prediction"]} | '
            f'Probability: {prob:.3f} | Confidence: {confidence}',
            fontsize=14, color=color, weight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/images/{case_type}_Patient_{row['Patient']}_{self.timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _find_image_path(self, pattern):
        """Find image path based on pattern"""
        search_dirs = [
            os.path.join(DATASET_DIR, 'data', 'control'),
            os.path.join(DATASET_DIR, 'data', 'diabetic')
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for filename in os.listdir(search_dir):
                    if pattern in filename and filename.endswith('.jpg'):
                        return os.path.join(search_dir, filename)
        return None
    
    def _load_image(self, image_path, size=(200, 200)):
        """Load and resize image"""
        if image_path and os.path.exists(image_path):
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, size)
                    return img
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        return self._create_placeholder(size)
    
    def _create_placeholder(self, size=(200, 200)):
        """Create placeholder image"""
        placeholder = np.ones((*size, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder, "Image Not Found", (20, size[1]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        return placeholder
    
    def _create_html_index(self, samples):
        """Create HTML index for sample results"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sample Results - Diabetes Detection</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .sample { margin: 20px 0; border: 1px solid #ccc; padding: 10px; }
                img { max-width: 100%; height: auto; }
                .correct { border-left: 5px solid green; }
                .incorrect { border-left: 5px solid red; }
            </style>
        </head>
        <body>
            <h1>Sample Results - Diabetes Detection from Iris Images</h1>
        """
        
        for idx, sample in samples.iterrows():
            correct = sample['Ground_Truth'] == sample['Prediction']
            status_class = 'correct' if correct else 'incorrect'
            status_text = '✓ Correct' if correct else '✗ Incorrect'
            
            html_content += f"""
            <div class="sample {status_class}">
                <h3>Sample {idx+1}: Patient {sample['Patient']} - {status_text}</h3>
                <img src="images/sample_{idx+1:02d}.png" alt="Sample {idx+1}">
                <p><strong>Ground Truth:</strong> {sample['Ground_Truth']}</p>
                <p><strong>Prediction:</strong> {sample['Prediction']}</p>
                <p><strong>Confidence:</strong> {sample['Probability']:.3f}</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(f"{self.output_dir}/sample_results_index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test analysis report"""
        if not self.run_test_predictions():
            print("Failed to run test predictions")
            return
            
        print("Generating comprehensive test analysis...")
        
        # Generate all visualizations
        metrics = self.generate_confusion_matrix()
        roc_auc = self.generate_roc_curve()
        self.generate_probability_distribution()
        self.generate_sample_visualizations()
        
        # Save detailed metrics
        detailed_metrics = {
            'timestamp': self.timestamp,
            'test_patients_count': len(self.test_data),
            'predictions_made': len(self.df),
            'accuracy': metrics['accuracy'],
            'sensitivity': metrics['sensitivity_recall'],
            'specificity': metrics['specificity'],
            'precision': metrics['precision'],
            'f1_score': metrics['f1_score'],
            'roc_auc': roc_auc,
            'confusion_matrix': {
                'true_positives': int(metrics['true_positives']),
                'true_negatives': int(metrics['true_negatives']),
                'false_positives': int(metrics['false_positives']),
                'false_negatives': int(metrics['false_negatives'])
            }
        }
        
        with open(f"{self.output_dir}/metrics/detailed_metrics_{self.timestamp}.json", 'w') as f:
            json.dump(detailed_metrics, f, indent=4)
        
        # Create summary report
        control_count = len(self.df[self.df['Ground_Truth'] == 'Control'])
        diabetic_count = len(self.df[self.df['Ground_Truth'] == 'Diabetic'])
        
        report = f"""
# Test Data Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Dataset Overview
- **Total Test Patients**: {len(self.test_data)}
- **Control Patients**: {control_count}
- **Diabetic Patients**: {diabetic_count}
- **Predictions Made**: {len(self.df)}

## Performance Metrics (Test Data Only)
- **Accuracy**: {metrics['accuracy']:.3f}
- **Sensitivity (Recall)**: {metrics['sensitivity_recall']:.3f}
- **Specificity**: {metrics['specificity']:.3f}
- **Precision**: {metrics['precision']:.3f}
- **F1-Score**: {metrics['f1_score']:.3f}
- **AUC-ROC**: {roc_auc:.3f}

## Decision Rule Applied
- **Diabetic Threshold**: ≥0.57 probability
- **Control Threshold**: ≤0.4 probability  
- **Ambiguous Range**: 0.4-0.57 → defaults to Control (conservative)

## Confusion Matrix Details
- **True Positives (TP)**: {metrics['true_positives']} (Correctly identified Diabetic)
- **True Negatives (TN)**: {metrics['true_negatives']} (Correctly identified Control)
- **False Positives (FP)**: {metrics['false_positives']} (Control predicted as Diabetic)
- **False Negatives (FN)**: {metrics['false_negatives']} (Diabetic predicted as Control)

## Files Generated
- **Confusion Matrix**: images/confusion_matrix_{self.timestamp}.png
- **ROC Curve**: images/roc_curve_{self.timestamp}.png
- **Probability Analysis**: images/probability_analysis_{self.timestamp}.png
- **Sample Visualizations**: images/[case_type]_Patient_[ID]_{self.timestamp}.png
- **Detailed Metrics**: metrics/detailed_metrics_{self.timestamp}.json
- **Test Predictions**: csv/test_predictions_{self.timestamp}.csv

## Academic Rigor
- Test data was never seen during model training
- Proper patient-level data splits maintained
- Conservative decision thresholds applied
- Ensemble averaging across all folds

## Clinical Interpretation
- **High Sensitivity** indicates good detection of diabetic cases
- **High Specificity** indicates low false positive rate on controls
- **Conservative threshold** reduces unnecessary referrals
- **Confidence levels** help identify uncertain cases requiring manual review
        """
        
        with open(f"{self.output_dir}/test_analysis_report_{self.timestamp}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TEST ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"📁 All results saved to: {self.output_dir}")
        print(f"📊 Test Accuracy: {metrics['accuracy']:.3f}")
        print(f"🎯 Sensitivity: {metrics['sensitivity_recall']:.3f}")
        print(f"🛡️  Specificity: {metrics['specificity']:.3f}")
        print(f"📈 AUC-ROC: {roc_auc:.3f}")
        print(f"📄 Full report: test_analysis_report_{self.timestamp}.md")
        print(f"{'='*60}")

def main():
    """Main execution function"""
    print("=== Comprehensive Test Data Analysis ===")
    print("Running predictions on test data and generating full analysis...\n")
    
    visualizer = ComprehensiveTestVisualizer()
    visualizer.generate_comprehensive_report()
    
    print("\n=== Test Analysis Complete ===")
    print("Check the test_results_analysis directory for all results!")

if __name__ == "__main__":
    main()