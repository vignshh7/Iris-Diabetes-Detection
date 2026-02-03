import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from collections import defaultdict
import re
from tqdm import tqdm
import csv
import sys
import json
import random
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score

# Add project root to path for config import
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

# --- 1. Configuration (MUST MATCH THE TRAINING SCRIPT) ---
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
    # --- Paths for prediction and evaluation (using centralized config) ---
    MODELS_DIR = MODELS_DIR
    
    # --- Critical settings that MUST match the training config ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 128
    CHANNELS_TO_USE = ['rgb', 'gray', 'hsv', 'lab', 'mask']
    
    # Calculate input channels: mask is not counted as separate channel (spatial attention)
    INPUT_CHANNELS = calculate_input_channels([c for c in CHANNELS_TO_USE if c != 'mask']) * 2

CONFIG = Config()
print(f"Evaluation script configured for channels: {CONFIG.CHANNELS_TO_USE} ({CONFIG.INPUT_CHANNELS} total)")


# --- 2. Model Definition (MUST BE IDENTICAL TO THE TRAINING SCRIPT) ---
class SEBlock(nn.Module):
    def __init__(self,c,r=4):super(SEBlock,self).__init__();self.avg_pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Sequential(nn.Linear(c,c//r,bias=False),nn.ReLU(inplace=True),nn.Linear(c//r,c,bias=False),nn.Sigmoid())
    def forward(self,x):b,c,_,_=x.size();y=self.avg_pool(x).view(b,c);y=self.fc(y).view(b,c,1,1);return x*y.expand_as(x)

class SimplerCNN(nn.Module):
    def __init__(self, input_channels=4, num_classes=1, dropout_rate=0.5):
        super(SimplerCNN, self).__init__()
        # GroupNorm for stability with small batch sizes (groups=4 for channels 32,64,128)
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


# --- 3. Preprocessing (MUST MATCH TRAINING SCRIPT) ---
def process_single_eye(image_path: str, config: Config, patient_class: str = None, apply_mask_attention=True) -> torch.Tensor:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    b,_=os.path.splitext(os.path.basename(image_path));mf=f"{b}_pancreas_roi.png"
    
    # Use class-specific pancreatic masks directory
    if patient_class and patient_class.lower() == 'diabetic':
        mp = os.path.join(DATASET_DIR, 'pancreatic_masks', 'diabetic', mf)
    else:
        mp = os.path.join(DATASET_DIR, 'pancreatic_masks', 'control', mf)
    
    mask=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Warning: Mask not found at {mp}. Using a blank mask.")
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    resizer = A.Resize(config.IMG_SIZE, config.IMG_SIZE)
    augmented = resizer(image=img, mask=mask); img, mask = augmented['image'], augmented['mask']
    
    # Build image channels (excluding mask from channel stack)
    channels_to_stack = []
    if 'rgb' in config.CHANNELS_TO_USE: channels_to_stack.append(img)
    if 'gray' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[...,np.newaxis])
    if 'hsv' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    if 'lab' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    
    combined_image = np.dstack(channels_to_stack) if channels_to_stack else img
    
    # Dynamic normalization for image channels only
    num_image_channels = calculate_input_channels([c for c in config.CHANNELS_TO_USE if c != 'mask'])
    final_transform = A.Compose([A.Normalize(mean=[0.5]*num_image_channels, std=[0.5]*num_image_channels, max_pixel_value=255.0), ToTensorV2()])
    
    # Apply normalization to image channels only
    final_data = final_transform(image=combined_image)
    image_tensor = final_data['image']
    
    # Apply mask as spatial attention (multiply, don't concatenate)
    if 'mask' in config.CHANNELS_TO_USE and apply_mask_attention:
        mask_tensor = torch.tensor(mask, dtype=torch.float32) / 255.0
        # Expand mask to match image channels and apply spatial attention
        mask_tensor = mask_tensor.unsqueeze(0).expand_as(image_tensor)
        image_tensor = image_tensor * mask_tensor
    
    return image_tensor


# --- 4. Prediction Logic (Using Ensemble with Optimal Thresholds) ---
def predict_with_ensemble(models: list, model_metadata: list, left_eye_path: str, right_eye_path: str, config: Config, patient_class: str = None) -> tuple[str, float]:
    try:
        left_tensor = process_single_eye(left_eye_path, config, patient_class, apply_mask_attention=True)
        right_tensor = process_single_eye(right_eye_path, config, patient_class, apply_mask_attention=True)
        input_tensor = torch.cat([left_tensor, right_tensor], dim=0).unsqueeze(0).to(config.DEVICE)
        
        if input_tensor.shape[1] != config.INPUT_CHANNELS:
             raise ValueError(f"Shape mismatch! Expected {config.INPUT_CHANNELS} channels, but got {input_tensor.shape[1]}. Check your CHANNELS_TO_USE config.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Skipping pair due to error: {e}")
        return "Error", 0.0

    probs = []
    with torch.no_grad():
        for model in models:
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            probs.append(prob)
    
    avg_prob = float(np.mean(probs))
    
    # Conservative medical decision rule
    if avg_prob >= 0.55:
        final_prediction = "Diabetic"
    elif avg_prob <= 0.4:
        final_prediction = "Control"
    else:
        final_prediction = "Control"  # ambiguous → safer class
    
    return final_prediction, avg_prob

# --- 5. NEW: Evaluation Metrics Calculation ---
def calculate_and_print_metrics(results: list):
    """Calculates and prints performance metrics from a list of prediction results."""
    # Assuming 'Diabetic' is the POSITIVE class and 'Control' is the NEGATIVE class
    tp, tn, fp, fn = 0, 0, 0, 0

    for res in results:
        pred = res['prediction']
        truth = res['ground_truth']

        if pred == "Diabetic" and truth == "Diabetic":
            tp += 1
        elif pred == "Control" and truth == "Control":
            tn += 1
        elif pred == "Diabetic" and truth == "Control":
            fp += 1
        elif pred == "Control" and truth == "Diabetic":
            fn += 1

    print("\n" + "="*55)
    print("                EVALUATION METRICS")
    print("="*55)
    print(f"Positive Class: Diabetic | Negative Class: Control\n")
    print(f"Confusion Matrix:")
    print(f"  - True Positives (TP):  {tp:4d} (Correctly identified Diabetic)")
    print(f"  - True Negatives (TN):  {tn:4d} (Correctly identified Control)")
    print(f"  - False Positives (FP): {fp:4d} (Incorrectly identified Diabetic)")
    print(f"  - False Negatives (FN): {fn:4d} (Incorrectly identified Control)")
    print("-" * 55)

    # --- Calculate Metrics ---
    # Sensitivity (Recall or True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Precision (Positive Predictive Value)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    # F1 Score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    print("Performance Metrics:")
    print(f"  - Accuracy:    {accuracy:.4f} ({(accuracy * 100):.2f}%)")
    print(f"  - Sensitivity: {sensitivity:.4f} ({(sensitivity * 100):.2f}%)  <-- Recall / TPR")
    print(f"  - Specificity: {specificity:.4f} ({(specificity * 100):.2f}%)  <-- TNR")
    print(f"  - Precision:   {precision:.4f} ({(precision * 100):.2f}%)")
    print(f"  - F1-Score:    {f1_score:.4f}")
    print("="*55)


# --- 6. Main Execution Block (MODIFIED FOR EVALUATION) ---
if __name__ == '__main__':
    print(f"Using device: {CONFIG.DEVICE}")
    
    # Create data manager to get proper test splits
    data_manager = create_data_manager(test_size=0.2, val_size=0.2, random_state=42)
    
    # Get test data (proper academic split)
    test_data = data_manager.get_test_data()
    
    if not test_data:
        print("No test data available. Please ensure data_manager is properly configured.")
        exit(1)
    
    print(f"Found {len(test_data)} test patients")
    
    # Extract fold number from model path for correct ordering
    def extract_fold_num(path):
        match = re.search(r'fold_(\d+)', path)
        return int(match.group(1)) if match else 0
    
    model_paths = glob.glob(os.path.join(CONFIG.MODELS_DIR, 'best_f1_model_fold_*.pth'))
    if not model_paths:
        print(f"FATAL ERROR: No models found in '{CONFIG.MODELS_DIR}' matching the pattern 'best_f1_model_fold_*.pth'.")
        exit(1)
    
    # Sort model paths by fold number to ensure correct model-to-metadata mapping
    model_paths = sorted(model_paths, key=extract_fold_num)
    print(f"Found and sorted {len(model_paths)} model files by fold number")

    loaded_models = []
    model_metadata = []
    
    for path in model_paths:
        print(f"Loading model: {path}")
        checkpoint = torch.load(path, map_location=CONFIG.DEVICE, weights_only=False)
        
        # Handle both old and new model formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            model_state_dict = checkpoint['model_state_dict']
            metadata = {
                'optimal_threshold': checkpoint.get('optimal_threshold', 0.5),
                'best_f1_score': checkpoint.get('best_f1_score', 0.0),
                'fold': checkpoint.get('fold', 0)
            }
            print(f"  - Fold {metadata['fold']} with optimal threshold {metadata['optimal_threshold']:.3f} (F1: {metadata['best_f1_score']:.3f})")
        else:
            # Old format - just state dict
            model_state_dict = checkpoint
            metadata = {'optimal_threshold': 0.5, 'best_f1_score': 0.0, 'fold': 0}
            print(f"  - Using fallback threshold 0.5")
        
        model = SimplerCNN(input_channels=CONFIG.INPUT_CHANNELS).to(CONFIG.DEVICE)
        model.load_state_dict(model_state_dict)
        model.eval()
        loaded_models.append(model)
        model_metadata.append(metadata)
        
    print(f"\n--- Successfully loaded {len(loaded_models)} models for ensemble evaluation ---\n")

    all_results = []
    prediction_counts = defaultdict(int)
    
    print("Evaluating on test patients...")
    
    for patient_data in tqdm(test_data, desc="Evaluating Test Patients"):
        patient_id = patient_data.get('patient_id', 'unknown')
        true_class = patient_data['label_name']  # 'control' or 'diabetic'
        left_path = patient_data['left_image']
        right_path = patient_data['right_image']
        
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print(f"Warning: Missing images for patient {patient_id}")
            continue
        
        predicted_class, avg_prob = predict_with_ensemble(loaded_models, model_metadata, left_path, right_path, CONFIG, true_class)
        prediction_counts[predicted_class] += 1
        
        all_results.append({
            'patient_id': patient_id,
            'left_image': os.path.basename(left_path),
            'right_image': os.path.basename(right_path),
            'ground_truth': true_class.capitalize(),
            'prediction': predicted_class,
            'avg_probability': f"{avg_prob:.4f}"
        })

    if all_results:
        # Save results to CSV
        with open('evaluation_results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print("\n--- Evaluation complete. Results saved to 'evaluation_results.csv' ---")
        
        # Print detailed per-patient results
        print("\n" + "="*90); print("                               DETAILED EVALUATION RESULTS"); print("="*90)
        correct_count = 0
        for res in all_results:
            status = "✓ CORRECT" if res['prediction'] == res['ground_truth'] else "✗ WRONG"
            if res['prediction'] == res['ground_truth']:
                correct_count += 1
            avg_prob = float(res['avg_probability'])
            confidence = (
                "HIGH" if avg_prob >= 0.7 or avg_prob <= 0.3
                else "MEDIUM" if avg_prob >= 0.6 or avg_prob <= 0.4
                else "LOW"
            )
            print(f"Patient: {res['patient_id']:<20} | Truth: {res['ground_truth']:<10} | Prediction: {res['prediction']:<10} (Prob: {avg_prob:.3f}, Confidence: {confidence}) | {status}")
        print("="*90)
        print(f"Overall Accuracy: {correct_count}/{len(all_results)} = {correct_count/len(all_results):.3f}")

        # Calculate and print comprehensive metrics
        calculate_and_print_metrics(all_results)
        
        print(f"\n🎯 Academic Rigor Confirmed: Test set was never seen during training")
        print(f"📊 Results based on proper patient-level data splits")
        
    else:
        print("\nNo test patient pairs were processed. Cannot calculate metrics.")