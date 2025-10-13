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
    # --- Paths for prediction and evaluation ---
    # *** NEW: Configure paths for both classes to enable evaluation ***
    # *** YOU MUST VERIFY THESE PATHS ARE CORRECT FOR YOUR SYSTEM ***
    TEST_DATA_CONFIG = {
        # The key is the 'ground truth' label
        "Control": {
            "images": "dataset/control/",
            "masks": "dataset/pancreas_masks_for_training/control/"
        },
        "Diabetic": {
            "images": "dataset/diabetic/",
            "masks": "dataset/pancreas_masks_for_training/diabetic/"
        }
    }
    MODELS_DIR = '.' 
    
    # --- Critical settings that MUST match the training config ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 128
    CHANNELS_TO_USE = ['rgb', 'gray', 'hsv', 'lab', 'mask'] 
    
    # These are calculated automatically
    INPUT_CHANNELS_PER_EYE = calculate_input_channels(CHANNELS_TO_USE)
    INPUT_CHANNELS_TOTAL = INPUT_CHANNELS_PER_EYE * 2 

CONFIG = Config()
print(f"Prediction script configured for channels: {CONFIG.CHANNELS_TO_USE} ({CONFIG.INPUT_CHANNELS_TOTAL} total)")


# --- 2. Model Definition (MUST BE IDENTICAL TO THE TRAINING SCRIPT) ---
class SEBlock(nn.Module):
    def __init__(self,c,r=4):super(SEBlock,self).__init__();self.avg_pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Sequential(nn.Linear(c,c//r,bias=False),nn.ReLU(inplace=True),nn.Linear(c//r,c,bias=False),nn.Sigmoid())
    def forward(self,x):b,c,_,_=x.size();y=self.avg_pool(x).view(b,c);y=self.fc(y).view(b,c,1,1);return x*y.expand_as(x)

class SimplerCNN(nn.Module):
    def __init__(self, input_channels=22, num_classes=1, dropout_rate=0.5):
        super(SimplerCNN, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(input_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2)); self.s1 = SEBlock(32)
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2)); self.s2 = SEBlock(64)
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2)); self.s3 = SEBlock(128)
        final_feature_map_size = CONFIG.IMG_SIZE // 8
        flattened_size = 128 * final_feature_map_size * final_feature_map_size
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.s1(self.c1(x)); x = self.s2(self.c2(x)); x = self.s3(self.c3(x))
        x = x.view(x.size(0), -1); return self.classifier(x)


# --- 3. Preprocessing (MUST BE IDENTICAL TO THE TRAINING SCRIPT) ---
# *** MODIFIED: to accept the mask directory as a parameter ***
def process_single_eye(image_path: str, mask_dir: str, config: Config) -> torch.Tensor:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    b,_=os.path.splitext(os.path.basename(image_path));mf=f"{b}_pancreas_roi.png"
    mp=os.path.join(mask_dir, mf)
    mask=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Warning: Mask not found at {mp}. Using a blank mask.")
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    resizer = A.Resize(config.IMG_SIZE, config.IMG_SIZE)
    augmented = resizer(image=img, mask=mask); img, mask = augmented['image'], augmented['mask']
    
    channels_to_stack = []
    if 'rgb' in config.CHANNELS_TO_USE: channels_to_stack.append(img)
    if 'gray' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[...,np.newaxis])
    if 'hsv' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    if 'lab' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    
    combined_image = np.dstack(channels_to_stack) if channels_to_stack else img
    
    # The number of channels for normalization must match the stacked channels (excluding the mask)
    num_norm_channels = len(config.CHANNELS_TO_USE) - 1 if 'mask' in config.CHANNELS_TO_USE else len(config.CHANNELS_TO_USE)
    norm_channels = calculate_input_channels([c for c in config.CHANNELS_TO_USE if c != 'mask'])
    final_transform = A.Compose([A.Normalize(mean=[0.5]*norm_channels, std=[0.5]*norm_channels, max_pixel_value=255.0), ToTensorV2()])
    
    if 'mask' in config.CHANNELS_TO_USE:
        final_data = final_transform(image=combined_image)
        image_tensor = final_data['image']
        mask_tensor = ToTensorV2()(image=mask)['image'].float() / 255.0
        return torch.cat([image_tensor, mask_tensor], dim=0)
    else:
        return final_transform(image=combined_image)['image']


# --- 4. Prediction Logic (Using Ensemble) ---
# *** MODIFIED: to accept the mask directory as a parameter ***
def predict_with_ensemble(models: list, left_eye_path: str, right_eye_path: str, mask_dir: str, config: Config) -> tuple[str, float]:
    try:
        left_tensor = process_single_eye(left_eye_path, mask_dir, config)
        right_tensor = process_single_eye(right_eye_path, mask_dir, config)
        input_tensor = torch.cat([left_tensor, right_tensor], dim=0).unsqueeze(0).to(config.DEVICE)
        
        if input_tensor.shape[1] != config.INPUT_CHANNELS_TOTAL:
             raise ValueError(f"Shape mismatch! Expected {config.INPUT_CHANNELS_TOTAL} channels, but got {input_tensor.shape[1]}. Check your CHANNELS_TO_USE config.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Skipping pair due to error: {e}")
        return "Error", 0.0

    probabilities = []
    with torch.no_grad():
        for model in models:
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            probabilities.append(prob)
    
    avg_prob = np.mean(probabilities)
    # The model predicts 'Diabetic' if probability is high
    label = "Diabetic" if avg_prob > 0.5 else "Control"
    return label, avg_prob

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
    
    model_paths = glob.glob(os.path.join(CONFIG.MODELS_DIR, 'best_f1_model_fold_*.pth'))
    if not model_paths:
        print(f"FATAL ERROR: No models found in '{CONFIG.MODELS_DIR}' matching the pattern 'best_f1_model_fold_*.pth'.")
        exit(1)

    loaded_models = []
    for path in model_paths:
        print(f"Loading model: {path}")
        model = SimplerCNN(input_channels=CONFIG.INPUT_CHANNELS_TOTAL).to(CONFIG.DEVICE)
        model.load_state_dict(torch.load(path, map_location=CONFIG.DEVICE))
        model.eval()
        loaded_models.append(model)
    print(f"\n--- Successfully loaded {len(loaded_models)} models for ensemble prediction ---\n")

    all_results = []
    prediction_counts = defaultdict(int)

    # --- Iterate through each configured dataset (e.g., 'Control', 'Diabetic') ---
    for ground_truth_label, paths in CONFIG.TEST_DATA_CONFIG.items():
        image_dir = paths['images']
        mask_dir = paths['masks']
        print(f"\n--- Processing dataset for class: '{ground_truth_label}' from '{image_dir}' ---")

        if not os.path.isdir(image_dir):
            print(f"Warning: Directory not found, skipping: {image_dir}")
            continue

        patient_files = defaultdict(lambda: {'L': None, 'R': None})
        for image_path in glob.glob(os.path.join(image_dir, '*.jpg')):
            filename = os.path.basename(image_path)
            match = re.search(r'(L|R)', filename, re.IGNORECASE)
            if match:
                patient_files[filename[:match.start()]][match.group(0).upper()] = image_path
        
        patients_to_predict = []
        for pid, eyes in patient_files.items():
            if eyes['L'] and eyes['R']:
                patients_to_predict.append((pid, eyes['L'], eyes['R']))
            else:
                print(f"Warning: Patient '{pid}' in '{ground_truth_label}' set is missing a complete L/R pair. Skipping.")
        
        print(f"Found {len(patients_to_predict)} complete patient pairs to process for this class.")
        
        if patients_to_predict:
            for pid, left_path, right_path in tqdm(patients_to_predict, desc=f"Predicting '{ground_truth_label}'"):
                predicted_class, avg_prob = predict_with_ensemble(loaded_models, left_path, right_path, mask_dir, CONFIG)
                
                prediction_counts[predicted_class] += 1
                
                all_results.append({
                    'patient_id': pid,
                    'left_image': os.path.basename(left_path),
                    'right_image': os.path.basename(right_path),
                    'ground_truth': ground_truth_label, # Add ground truth
                    'prediction': predicted_class,
                    'avg_probability': f"{avg_prob:.4f}"
                })

    if all_results:
        # Save results to CSV
        with open('evaluation_results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print("\n--- Predictions complete. Results saved to 'evaluation_results.csv' ---")
        
        # Print detailed per-patient results
        print("\n" + "="*80); print("                               DETAILED PREDICTIONS"); print("="*80)
        for res in all_results:
            status = "CORRECT" if res['prediction'] == res['ground_truth'] else "WRONG"
            print(f"Patient: {res['patient_id']:<20} | Truth: {res['ground_truth']:<10} | Prediction: {res['prediction']:<10} (Prob: {res['avg_probability']}) | {status}")
        print("="*80)

        # Calculate and print the final metrics
        calculate_and_print_metrics(all_results)
    else:
        print("\nNo patient pairs were processed. Cannot calculate metrics. Check TEST_DATA_CONFIG paths.")