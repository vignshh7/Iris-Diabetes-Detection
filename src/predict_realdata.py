
import os
import sys
import torch
import cv2
import numpy as np
import csv
import glob
from tqdm import tqdm

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

from src.cnnpredict import SimplerCNN, calculate_input_channels, Config
from src.generate_masks import generate_iris_mask, create_pancreas_mask_from_iris_mask, get_inference_transforms, load_iris_model


REALDATA_DIR = os.path.join(os.getcwd(), 'realdata')
IMAGES_DIR = os.path.join(REALDATA_DIR, 'images')
MASKS_DIR = os.path.join(REALDATA_DIR, 'masks')
PANCREATIC_MASKS_DIR = os.path.join(REALDATA_DIR, 'pancreatic_masks')
CSV_OUTPUT = 'realdata_predictions.csv'
def generate_masks_for_realdata():
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(PANCREATIC_MASKS_DIR, exist_ok=True)
    iris_model_path = os.path.join(MODELS_DIR, 'best_iris_model_3class.pth')
    if not os.path.exists(iris_model_path):
        iris_model_path = os.path.join(MODELS_DIR, 'best_iris_model_2class.pth')
    if not os.path.exists(iris_model_path):
        print("[ERROR] No iris segmentation model found!")
        return
    model = load_iris_model(iris_model_path)
    transforms = get_inference_transforms()
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]
    for f in tqdm(files, desc='Generating masks for realdata'):
        img_path = os.path.join(IMAGES_DIR, f)
        base_name = os.path.splitext(f)[0]
        match = re.match(r"(\d+)([LR])\.", f, re.IGNORECASE)
        if not match:
            continue
        eye_side = 'left' if match.group(2).upper() == 'L' else 'right'
        iris_mask = generate_iris_mask(model, img_path, transforms)
        if iris_mask is not None:
            iris_mask_path = os.path.join(MASKS_DIR, f"{base_name}_mask.png")
            cv2.imwrite(iris_mask_path, iris_mask)
            pancreas_mask = create_pancreas_mask_from_iris_mask(iris_mask, eye_side)
            if pancreas_mask is not None:
                pancreas_mask_path = os.path.join(PANCREATIC_MASKS_DIR, f"{base_name}_pancreas_roi.png")
                cv2.imwrite(pancreas_mask_path, pancreas_mask)

import re
def find_image_pairs(directory):
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]
    patient_dict = {}
    for f in files:
        match = re.match(r"(\d+)([LR])\.", f, re.IGNORECASE)
        if match:
            patient_num = match.group(1)
            eye = match.group(2).upper()
            if patient_num not in patient_dict:
                patient_dict[patient_num] = {'L': [], 'R': []}
            patient_dict[patient_num][eye].append(f)
    pairs = []
    for patient_num, eyes in patient_dict.items():
        # Pair by closest timestamp if multiple exist, else just first
        if eyes['L'] and eyes['R']:
            # If multiple L/R, pair by sorted order (could be improved by timestamp logic)
            lefts = sorted(eyes['L'])
            rights = sorted(eyes['R'])
            for l, r in zip(lefts, rights):
                pairs.append((l, r))
    return pairs

def load_ensemble_models():
    model_paths = glob.glob(os.path.join(MODELS_DIR, 'best_f1_model_fold_*.pth'))
    model_paths = sorted(model_paths, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    models = []
    for path in model_paths:
        # Fix: set weights_only=False for compatibility with older checkpoints
        checkpoint = torch.load(path, map_location=Config.DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        model = SimplerCNN(input_channels=Config.INPUT_CHANNELS).to(Config.DEVICE)
        model.load_state_dict(model_state_dict)
        model.eval()
        models.append(model)
    return models

def process_single_eye_realdata(image_path, config, apply_mask_attention=True):
    img_bgr = cv2.imread(os.path.join(IMAGES_DIR, os.path.basename(image_path)))
    if img_bgr is None: raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    b,_=os.path.splitext(os.path.basename(image_path))
    mp = os.path.join(PANCREATIC_MASKS_DIR, f"{b}_pancreas_roi.png")
    mask=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Mask not found at {mp}. Using a blank mask.")
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    mask = cv2.resize(mask, (config.IMG_SIZE, config.IMG_SIZE))
    channels_to_stack = []
    if 'rgb' in config.CHANNELS_TO_USE: channels_to_stack.append(img)
    if 'gray' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[...,np.newaxis])
    if 'hsv' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    if 'lab' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    combined_image = np.dstack(channels_to_stack) if channels_to_stack else img
    num_image_channels = calculate_input_channels([c for c in config.CHANNELS_TO_USE if c != 'mask'])
    from albumentations.pytorch import ToTensorV2
    import albumentations as A
    final_transform = A.Compose([A.Normalize(mean=[0.5]*num_image_channels, std=[0.5]*num_image_channels, max_pixel_value=255.0), ToTensorV2()])
    final_data = final_transform(image=combined_image)
    image_tensor = final_data['image']
    if 'mask' in config.CHANNELS_TO_USE and apply_mask_attention:
        mask_tensor = torch.tensor(mask, dtype=torch.float32) / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).expand_as(image_tensor)
        image_tensor = image_tensor * mask_tensor
    return image_tensor

def predict_with_ensemble(models, left_eye_path, right_eye_path, config):
    try:
        left_tensor = process_single_eye_realdata(left_eye_path, config, apply_mask_attention=True)
        right_tensor = process_single_eye_realdata(right_eye_path, config, apply_mask_attention=True)
        input_tensor = torch.cat([left_tensor, right_tensor], dim=0).unsqueeze(0).to(config.DEVICE)
        if input_tensor.shape[1] != config.INPUT_CHANNELS:
            raise ValueError(f"Shape mismatch! Expected {config.INPUT_CHANNELS} channels, but got {input_tensor.shape[1]}")
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
    if avg_prob >= 0.55:
        final_prediction = "Diabetic"
    elif avg_prob <= 0.4:
        final_prediction = "Control"
    else:
        final_prediction = "Control"
    return final_prediction, avg_prob


def predict_realdata():
    print("[INFO] Generating masks for all images in realdata/images...")
    generate_masks_for_realdata()
    print("[INFO] Loading ensemble models...")
    models = load_ensemble_models()
    print(f"[INFO] Loaded {len(models)} models for ensemble prediction.")
    print(f"[INFO] Searching for image pairs in: {IMAGES_DIR}")
    pairs = find_image_pairs(IMAGES_DIR)
    print(f"[INFO] Found {len(pairs)} image pairs for prediction.")
    results = []
    for idx, (left_img, right_img) in enumerate(pairs, 1):
        left_path = os.path.join(IMAGES_DIR, left_img)
        right_path = os.path.join(IMAGES_DIR, right_img)
        print(f"[PAIR {idx}/{len(pairs)}] Predicting: {left_img} & {right_img}")
        pred, prob = predict_with_ensemble(models, left_path, right_path, Config)
        confidence = (
            "HIGH" if prob >= 0.7 or prob <= 0.3
            else "MEDIUM" if prob >= 0.6 or prob <= 0.4
            else "LOW"
        )
        print(f"    -> Prediction: {pred} | Probability: {prob:.4f} | Confidence: {confidence}")
        results.append({
            'Left_Image': left_img,
            'Right_Image': right_img,
            'Prediction': pred,
            'Probability': f"{prob:.4f}",
            'Confidence': confidence
        })
    if results:
        with open(CSV_OUTPUT, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[INFO] Prediction results saved to {CSV_OUTPUT}")
    else:
        print("[WARN] No image pairs were processed.")

if __name__ == '__main__':
    predict_realdata()
