#!/usr/bin/env python3
"""
Unified Mask Generation Script
Combines iris segmentation and pancreatic ROI generation using trained models.
Generates both iris masks and pancreatic masks for all images in dataset.
"""

import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import glob
import sys

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class Config:
    # Model paths
    IRIS_2CLASS_MODEL = os.path.join(MODELS_DIR, 'best_iris_model_2class.pth')
    IRIS_3CLASS_MODEL = os.path.join(MODELS_DIR, 'best_iris_model_3class.pth')
    
    # New dataset structure paths
    CONTROL_IMAGES_DIR = os.path.join(DATASET_DIR, 'data', 'control')
    DIABETIC_IMAGES_DIR = os.path.join(DATASET_DIR, 'data', 'diabetic')
    
    # Separate mask output directories
    MASKS_OUTPUT_DIR = os.path.join(DATASET_DIR, 'masks')
    CONTROL_MASKS_DIR = os.path.join(MASKS_OUTPUT_DIR, 'control')
    DIABETIC_MASKS_DIR = os.path.join(MASKS_OUTPUT_DIR, 'diabetic')
    
    PANCREATIC_MASKS_OUTPUT_DIR = os.path.join(DATASET_DIR, 'pancreatic_masks')
    CONTROL_PANCREATIC_MASKS_DIR = os.path.join(PANCREATIC_MASKS_OUTPUT_DIR, 'control')
    DIABETIC_PANCREATIC_MASKS_DIR = os.path.join(PANCREATIC_MASKS_OUTPUT_DIR, 'diabetic')
    
    # Model parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 256
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'

CONFIG = Config()

def load_iris_model(model_path):
    """Load trained iris segmentation model"""
    model = smp.Unet(
        encoder_name=CONFIG.ENCODER,
        encoder_weights=CONFIG.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device(CONFIG.DEVICE)))
    model.to(CONFIG.DEVICE)
    model.eval()
    return model

def get_inference_transforms():
    """Get image preprocessing transforms for model inference"""
    return A.Compose([
        A.Resize(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def generate_iris_mask(model, image_path, transforms):
    """Generate iris mask using trained model"""
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    
    # Apply transforms
    augmented = transforms(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(CONFIG.DEVICE)
    
    # Generate prediction
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    
    # Convert back to numpy and resize to original dimensions
    mask = prediction.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (original_width, original_height))
    mask = (mask * 255).astype(np.uint8)
    
    return mask

def create_pancreas_mask_from_iris_mask(iris_mask, eye_side):
    """Generate pancreatic ROI mask from iris mask using correct ellipse method"""
    if iris_mask is None:
        return None
        
    # Find iris contours
    contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get the largest contour (iris)
    iris_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(iris_contour)
    
    iris_center = (int(x), int(y))
    iris_radius = int(radius)
    
    height, width = iris_mask.shape
    pancreas_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define ellipse parameters based on iris radius
    radius_inner = int(iris_radius * 0.40)
    radius_outer = int(iris_radius * 0.95)
    axes = (radius_outer, radius_outer)
    axes_inner = (radius_inner, radius_inner)
    
    # Determine eye side and create appropriate pancreatic region
    eye_side_lower = eye_side.lower()
    
    if eye_side_lower == 'r' or 'right' in eye_side_lower:
        # Right eye: pancreatic region at specific angles
        start_angle, end_angle = 120, 150
        cv2.ellipse(pancreas_mask, iris_center, axes, 0, start_angle, end_angle, 255, -1)
        cv2.ellipse(pancreas_mask, iris_center, axes_inner, 0, start_angle, end_angle, 0, -1)
    
    elif eye_side_lower == 'l' or 'left' in eye_side_lower:
        # Left eye: pancreatic body and tail regions
        start_angle_body, end_angle_body = 120, 150
        cv2.ellipse(pancreas_mask, iris_center, axes, 0, start_angle_body, end_angle_body, 255, -1)
        cv2.ellipse(pancreas_mask, iris_center, axes_inner, 0, start_angle_body, end_angle_body, 0, -1)
        
        start_angle_tail, end_angle_tail = 30, 60
        cv2.ellipse(pancreas_mask, iris_center, axes, 0, start_angle_tail, end_angle_tail, 255, -1)
        cv2.ellipse(pancreas_mask, iris_center, axes_inner, 0, start_angle_tail, end_angle_tail, 0, -1)
    
    return pancreas_mask

def process_directory(image_dir, class_name, model, transforms):
    """Process all images in a directory"""
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    
    # Determine output directories based on class
    if class_name.lower() == 'control':
        iris_output_dir = CONFIG.CONTROL_MASKS_DIR
        pancreatic_output_dir = CONFIG.CONTROL_PANCREATIC_MASKS_DIR
    else:  # diabetic
        iris_output_dir = CONFIG.DIABETIC_MASKS_DIR
        pancreatic_output_dir = CONFIG.DIABETIC_PANCREATIC_MASKS_DIR
    
    print(f"Processing {len(image_files)} {class_name} images...")
    
    for image_path in tqdm(image_files, desc=f"Generating {class_name} masks"):
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        
        # Determine eye side from filename
        if 'L' in filename.upper():
            eye_side = 'left'
        elif 'R' in filename.upper():
            eye_side = 'right'
        else:
            continue  # Skip if eye side cannot be determined
        
        # Generate iris mask
        iris_mask = generate_iris_mask(model, image_path, transforms)
        if iris_mask is None:
            continue
            
        # Save iris mask in appropriate class folder
        iris_mask_path = os.path.join(iris_output_dir, f"{base_name}_mask.png")
        cv2.imwrite(iris_mask_path, iris_mask)
        
        # Generate pancreatic mask using corrected method
        pancreatic_mask = create_pancreas_mask_from_iris_mask(iris_mask, eye_side)
        if pancreatic_mask is not None:
            # Save pancreatic mask in appropriate class folder
            pancreatic_mask_path = os.path.join(pancreatic_output_dir, f"{base_name}_pancreas_roi.png")
            cv2.imwrite(pancreatic_mask_path, pancreatic_mask)

def main():
    """Main execution function"""
    print("=== Unified Mask Generation System ===")
    print(f"Device: {CONFIG.DEVICE}")
    
    # Create output directories for both control and diabetic
    directories_to_create = [
        CONFIG.CONTROL_MASKS_DIR,
        CONFIG.DIABETIC_MASKS_DIR,
        CONFIG.CONTROL_PANCREATIC_MASKS_DIR,
        CONFIG.DIABETIC_PANCREATIC_MASKS_DIR
    ]
    
    for directory in directories_to_create:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Load iris segmentation model
    print("Loading iris segmentation model...")
    if os.path.exists(CONFIG.IRIS_3CLASS_MODEL):
        model = load_iris_model(CONFIG.IRIS_3CLASS_MODEL)
        print(f"Loaded 3-class model")
    elif os.path.exists(CONFIG.IRIS_2CLASS_MODEL):
        model = load_iris_model(CONFIG.IRIS_2CLASS_MODEL)
        print(f"Loaded 2-class model")
    else:
        print("Error: No trained iris model found!")
        return
    
    # Get preprocessing transforms
    transforms = get_inference_transforms()
    
    # Process control images
    if os.path.exists(CONFIG.CONTROL_IMAGES_DIR):
        process_directory(CONFIG.CONTROL_IMAGES_DIR, "Control", model, transforms)
    
    # Process diabetic images
    if os.path.exists(CONFIG.DIABETIC_IMAGES_DIR):
        process_directory(CONFIG.DIABETIC_IMAGES_DIR, "Diabetic", model, transforms)
    
    print("=== Mask Generation Complete ===")
    print(f"Control iris masks saved to: {CONFIG.CONTROL_MASKS_DIR}")
    print(f"Control pancreatic masks saved to: {CONFIG.CONTROL_PANCREATIC_MASKS_DIR}")
    print(f"Diabetic iris masks saved to: {CONFIG.DIABETIC_MASKS_DIR}")
    print(f"Diabetic pancreatic masks saved to: {CONFIG.DIABETIC_PANCREATIC_MASKS_DIR}")

if __name__ == "__main__":
    main()