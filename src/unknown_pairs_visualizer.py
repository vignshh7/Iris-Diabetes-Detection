import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
import re

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Prediction Configuration and Model Classes
def calculate_input_channels(channels_list):
    """Helper function to calculate total input channels based on a list."""
    count = 0
    if 'rgb' in channels_list: count += 3
    if 'gray' in channels_list: count += 1
    if 'hsv' in channels_list: count += 3
    if 'lab' in channels_list: count += 3
    if 'mask' in channels_list: count += 1
    return count

class PredictionConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 128
    CHANNELS_TO_USE = ['rgb', 'gray', 'hsv', 'lab', 'mask']
    INPUT_CHANNELS_PER_EYE = calculate_input_channels(CHANNELS_TO_USE)
    INPUT_CHANNELS_TOTAL = INPUT_CHANNELS_PER_EYE * 2

class SEBlock(nn.Module):
    def __init__(self,c,r=4):super(SEBlock,self).__init__();self.avg_pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Sequential(nn.Linear(c,c//r,bias=False),nn.ReLU(inplace=True),nn.Linear(c//r,c,bias=False),nn.Sigmoid())
    def forward(self,x):b,c,_,_=x.size();y=self.avg_pool(x).view(b,c);y=self.fc(y).view(b,c,1,1);return x*y.expand_as(x)

class SimplerCNN(nn.Module):
    def __init__(self, input_channels=22, num_classes=1, dropout_rate=0.5):
        super(SimplerCNN, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(input_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2)); self.s1 = SEBlock(32)
        self.c2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2)); self.s2 = SEBlock(64)
        self.c3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2)); self.s3 = SEBlock(128)
        final_feature_map_size = PredictionConfig.IMG_SIZE // 8
        flattened_size = 128 * final_feature_map_size * final_feature_map_size
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.s1(self.c1(x)); x = self.s2(self.c2(x)); x = self.s3(self.c3(x))
        x = x.view(x.size(0), -1); return self.classifier(x)

def load_and_process_image(image_path, target_size=(256, 256)):
    """Load and resize image while preserving original colors"""
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Use INTER_AREA for better quality when downscaling, preserves colors better
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            return img
    return None

def load_mask(mask_path, target_size=(256, 256)):
    """Load and process mask"""
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Use INTER_NEAREST for masks to preserve sharp edges
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            return mask
    return None

def create_border_overlay(image, mask, thickness=1):
    """Create thin green border overlay on original image"""
    if image is None or mask is None:
        return None
    
    overlay = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Draw very thin green border (thickness=1 for minimal thickness)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness)
    
    return overlay

def process_single_eye_for_prediction(image_path: str, config) -> torch.Tensor:
    """Process single eye image for prediction"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: 
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Try to find mask - if not found, use blank mask
    base_name, _ = os.path.splitext(os.path.basename(image_path))
    mask_filename = f"{base_name}_mask.png"
    
    # Try different mask directories
    mask_paths = [
        os.path.join(TEST_RESULTS_CONTROL_MASKS, mask_filename),
        os.path.join(TEST_RESULTS_DIABETIC_MASKS, mask_filename),
        os.path.join(os.path.dirname(image_path), 'masks', mask_filename),
        os.path.join(os.path.dirname(image_path), mask_filename)
    ]
    
    mask = None
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            break
    
    if mask is None:
        print(f"Info: No mask found for {image_path}. Using blank mask for prediction.")
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    resizer = A.Resize(config.IMG_SIZE, config.IMG_SIZE)
    augmented = resizer(image=img, mask=mask)
    img, mask = augmented['image'], augmented['mask']
    
    channels_to_stack = []
    if 'rgb' in config.CHANNELS_TO_USE: channels_to_stack.append(img)
    if 'gray' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[...,np.newaxis])
    if 'hsv' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    if 'lab' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    
    combined_image = np.dstack(channels_to_stack) if channels_to_stack else img
    
    final_transform = A.Compose([A.Normalize(mean=[0.5]*10, std=[0.5]*10, max_pixel_value=255.0), ToTensorV2()])
    
    if 'mask' in config.CHANNELS_TO_USE:
        final_data = final_transform(image=combined_image)
        image_tensor = final_data['image']
        mask_tensor = ToTensorV2()(image=mask)['image'].float() / 255.0
        return torch.cat([image_tensor, mask_tensor], dim=0)
    else:
        return final_transform(image=combined_image)['image']

def predict_with_ensemble(models: list, left_eye_path: str, right_eye_path: str, config) -> tuple:
    """Make prediction using ensemble of models"""
    try:
        left_tensor = process_single_eye_for_prediction(left_eye_path, config)
        right_tensor = process_single_eye_for_prediction(right_eye_path, config)
        input_tensor = torch.cat([left_tensor, right_tensor], dim=0).unsqueeze(0).to(config.DEVICE)
        
        if input_tensor.shape[1] != config.INPUT_CHANNELS_TOTAL:
            raise ValueError(f"Shape mismatch! Expected {config.INPUT_CHANNELS_TOTAL} channels, got {input_tensor.shape[1]}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Prediction error for {left_eye_path}, {right_eye_path}: {e}")
        return "Error", 0.0

    probabilities = []
    with torch.no_grad():
        for model in models:
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            probabilities.append(prob)
    
    avg_prob = np.mean(probabilities)
    label = "Diabetic" if avg_prob > 0.5 else "Control"
    return label, avg_prob

def load_prediction_models():
    """Load all trained models for ensemble prediction"""
    config = PredictionConfig()
    model_paths = glob.glob(os.path.join(MODELS_DIR, 'best_f1_model_fold_*.pth'))
    
    if not model_paths:
        print(f"Warning: No models found in '{MODELS_DIR}' matching pattern 'best_f1_model_fold_*.pth'")
        return [], config
    
    loaded_models = []
    for path in model_paths:
        print(f"Loading model: {os.path.basename(path)}")
        model = SimplerCNN(input_channels=config.INPUT_CHANNELS_TOTAL).to(config.DEVICE)
        model.load_state_dict(torch.load(path, map_location=config.DEVICE))
        model.eval()
        loaded_models.append(model)
    
    print(f"Successfully loaded {len(loaded_models)} models for predictions")
    return loaded_models, config

def discover_unknown_image_pairs(unknown_dir):
    """Discover all left-right image pairs from unknown directory"""
    pairs = []
    
    if not os.path.exists(unknown_dir):
        print(f"⚠️ Directory not found: {unknown_dir}")
        return pairs
        
    print(f"🔍 Processing unknown images in: {unknown_dir}")
    image_files = [f for f in os.listdir(unknown_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_filenames = image_files
    
    # Group images by patient ID
    left_images = [f for f in image_files if 'L.' in f or '_L.' in f or 'left' in f.lower()]
    
    for left_img in left_images:
        # Extract patient ID
        try:
            if 'L.' in left_img:
                patient_id = left_img.split('L.')[0]
                right_pattern = f"{patient_id}R."
            elif '_L.' in left_img:
                patient_id = left_img.split('_L.')[0]
                right_pattern = f"{patient_id}_R."
            else:
                # Handle other patterns
                patient_id = left_img.replace('left', '').replace('_', '').replace('.jpg', '').replace('.png', '')
                right_pattern = patient_id.replace('left', 'right')
        except:
            continue
        
        # Find corresponding right image with same patient ID
        right_candidates = [f for f in all_filenames if f.startswith(right_pattern) or right_pattern.lower() in f.lower()]
        
        if right_candidates:
            right_img = right_candidates[0]  # Take first match
            pairs.append({
                'patient_id': patient_id,
                'left_image': left_img,
                'right_image': right_img,
                'left_path': os.path.join(unknown_dir, left_img),
                'right_path': os.path.join(unknown_dir, right_img)
            })
            print(f"✅ Found pair: Patient {patient_id}")
        else:
            print(f"⚠️ Missing right image for: {left_img}")
    
    return pairs

def create_unknown_visualization(pair, output_dir, sample_number, models, pred_config):
    """Create visualization for unknown image pair (no ground truth)"""
    
    # Load images
    left_img = load_and_process_image(pair['left_path'])
    right_img = load_and_process_image(pair['right_path'])
    
    # Try to find masks (optional for unknown images)
    left_mask_path = pair['left_path'].replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
    right_mask_path = pair['right_path'].replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
    
    left_mask = load_mask(left_mask_path)
    right_mask = load_mask(right_mask_path)
    
    # Create border overlays with ultra-thin borders (thickness=1)
    left_border = create_border_overlay(left_img, left_mask, thickness=1) if left_img is not None and left_mask is not None else left_img
    right_border = create_border_overlay(right_img, right_mask, thickness=1) if right_img is not None and right_mask is not None else right_img
    
    # Get prediction (no ground truth comparison)
    if models:
        prediction, probability = predict_with_ensemble(models, pair['left_path'], pair['right_path'], pred_config)
    else:
        prediction, probability = "Error", 0.0
    
    if left_img is not None or right_img is not None:
        # Create 2x4 grid layout
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.patch.set_facecolor('white')
        
        # Title - NO ground truth comparison, only prediction
        status_text = f"Sample {sample_number:02d}: Patient {pair['patient_id']} - Predicted: {prediction}"
        fig.suptitle(status_text, fontsize=16, fontweight='bold', color='blue')
        
        # Row 1: Left Eye
        # Original
        if left_img is not None:
            axes[0, 0].imshow(left_img, vmin=0, vmax=255)
        else:
            axes[0, 0].imshow(np.zeros((256, 256, 3)))
        axes[0, 0].set_title('Left Eye\nOriginal', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Generated Mask (if available)
        if left_mask is not None:
            axes[0, 1].imshow(left_mask, cmap='gray')
        else:
            axes[0, 1].imshow(np.zeros((256, 256)), cmap='gray')
            axes[0, 1].text(0.5, 0.5, 'No Mask\nAvailable', ha='center', va='center', 
                          fontsize=12, transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Left Eye\nGenerated Mask', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Iris Border
        if left_border is not None:
            axes[0, 2].imshow(left_border, vmin=0, vmax=255)
        else:
            axes[0, 2].imshow(np.zeros((256, 256, 3)))
        axes[0, 2].set_title('Left Eye\nIris Border', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Results Panel - NO ground truth information
        axes[0, 3].text(0.1, 0.95, 'Results', fontsize=12, fontweight='bold', transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.85, f'Left Eye: {pair["left_image"]}', fontsize=9, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.75, f'Right Eye: {pair["right_image"]}', fontsize=9, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.65, f'Patient: {pair["patient_id"]}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.55, f'Status: Unknown', fontsize=10, transform=axes[0, 3].transAxes, style='italic')
        axes[0, 3].text(0.1, 0.45, f'Prediction: {prediction}', fontsize=10, fontweight='bold', transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.35, f'Confidence: {probability:.3f}', fontsize=10, transform=axes[0, 3].transAxes)
        
        # Add thin border around results
        rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1, edgecolor='black', facecolor='none', transform=axes[0, 3].transAxes)
        axes[0, 3].add_patch(rect)
        axes[0, 3].set_xlim(0, 1)
        axes[0, 3].set_ylim(0, 1)
        axes[0, 3].axis('off')
        
        # Row 2: Right Eye
        # Original
        if right_img is not None:
            axes[1, 0].imshow(right_img, vmin=0, vmax=255)
        else:
            axes[1, 0].imshow(np.zeros((256, 256, 3)))
        axes[1, 0].set_title('Right Eye\nOriginal', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Generated Mask (if available)
        if right_mask is not None:
            axes[1, 1].imshow(right_mask, cmap='gray')
        else:
            axes[1, 1].imshow(np.zeros((256, 256)), cmap='gray')
            axes[1, 1].text(0.5, 0.5, 'No Mask\nAvailable', ha='center', va='center', 
                          fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Right Eye\nGenerated Mask', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Iris Border
        if right_border is not None:
            axes[1, 2].imshow(right_border, vmin=0, vmax=255)
        else:
            axes[1, 2].imshow(np.zeros((256, 256, 3)))
        axes[1, 2].set_title('Right Eye\nIris Border', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Hide bottom right panel
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        # Save with naming format: unknown_sample_01_patient_10.png
        output_filename = f"unknown_sample_{sample_number:03d}_patient_{pair['patient_id']}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True, output_path
    else:
        print(f"❌ Failed to load images for Patient {pair['patient_id']}")
        return False, None

def create_unknown_pairs_summary(pairs, predictions, output_dir):
    """Create summary of unknown image pairs predictions"""
    
    # Create dataframe with predictions
    summary_data = []
    for i, pair in enumerate(pairs):
        pred_data = predictions.get(i, ("Error", 0.0))
        summary_data.append({
            'Sample': f"{i+1:03d}",
            'Patient_ID': pair['patient_id'],
            'Left_Image': pair['left_image'],
            'Right_Image': pair['right_image'],
            'Prediction': pred_data[0],
            'Confidence': f"{pred_data[1]:.3f}"
        })
    
    # Save to CSV
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'unknown_pairs_predictions.csv')
    df.to_csv(csv_path, index=False)
    
    # Create summary statistics
    prediction_counts = df['Prediction'].value_counts()
    
    # Create summary visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    
    # Prediction distribution
    colors = ['lightcoral' if pred == 'Diabetic' else 'lightblue' for pred in prediction_counts.index]
    ax1.pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('Prediction Distribution\n(Unknown Dataset)', fontsize=14, fontweight='bold')
    
    # Confidence distribution
    confidences = [float(x) for x in df['Confidence']]
    ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary image
    summary_image_path = os.path.join(output_dir, 'unknown_pairs_summary.png')
    plt.savefig(summary_image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return df, csv_path, summary_image_path

def generate_unknown_pairs_visualizations(unknown_images_dir):
    """Main function to generate visualizations for unknown image pairs"""
    
    print("🔍 Generating visualizations for UNKNOWN image pairs...")
    print(f"📁 Input directory: {unknown_images_dir}")
    
    # Load prediction models
    print("🤖 Loading prediction models...")
    models, pred_config = load_prediction_models()
    
    if not models:
        print("❌ No models loaded. Cannot make predictions.")
        return
    
    # Create output directory
    output_dir = os.path.join(PERFORMANCE_DIR, "unknown_pairs_results")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Discover unknown image pairs
    pairs = discover_unknown_image_pairs(unknown_images_dir)
    print(f"📊 Found {len(pairs)} unknown image pairs")
    
    if not pairs:
        print("❌ No image pairs found in unknown directory!")
        return
    
    # Generate visualizations
    print("🎨 Generating visualizations...")
    successful_pairs = []
    predictions = {}
    
    for i, pair in enumerate(tqdm(pairs, desc="Processing unknown pairs"), 1):
        success, output_path = create_unknown_visualization(pair, images_dir, i, models, pred_config)
        if success:
            # Store prediction for summary
            prediction, probability = predict_with_ensemble(models, pair['left_path'], pair['right_path'], pred_config)
            predictions[i-1] = (prediction, probability)
            successful_pairs.append(pair)
            if i <= 5:  # Show first 5 for reference
                print(f"✅ Sample {i:03d}: Patient {pair['patient_id']} - Predicted: {prediction}")
        else:
            print(f"❌ Failed: Patient {pair['patient_id']}")
    
    # Create summary
    print("📊 Creating predictions summary...")
    summary_df, csv_path, summary_image_path = create_unknown_pairs_summary(pairs, predictions, output_dir)
    
    # Final report
    print("\n" + "="*80)
    print("🎉 Unknown pairs analysis complete!")
    print(f"✅ Successfully processed: {len(successful_pairs)}/{len(pairs)} pairs")
    print(f"📁 Images saved in: {images_dir}")
    print(f"📊 Predictions CSV: {csv_path}")
    print(f"🖼️ Summary image: {summary_image_path}")
    print("="*80)
    
    # Show prediction summary
    prediction_counts = summary_df['Prediction'].value_counts()
    print(f"\n📊 Prediction Summary:")
    for pred, count in prediction_counts.items():
        percentage = (count / len(pairs)) * 100
        print(f"   - {pred}: {count} pairs ({percentage:.1f}%)")
    
    return successful_pairs, predictions

# Example usage function
def main():
    """
    Example usage of the unknown pairs visualizer
    
    To use this script:
    1. Place unknown eye images in a directory
    2. Ensure left/right pairs follow naming convention (e.g., patient_1L.jpg, patient_1R.jpg)
    3. Run: python unknown_pairs_visualizer.py
    """
    
    # CONFIGURE THIS PATH TO YOUR UNKNOWN IMAGES DIRECTORY
    unknown_images_directory = input("Enter path to unknown images directory: ").strip()
    
    if not os.path.exists(unknown_images_directory):
        print(f"❌ Directory not found: {unknown_images_directory}")
        return
    
    # Generate visualizations
    results = generate_unknown_pairs_visualizations(unknown_images_directory)
    
    if results:
        successful_pairs, predictions = results
        print(f"\n🎯 Analysis complete! Check the results in '{os.path.join(PERFORMANCE_DIR, 'unknown_pairs_results')}'")
    else:
        print("❌ Analysis failed or no results generated.")

if __name__ == "__main__":
    # Example: Uncomment the line below to run with a specific directory
    # generate_unknown_pairs_visualizations("path/to/your/unknown/images")
    
    # Or run interactive mode
    main()