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
    """Process single eye image for prediction - same as cnnpredict.py"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: 
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Get mask path - try different possible mask locations
    base_name, _ = os.path.splitext(os.path.basename(image_path))
    mask_filename = f"{base_name}_pancreas_roi.png"
    
    # Try different mask directories
    mask_paths = [
        os.path.join(DATASET_DIR, 'pancreas_masks_for_training', 'control', mask_filename),
        os.path.join(DATASET_DIR, 'pancreas_masks_for_training', 'diabetic', mask_filename),
        os.path.join(TEST_RESULTS_CONTROL_MASKS, mask_filename.replace('_pancreas_roi', '_mask')),
        os.path.join(TEST_RESULTS_DIABETIC_MASKS, mask_filename.replace('_pancreas_roi', '_mask'))
    ]
    
    mask = None
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            break
    
    if mask is None:
        print(f"Warning: Mask not found for {image_path}. Using blank mask.")
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

def get_all_image_pairs():
    """Get all available image pairs from control and diabetic directories"""
    pairs = []
    
    def process_directory(directory, ground_truth, category):
        """Process a single directory for image pairs"""
        all_images = glob.glob(os.path.join(directory, "*.jpg"))
        all_filenames = [os.path.basename(img) for img in all_images]
        
        left_images = [img for img in all_images if 'L.' in os.path.basename(img)]
        
        for left_path in left_images:
            left_filename = os.path.basename(left_path)
            
            # Extract patient ID
            if 'L.' in left_filename:
                patient_id = left_filename.split('L.')[0]
            else:
                continue  # Skip if pattern doesn't match
            
            # Find any right image with same patient ID (different timestamps OK)
            right_candidates = [f for f in all_filenames if f.startswith(f"{patient_id}R.")]
            
            if right_candidates:
                right_filename = right_candidates[0]  # Take first match
                right_path = os.path.join(directory, right_filename)
                
                pairs.append({
                    'patient_id': patient_id,
                    'left_image': left_filename,
                    'right_image': right_filename,
                    'left_path': left_path,
                    'right_path': right_path,
                    'ground_truth': ground_truth,
                    'category': category
                })
                print(f"✅ Found pair: Patient {patient_id} ({ground_truth})")
            else:
                print(f"⚠️ Missing right image for: {left_filename}")
    
    # Process Control images
    print("🔍 Processing Control images...")
    process_directory(CONTROL_DIR, 'Control', 'control')
    
    # Process Diabetic images  
    print("🔍 Processing Diabetic images...")
    process_directory(DIABETIC_DIR, 'Diabetic', 'diabetic')
    
    return pairs

def create_visualization_for_pair(pair, output_dir, sample_number, models, pred_config):
    """Create visualization for a single image pair with real predictions"""
    
    # Load images
    left_img = load_and_process_image(pair['left_path'])
    right_img = load_and_process_image(pair['right_path'])
    
    # Load masks
    mask_dir = TEST_RESULTS_CONTROL_MASKS if pair['category'] == 'control' else TEST_RESULTS_DIABETIC_MASKS
    left_mask_path = os.path.join(mask_dir, pair['left_image'].replace('.jpg', '_mask.png'))
    right_mask_path = os.path.join(mask_dir, pair['right_image'].replace('.jpg', '_mask.png'))
    
    left_mask = load_mask(left_mask_path)
    right_mask = load_mask(right_mask_path)
    
    # Create ultra-thin border overlays (thickness=1)
    left_border = create_border_overlay(left_img, left_mask, thickness=1) if left_img is not None and left_mask is not None else None
    right_border = create_border_overlay(right_img, right_mask, thickness=1) if right_img is not None and right_mask is not None else None
    
    # Get real prediction
    if models:
        prediction, probability = predict_with_ensemble(models, pair['left_path'], pair['right_path'], pred_config)
    else:
        prediction, probability = "Error", 0.0
    
    if left_img is not None or right_img is not None:
        # Create 2x4 grid layout exactly like reference
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.patch.set_facecolor('white')
        
        # Title with actual prediction status
        is_correct = prediction == pair['ground_truth']
        status_text = f"Sample {sample_number:02d}: {pair['ground_truth']} → {prediction} ({'CORRECT' if is_correct else 'INCORRECT'})"
        title_color = 'green' if is_correct else 'red'
        fig.suptitle(status_text, fontsize=16, fontweight='bold', color=title_color)
        
        # Row 1: Left Eye
        # Original
        if left_img is not None:
            axes[0, 0].imshow(left_img, vmin=0, vmax=255)  # Preserve original colors
        else:
            axes[0, 0].imshow(np.zeros((256, 256, 3)))
        axes[0, 0].set_title('Left Eye\nOriginal', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Generated Mask
        if left_mask is not None:
            axes[0, 1].imshow(left_mask, cmap='gray')
        else:
            axes[0, 1].imshow(np.zeros((256, 256)), cmap='gray')
        axes[0, 1].set_title('Left Eye\nGenerated Mask', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Iris Border
        if left_border is not None:
            axes[0, 2].imshow(left_border, vmin=0, vmax=255)  # Already in RGB format
        else:
            axes[0, 2].imshow(np.zeros((256, 256, 3)))
        axes[0, 2].set_title('Left Eye\nIris Border', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Results Panel (top right) - centered content with real predictions
        axes[0, 3].text(0.1, 0.95, 'Results', fontsize=12, fontweight='bold', transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.85, f'Left Eye: {pair["left_image"]}', fontsize=9, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.75, f'Right Eye: {pair["right_image"]}', fontsize=9, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.65, f'Patient: {pair["patient_id"]}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.55, f'Ground Truth: {pair["ground_truth"]}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.45, f'Prediction: {prediction}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.35, f'Probability: {probability:.3f}', fontsize=10, transform=axes[0, 3].transAxes)
        
        # Add thin border around results
        rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1, edgecolor='black', facecolor='none', transform=axes[0, 3].transAxes)
        axes[0, 3].add_patch(rect)
        axes[0, 3].set_xlim(0, 1)
        axes[0, 3].set_ylim(0, 1)
        axes[0, 3].axis('off')
        
        # Row 2: Right Eye
        # Original
        if right_img is not None:
            axes[1, 0].imshow(right_img, vmin=0, vmax=255)  # Preserve original colors
        else:
            axes[1, 0].imshow(np.zeros((256, 256, 3)))
        axes[1, 0].set_title('Right Eye\nOriginal', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Generated Mask
        if right_mask is not None:
            axes[1, 1].imshow(right_mask, cmap='gray')
        else:
            axes[1, 1].imshow(np.zeros((256, 256)), cmap='gray')
        axes[1, 1].set_title('Right Eye\nGenerated Mask', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Iris Border
        if right_border is not None:
            axes[1, 2].imshow(right_border, vmin=0, vmax=255)  # Already in RGB format
        else:
            axes[1, 2].imshow(np.zeros((256, 256, 3)))
        axes[1, 2].set_title('Right Eye\nIris Border', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Hide bottom right panel (results spans both rows)
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        # Save with exact naming format: sample_01_patient_10_control.png
        output_filename = f'sample_{sample_number:02d}_patient_{pair["patient_id"]}_{pair["category"]}.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True, output_path
    
    return False, None

def create_all_pairs_summary(pairs, output_dir):
    """Create a summary CSV and image table for all pairs"""
    
    # Create DataFrame
    df_data = []
    for i, pair in enumerate(pairs, 1):
        df_data.append({
            'Sample_Number': f'{i:03d}',
            'Patient_ID': pair['patient_id'],
            'Left_Image': pair['left_image'],
            'Right_Image': pair['right_image'],
            'Ground_Truth': pair['ground_truth'],
            'Category': pair['category'].title(),
            'Images_Available': 'Yes' if os.path.exists(pair['left_path']) and os.path.exists(pair['right_path']) else 'No'
        })
    
    df = pd.DataFrame(df_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "all_image_pairs_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 Summary CSV saved: {csv_path}")
    
    # Create summary statistics image
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics
    total_pairs = len(df)
    control_pairs = len(df[df['Ground_Truth'] == 'Control'])
    diabetic_pairs = len(df[df['Ground_Truth'] == 'Diabetic'])
    available_pairs = len(df[df['Images_Available'] == 'Yes'])
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Image Pairs', str(total_pairs)],
        ['Control Pairs', str(control_pairs)],
        ['Diabetic Pairs', str(diabetic_pairs)],
        ['Available Pairs', str(available_pairs)],
        ['Success Rate', f'{(available_pairs/total_pairs)*100:.1f}%']
    ]
    
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                    cellLoc='center', loc='center',
                    colWidths=[0.4, 0.3])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 3)
    
    # Style header row
    for j in range(len(summary_data[0])):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Color code rows
    colors = ['#e8f5e8', '#f8f9fa', '#ffe8e8', '#e8f4f8', '#fff2e8', '#f0e8ff']
    for i in range(1, len(summary_data)):
        for j in range(len(summary_data[0])):
            table[(i, j)].set_facecolor(colors[(i-1) % len(colors)])
    
    plt.title('📊 All Image Pairs - Summary Statistics', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Add additional info
    info_text = f"""Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
🔬 Complete Image Pair Analysis - Diabetes Detection from Iris Images
📁 Visualizations saved with ultra-thin iris borders (thickness=1px)
🎯 Author: Vignesh Venaktesan | Year: 2025"""
    
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Save summary image
    summary_image_path = os.path.join(output_dir, "all_pairs_summary.png")
    plt.savefig(summary_image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🖼️ Summary image saved: {summary_image_path}")
    return df

def generate_all_pairs_visualizations():
    """Main function to generate visualizations for all image pairs with real predictions"""
    
    print("🎯 Generating visualizations for ALL image pairs...")
    
    # Load prediction models
    print("🤖 Loading prediction models...")
    models, pred_config = load_prediction_models()
    
    # Create output directory
    output_dir = os.path.join(PERFORMANCE_DIR, "all_pairs_results")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Get all image pairs
    print("🔍 Discovering all image pairs...")
    pairs = get_all_image_pairs()
    print(f"📊 Found {len(pairs)} image pairs")
    
    if not pairs:
        print("❌ No image pairs found!")
        return [], []
    
    # Generate visualizations
    print("🎨 Generating visualizations...")
    successful_pairs = []
    failed_pairs = []
    
    for i, pair in enumerate(tqdm(pairs, desc="Processing pairs"), 1):
        success, output_path = create_visualization_for_pair(pair, images_dir, i, models, pred_config)
        if success:
            successful_pairs.append(pair)
            if i <= 5:  # Show first 5 for reference
                print(f"✅ Sample {i:03d}: Patient {pair['patient_id']} ({pair['ground_truth']})")
        else:
            failed_pairs.append(pair)
            print(f"❌ Failed: Patient {pair['patient_id']} ({pair['ground_truth']})")
    
    # Create summary
    print("📊 Creating summary...")
    summary_df = create_all_pairs_summary(pairs, output_dir)
    
    # Final report
    print(f"\n🎉 All pairs visualization complete!")
    print(f"✅ Successfully processed: {len(successful_pairs)}/{len(pairs)} pairs")
    print(f"📁 Images saved in: {images_dir}")
    print(f"📊 Summary saved in: {output_dir}")
    
    if failed_pairs:
        print(f"⚠️ Failed pairs: {len(failed_pairs)}")
        for pair in failed_pairs[:5]:  # Show first 5 failures
            print(f"   - Patient {pair['patient_id']} ({pair['ground_truth']})")
    
    print(f"\n🎯 Key Features:")
    print(f"   - Ultra-thin iris borders (thickness=1px)")
    print(f"   - 2x4 panel layout per pair")
    print(f"   - High-resolution images (300 DPI)")
    print(f"   - Complete dataset coverage")
    
    return successful_pairs, failed_pairs

if __name__ == "__main__":
    try:
        result = generate_all_pairs_visualizations()
        if result:
            successful, failed = result
            print(f"\n🎯 Final Results:")
            print(f"✅ Successful: {len(successful)}")
            print(f"❌ Failed: {len(failed)}")
        else:
            print("❌ No results generated")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()