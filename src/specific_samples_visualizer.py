import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.patches import Rectangle
import pandas as pd

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def load_and_process_image(image_path, target_size=(256, 256)):
    """Load and resize image"""
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            return img
    return None

def load_mask(mask_path, target_size=(256, 256)):
    """Load and process mask"""
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, target_size)
            return mask
    return None

def create_border_overlay(image, mask):
    """Create green border overlay on original image"""
    if image is None or mask is None:
        return None
    
    overlay = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Draw green border (contour only)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)
    
    return overlay

def create_border_overlay(image, mask):
    """Create green border overlay on original image with thinner lines"""
    if image is None or mask is None:
        return None
    
    overlay = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Draw thinner green border (contour only) - reduced thickness from 3 to 2
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    return overlay

def create_sample_visualization():
    """Create visualizations for specific 10 samples with merged left-right eyes"""
    
    # Define the specific samples from the user's table - NON-OVERLAPPING PATIENT IDs
    samples = [
        # Control subjects (Patient IDs: 10, 11, 12, 13, 14, 16, 36)
        {"patient_id": 10, "left": "10L.IMG20250525165821.jpg", "right": "10R.IMG20250525165746.jpg", 
         "ground_truth": "Control", "prediction": "Control", "probability": 0.3626, "status": "Correct"},
        {"patient_id": 11, "left": "11L.IMG20250525171900.jpg", "right": "11R.IMG20250525170623.jpg", 
         "ground_truth": "Control", "prediction": "Control", "probability": 0.3549, "status": "Correct"},
        {"patient_id": 12, "left": "12L.IMG20250525173006.jpg", "right": "12R.IMG20250525172933.jpg", 
         "ground_truth": "Control", "prediction": "Control", "probability": 0.4752, "status": "Correct"},
        {"patient_id": 13, "left": "13L.IMG20250525173535.jpg", "right": "13R.IMG20250525173511.jpg", 
         "ground_truth": "Control", "prediction": "Control", "probability": 0.3757, "status": "Correct"},
        {"patient_id": 14, "left": "14L.IMG20250529204151.jpg", "right": "14R.IMG20250525173609.jpg", 
         "ground_truth": "Control", "prediction": "Control", "probability": 0.4200, "status": "Correct"},
        {"patient_id": 16, "left": "16L.IMG20250525174009.jpg", "right": "16R.IMG20250525173953.jpg", 
         "ground_truth": "Control", "prediction": "Diabetic", "probability": 0.5088, "status": "Incorrect"},
        {"patient_id": 36, "left": "36L.IMG20250528153939.jpg", "right": "36R.IMG20250528154014.jpg", 
         "ground_truth": "Control", "prediction": "Diabetic", "probability": 0.5840, "status": "Incorrect"},
        
        # Diabetic subjects (Different Patient IDs: 20, 21, 22) - Using different IDs to avoid overlap
        {"patient_id": 20, "left": "10L.IMG20250609110754.jpg", "right": "10R.IMG20250609110821.jpg", 
         "ground_truth": "Diabetic", "prediction": "Diabetic", "probability": 0.6674, "status": "Correct"},
        {"patient_id": 21, "left": "11L.IMG20250609111305.jpg", "right": "11R.IMG20250609111457.jpg", 
         "ground_truth": "Diabetic", "prediction": "Diabetic", "probability": 0.5926, "status": "Correct"},
        {"patient_id": 22, "left": "12L.IMG20250609115938.jpg", "right": "12R.IMG20250609115859.jpg", 
         "ground_truth": "Diabetic", "prediction": "Diabetic", "probability": 0.6133, "status": "Correct"}
    ]
    
    # Base directories from config
    dataset_dir = DATASET_DIR
    mask_dir = TEST_RESULTS_MASKS_DIR
    output_dir = SAMPLE_IMAGES_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Creating visualizations for specific 10 samples...")
    
    successful_samples = []
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/10: Patient {sample['patient_id']} - {sample['ground_truth']}")
        
        # Determine source directory based on ground truth
        if sample['ground_truth'] == 'Control':
            img_dir = CONTROL_DIR
            mask_subdir = TEST_RESULTS_CONTROL_MASKS
        else:
            img_dir = DIABETIC_DIR
            mask_subdir = TEST_RESULTS_DIABETIC_MASKS
        
        # Load both left and right eye images
        left_img_path = os.path.join(img_dir, sample['left'])
        right_img_path = os.path.join(img_dir, sample['right'])
        left_mask_path = os.path.join(mask_subdir, sample['left'].replace('.jpg', '_mask.png'))
        right_mask_path = os.path.join(mask_subdir, sample['right'].replace('.jpg', '_mask.png'))
        
        left_img = load_and_process_image(left_img_path)
        right_img = load_and_process_image(right_img_path)
        left_mask = load_mask(left_mask_path)
        right_mask = load_mask(right_mask_path)
        
        # Create border overlays with thinner green lines
        left_border = create_border_overlay(left_img, left_mask) if left_img is not None and left_mask is not None else None
        right_border = create_border_overlay(right_img, right_mask) if right_img is not None and right_mask is not None else None
        
        if left_img is not None or right_img is not None:
            # Create 7-panel visualization like the user's example (3 columns for left, 3 for right, 1 for results)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            # Title with color coding
            status_color = 'green' if sample['status'] == 'Correct' else 'red'
            sample_number = f"Sample {i+1:02d}"
            prediction_status = f"{sample['ground_truth']} → {sample['prediction']} ({'CORRECT' if sample['status'] == 'Correct' else 'INCORRECT'})"
            
            fig.suptitle(f'{sample_number}: {prediction_status}', 
                        fontsize=16, fontweight='bold', color=status_color)
            
            # Left Eye Row (Top)
            # Panel 1: Left Eye Original
            if left_img is not None:
                axes[0, 0].imshow(left_img)
                axes[0, 0].set_title('Left Eye\nOriginal', fontsize=10, fontweight='bold')
            else:
                axes[0, 0].text(0.5, 0.5, 'Left Eye\nNot Available', ha='center', va='center', 
                               transform=axes[0, 0].transAxes, fontsize=10)
            axes[0, 0].axis('off')
            
            # Panel 2: Left Eye Mask
            if left_mask is not None:
                axes[0, 1].imshow(left_mask, cmap='gray')
                axes[0, 1].set_title('Left Eye\nGenerated Mask', fontsize=10, fontweight='bold')
            else:
                axes[0, 1].text(0.5, 0.5, 'Left Mask\nNot Available', ha='center', va='center', 
                               transform=axes[0, 1].transAxes, fontsize=10)
            axes[0, 1].axis('off')
            
            # Panel 3: Left Eye Border
            if left_border is not None:
                axes[0, 2].imshow(left_border)
                axes[0, 2].set_title('Left Eye\nIris Border', fontsize=10, fontweight='bold')
            else:
                axes[0, 2].text(0.5, 0.5, 'Left Border\nNot Available', ha='center', va='center', 
                               transform=axes[0, 2].transAxes, fontsize=10)
            axes[0, 2].axis('off')
            
            # Results Panel (spans both rows)
            axes[0, 3].axis('off')
            axes[1, 3].axis('off')
            
            # Combine the results area
            result_text = f"""Results

{sample_number}
Patient {sample['patient_id']}

Ground Truth:
{sample['ground_truth']}

Prediction:
{sample['prediction']}

Probability:
{sample['probability']:.3f}"""
            
            # Create a combined text area across both result panels
            fig.text(0.78, 0.5, result_text, fontsize=12, verticalalignment='center', 
                    horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            # Right Eye Row (Bottom)
            # Panel 4: Right Eye Original
            if right_img is not None:
                axes[1, 0].imshow(right_img)
                axes[1, 0].set_title('Right Eye\nOriginal', fontsize=10, fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'Right Eye\nNot Available', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=10)
            axes[1, 0].axis('off')
            
            # Panel 5: Right Eye Mask
            if right_mask is not None:
                axes[1, 1].imshow(right_mask, cmap='gray')
                axes[1, 1].set_title('Right Eye\nGenerated Mask', fontsize=10, fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, 'Right Mask\nNot Available', ha='center', va='center', 
                               transform=axes[1, 1].transAxes, fontsize=10)
            axes[1, 1].axis('off')
            
            # Panel 6: Right Eye Border
            if right_border is not None:
                axes[1, 2].imshow(right_border)
                axes[1, 2].set_title('Right Eye\nIris Border', fontsize=10, fontweight='bold')
            else:
                axes[1, 2].text(0.5, 0.5, 'Right Border\nNot Available', ha='center', va='center', 
                               transform=axes[1, 2].transAxes, fontsize=10)
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)  # Make room for results panel
            
            # Save the visualization
            output_path = os.path.join(output_dir, f'sample_{i+1:02d}_patient_{sample["patient_id"]}_{sample["ground_truth"].lower()}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            successful_samples.append(sample)
            print(f"✅ Saved visualization: {output_path}")
        else:
            print(f"❌ Could not load images for Patient {sample['patient_id']}")
            print(f"   Left image: {left_img_path} ({'Found' if left_img is not None else 'Not found'})")
            print(f"   Right image: {right_img_path} ({'Found' if right_img is not None else 'Not found'})")
            print(f"   Left mask: {left_mask_path} ({'Found' if left_mask is not None else 'Not found'})")
            print(f"   Right mask: {right_mask_path} ({'Found' if right_mask is not None else 'Not found'})")
    
    return successful_samples

def create_results_table_image(samples):
    """Create a results table as an image instead of HTML"""
    
    # Create DataFrame
    df_data = []
    for sample in samples:
        df_data.append({
            'Patient_ID': sample['patient_id'],
            'Left_Image': sample['left'],
            'Right_Image': sample['right'],
            'Ground_Truth': sample['ground_truth'],
            'Prediction': sample['prediction'],
            'Avg_Probability': f"{sample['probability']:.4f}",
            'Result_Status': sample['status']
        })
    
    df = pd.DataFrame(df_data)
    
    # Save as CSV
    csv_path = os.path.join(SAMPLE_RESULTS_DIR, "specific_samples_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 Results CSV saved: {csv_path}")
    
    # Create table as image
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data with proper formatting
    table_data = []
    headers = ['Patient ID', 'Left Image', 'Right Image', 'Ground Truth', 'Prediction', 'Probability', 'Result Status']
    table_data.append(headers)
    
    for _, row in df.iterrows():
        table_data.append([
            str(row['Patient_ID']),
            row['Left_Image'][:25] + '...' if len(row['Left_Image']) > 25 else row['Left_Image'],
            row['Right_Image'][:25] + '...' if len(row['Right_Image']) > 25 else row['Right_Image'],
            row['Ground_Truth'],
            row['Prediction'],
            row['Avg_Probability'],
            row['Result_Status']
        ])
    
    # Create the table
    table = ax.table(cellText=table_data[1:], colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.08, 0.25, 0.25, 0.12, 0.12, 0.10, 0.12])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the cells
    for i in range(len(df)):
        row_idx = i + 1  # +1 because of header
        
        # Color based on ground truth
        if df.iloc[i]['Ground_Truth'] == 'Control':
            bg_color = '#e8f5e8'  # Light green
        else:
            bg_color = '#ffe8e8'  # Light red
        
        # Apply background color to ground truth column
        table[(row_idx, 3)].set_facecolor(bg_color)
        
        # Color based on result status
        if df.iloc[i]['Result_Status'] == 'Correct':
            status_color = '#27ae60'  # Green
        else:
            status_color = '#e74c3c'  # Red
        
        table[(row_idx, 6)].set_facecolor(bg_color)
        table[(row_idx, 6)].set_text_props(weight='bold', color=status_color)
        
        # Color prediction column based on correctness
        table[(row_idx, 4)].set_text_props(weight='bold', 
                                          color=status_color if df.iloc[i]['Result_Status'] == 'Correct' else '#e74c3c')
    
    # Style header row
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Add title and statistics
    plt.title('📊 Sample Results - Diabetes Detection from Iris Images', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Calculate statistics
    total_samples = len(df)
    correct_predictions = len(df[df['Result_Status'] == 'Correct'])
    accuracy = (correct_predictions / total_samples) * 100
    control_samples = len(df[df['Ground_Truth'] == 'Control'])
    diabetic_samples = len(df[df['Ground_Truth'] == 'Diabetic'])
    
    # Add statistics text
    stats_text = f"""Summary Statistics:
Total Samples: {total_samples} | Correct Predictions: {correct_predictions} | Sample Accuracy: {accuracy:.1f}%
Control Subjects: {control_samples} | Diabetic Subjects: {diabetic_samples}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
🔬 Advanced Computer Vision System for Non-Invasive Diabetes Detection"""
    
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Save table as image
    table_image_path = os.path.join(SAMPLE_RESULTS_DIR, "results_table_image.png")
    plt.savefig(table_image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🖼️ Results table image saved: {table_image_path}")
    return df

if __name__ == "__main__":
    print("🎯 Creating specific sample visualizations...")
    
    # Create visualizations
    successful_samples = create_sample_visualization()
    
    # Create results table as image
    if successful_samples:
        results_df = create_results_table_image(successful_samples)
        print(f"\n✅ Successfully processed {len(successful_samples)} samples")
        print(f"📁 Visualizations saved in: {SAMPLE_IMAGES_DIR}")
        print(f"📊 Results CSV: {os.path.join(SAMPLE_RESULTS_DIR, 'specific_samples_results.csv')}")
        print(f"🖼️ Results table image: {os.path.join(SAMPLE_RESULTS_DIR, 'results_table_image.png')}")
    else:
        print("❌ No samples were successfully processed")