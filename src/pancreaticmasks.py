import os
import cv2
import numpy as np
import glob
import sys
from tqdm import tqdm

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# --- 1. Configuration with Separated Outputs ---
class Config:
    # --- Input Directories ---
    # These folders contain the original iris images in .jpg format
    DIABETIC_IMAGE_DIR = DIABETIC_DIR + '/'
    CONTROL_IMAGE_DIR = CONTROL_DIR + '/'
    # --- NEW: Add the input directory for testing images ---
    TESTING_IMAGE_DIR = TESTING_DIR + '/'

    # These folders contain the corresponding accurate IRIS MASKS in .png format
    DIABETIC_IRIS_MASK_DIR = TEST_RESULTS_DIABETIC_MASKS + '/'
    CONTROL_IRIS_MASK_DIR = TEST_RESULTS_CONTROL_MASKS + '/'
    # --- NEW: Add the input directory for testing iris masks ---
    TESTING_IRIS_MASK_DIR = TEST_RESULTS_TESTING_MASKS + '/'

    # --- Output Directories ---
    # Define separate output folders for the generated pancreas masks
    OUTPUT_DIABETIC_PANCREAS_DIR = PANCREAS_DIABETIC_DIR + '/'
    OUTPUT_CONTROL_PANCREAS_DIR = PANCREAS_CONTROL_DIR + '/'
    # --- NEW: Add the output directory for testing pancreas masks ---
    OUTPUT_TESTING_PANCREAS_DIR = PANCREAS_TESTING_DIR + '/'

CONFIG = Config()

# --- 2. Core Mask Generation Function ---
# This function is unchanged.
def create_pancreas_mask_from_iris_mask(iris_mask_path, eye_side):
    iris_mask_img = cv2.imread(iris_mask_path, cv2.IMREAD_GRAYSCALE)
    if iris_mask_img is None:
        print(f"\nWarning: Could not read or mask is empty: {iris_mask_path}")
        return None

    contours, _ = cv2.findContours(iris_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"\nWarning: No contours found in iris mask: {iris_mask_path}")
        return None

    iris_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(iris_contour)
    iris_center = (int(x), int(y))
    iris_radius = int(radius)

    height, width = iris_mask_img.shape
    pancreas_mask = np.zeros((height, width), dtype=np.uint8)

    radius_inner = int(iris_radius * 0.40)
    radius_outer = int(iris_radius * 0.95)
    axes = (radius_outer, radius_outer)
    axes_inner = (radius_inner, radius_inner)

    if eye_side == 'right':
        start_angle, end_angle = 120, 150
        cv2.ellipse(pancreas_mask, iris_center, axes, 0, start_angle, end_angle, 255, -1)
        cv2.ellipse(pancreas_mask, iris_center, axes_inner, 0, start_angle, end_angle, 0, -1)
    
    elif eye_side == 'left':
        start_angle_body, end_angle_body = 120, 150
        cv2.ellipse(pancreas_mask, iris_center, axes, 0, start_angle_body, end_angle_body, 255, -1)
        cv2.ellipse(pancreas_mask, iris_center, axes_inner, 0, start_angle_body, end_angle_body, 0, -1)
        
        start_angle_tail, end_angle_tail = 30, 60
        cv2.ellipse(pancreas_mask, iris_center, axes, 0, start_angle_tail, end_angle_tail, 255, -1)
        cv2.ellipse(pancreas_mask, iris_center, axes_inner, 0, start_angle_tail, end_angle_tail, 0, -1)
    
    return pancreas_mask


# --- 3. Main Execution Block with Sorted Output Logic ---
if __name__ == '__main__':
    print("--- Pancreas ROI Mask Generation with Sorted Output ---")
    
    # Define a list that includes the specific output directory for each category
    processing_sets = [
        ("Diabetic", CONFIG.DIABETIC_IMAGE_DIR, CONFIG.DIABETIC_IRIS_MASK_DIR, CONFIG.OUTPUT_DIABETIC_PANCREAS_DIR),
        ("Control", CONFIG.CONTROL_IMAGE_DIR, CONFIG.CONTROL_IRIS_MASK_DIR, CONFIG.OUTPUT_CONTROL_PANCREAS_DIR),
        # --- NEW: Add the "Testing" set to the list of jobs to be processed ---
        ("Testing", CONFIG.TESTING_IMAGE_DIR, CONFIG.TESTING_IRIS_MASK_DIR, CONFIG.OUTPUT_TESTING_PANCREAS_DIR)
    ]
    
    # Create all necessary output directories before starting
    for _, _, _, output_dir in processing_sets:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensuring output directory exists: '{output_dir}'")
    
    total_processed = 0
    total_skipped = 0

    # The loop will now automatically process the "Testing" set as well
    for category, image_dir, iris_mask_dir, output_dir in processing_sets:
        print(f"\n--- Processing Category: {category} ---")
        
        image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
        
        if not image_files:
            print(f"No .jpg images found in '{image_dir}'. Skipping category.")
            continue
            
        for image_path in tqdm(image_files, desc=f"Generating {category} masks"):
            image_filename = os.path.basename(image_path)
            base_name, _ = os.path.splitext(image_filename)
            
            iris_mask_filename = f"{base_name}_mask.png"
            iris_mask_path = os.path.join(iris_mask_dir, iris_mask_filename)
            
            if not os.path.exists(iris_mask_path):
                print(f"\n[SKIP] Could not find mask for '{image_filename}'")
                total_skipped += 1
                continue

            # This logic assumes your test file names also contain 'L.IMG' or 'R.IMG'
            if 'L.IMG' in image_filename.upper():
                eye_side = 'left'
            elif 'R.IMG' in image_filename.upper():
                eye_side = 'right'
            else:
                print(f"\n[SKIP] Could not determine eye side for {image_filename}.")
                total_skipped += 1
                continue

            pancreas_mask = create_pancreas_mask_from_iris_mask(iris_mask_path, eye_side)
            
            if pancreas_mask is not None:
                output_filename = f"{base_name}_pancreas_roi.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, pancreas_mask)
                total_processed += 1

    print("\n--- Processing Summary ---")
    print(f"Successfully generated {total_processed} pancreas masks.")
    print(f"Skipped {total_skipped} images due to missing masks or naming issues.")
    print("Your sorted pancreas mask dataset is ready.")