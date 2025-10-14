import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import sys

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class Config:
    # Use centralized config paths
    MODEL_PATH = os.path.join(MODELS_DIR, 'best_iris_model_3class.pth')
    TEST_DIR = CONTROL_DIR
    OUTPUT_DIR = TEST_RESULTS_CONTROL_MASKS
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    IMG_SIZE = 256

CONFIG = Config()

# --- 2. Model Loading Function ---
def load_model(model_path):
    model = smp.Unet(
        encoder_name=CONFIG.ENCODER,
        encoder_weights=CONFIG.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device(CONFIG.DEVICE)))
    model.to(CONFIG.DEVICE)
    model.eval()
    print("Model loaded successfully for inference.")
    return model

# --- 3. Pre-processing Function ---
def get_inference_transforms():
    return A.Compose([
        A.Resize(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# --- 4. Main Prediction Loop ---
if __name__ == '__main__':
    model = load_model(CONFIG.MODEL_PATH)
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    image_files = [f for f in os.listdir(CONFIG.TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in '{CONFIG.TEST_DIR}'")

    transforms = get_inference_transforms()

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(CONFIG.TEST_DIR, filename)
        
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue
            
        original_h, original_w = original_image.shape[:2]
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        augmented = transforms(image=image_rgb)
        image_tensor = augmented['image'].unsqueeze(0).to(CONFIG.DEVICE)
        
        with torch.no_grad():
            logits = model(image_tensor)
            
        probs = torch.sigmoid(logits)
        predicted_mask = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        
        mask_resized = cv2.resize(predicted_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        # --- Save the results ---
        
    
        mask_to_save = mask_resized * 255
        mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
        mask_save_path = os.path.join(CONFIG.OUTPUT_DIR, mask_filename)
        cv2.imwrite(mask_save_path, mask_to_save)
    
        final_visualization = original_image.copy()
        
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(final_visualization, contours, -1, (0, 255, 0), 2)
        
        # *** NEW VISUALIZATION LOGIC ENDS HERE ***
        
        viz_filename = f"{os.path.splitext(filename)[0]}_contour.png" # Changed name for clarity
        viz_save_path = os.path.join(CONFIG.OUTPUT_DIR, viz_filename)
        '''cv2.imwrite(viz_save_path, final_visualization)'''
    print(f"\nProcessing complete! All results saved to the '{CONFIG.OUTPUT_DIR}' folder.")