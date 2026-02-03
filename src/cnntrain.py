import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import glob
import torchmetrics
from collections import defaultdict
import re
import numpy as np
from pkg_resources import parse_version
import sys
import random
from sklearn.metrics import f1_score
import json

# --- REPRODUCIBILITY SEEDS (Academic Standard) ---
# Fixed seeds ensure reproducible results across runs for academic validation
def set_reproducibility_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_reproducibility_seeds(42)

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.data_manager import create_data_manager

# --- 0. Environment Sanity Check ---
assert parse_version(A.__version__) >= parse_version("1.0.0"), "Update albumentations"
print(f"Albumentations version {A.__version__} is OK.")

# --- 1. Focal Loss (Unchanged) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(); self.alpha=alpha; self.gamma=gamma; self.reduction=reduction
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

# --- 2. Configuration (TUNED & RECONFIGURED) ---
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
    # Use centralized config paths
    DIABETIC_IMAGE_DIR = DIABETIC_DIR
    CONTROL_IMAGE_DIR = CONTROL_DIR
    DIABETIC_PANCREAS_MASK_DIR = DIABETIC_PANCREATIC_MASKS_DIR
    CONTROL_PANCREAS_MASK_DIR = CONTROL_PANCREATIC_MASKS_DIR
    
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'best_f1_model_fold_{}.pth')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE=128; BATCH_SIZE=4; LEARNING_RATE=1e-4  # Increased for small dataset + focal loss
    NUM_EPOCHS=50
    
    K_FOLDS = 5 

    # --- MODIFIED: Using all available image channels with mask as spatial attention ---
    CHANNELS_TO_USE = ['rgb', 'gray', 'hsv', 'lab', 'mask'] 
    
    # Calculate input channels: mask is not counted as separate channel (spatial attention)
    INPUT_CHANNELS = calculate_input_channels([c for c in CHANNELS_TO_USE if c != 'mask']) * 2 
    
    WEIGHT_DECAY = 5e-4
    DROPOUT_RATE = 0.5 
    FOCAL_LOSS_ALPHA = 0.4
    FOCAL_LOSS_GAMMA = 3.0

CONFIG = Config()
print(f"Using {CONFIG.K_FOLDS}-Fold Cross-Validation.")
print(f"Input channels configured to use: {CONFIG.CHANNELS_TO_USE} ({CONFIG.INPUT_CHANNELS} total channels)")

# --- 3. Model Definitions ---
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

# --- 4. Data Handling ---
def get_transforms(is_train=True, num_image_channels=None):
    geo_transforms = [A.Resize(Config.IMG_SIZE, Config.IMG_SIZE)]
    
    # Dynamic normalization for image channels only (not mask)
    # Infer channels if not provided - mask will be handled separately
    if num_image_channels is None:
        num_image_channels = calculate_input_channels([c for c in Config.CHANNELS_TO_USE if c != 'mask'])
    
    final_transform = A.Compose([A.Normalize(mean=[0.5]*num_image_channels, std=[0.5]*num_image_channels, max_pixel_value=255.0), ToTensorV2()])
    
    if is_train:
        geo_transforms.extend([
            A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.8),
            A.RandomBrightnessContrast(p=0.8), A.GaussNoise(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5)
        ])
    return A.Compose(geo_transforms), final_transform

def process_single_eye(image_path, label, config, geo_transform, final_transform, apply_mask_attention=True):
    img=cv2.imread(image_path);img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    b,_=os.path.splitext(os.path.basename(image_path));mf=f"{b}_pancreas_roi.png"
    
    # Use class-specific pancreatic masks directory
    if label == 1:  # Diabetic
        mp=os.path.join(config.DIABETIC_PANCREAS_MASK_DIR, mf)
    else:  # Control
        mp=os.path.join(config.CONTROL_PANCREAS_MASK_DIR, mf)
    
    # Load mask silently - create default if missing
    if os.path.exists(mp):
        mask=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
    else:
        mask = None
    
    if mask is None: mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if geo_transform: aug=geo_transform(image=img,mask=mask);img,mask=aug['image'],aug['mask']
    
    # Build image channels (excluding mask from channel stack)
    channels_to_stack = []
    if 'rgb' in config.CHANNELS_TO_USE: channels_to_stack.append(img)
    if 'gray' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[...,np.newaxis])
    if 'hsv' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    if 'lab' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    
    combined_image = np.dstack(channels_to_stack) if channels_to_stack else img
    
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

class PairedIrisDataset(Dataset):
    def __init__(self,p,l,c,g=None,f=None):self.p,self.l,self.c,self.g,self.f=p,l,c,g,f
    def __len__(self):return len(self.p)
    def __getitem__(self,i):
        lp,rp=self.p[i];lab=self.l[i]
        # Process both eyes with spatial attention masking
        lt=process_single_eye(lp,lab,self.c,self.g,self.f,apply_mask_attention=True)
        rt=process_single_eye(rp,lab,self.c,self.g,self.f,apply_mask_attention=True)
        return torch.cat([lt,rt],dim=0),torch.tensor(lab,dtype=torch.float32)

# --- 4.5. Threshold Optimization Utility ---
def find_optimal_threshold(y_true, y_probs, thresholds=None):
    """Find optimal threshold that maximizes F1-score on validation data"""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.01)  # Search range for medical applications
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
    
    return best_threshold, best_f1

# --- 5. Main K-Fold Training Execution ---
if __name__ == '__main__':
    # Create data manager with proper splits
    data_manager = create_data_manager(test_size=0.2, val_size=0.2, random_state=42)
    
    # Save split info for reproducibility
    data_manager.save_split_info()
    
    # Get K-fold splits from training data only (no data leakage)
    kfold_splits = data_manager.get_kfold_splits(n_splits=CONFIG.K_FOLDS)
    fold_results = []
    
    print(f"Training on {len(data_manager.get_train_data())} patients with {CONFIG.K_FOLDS}-fold CV")
    print(f"Validation set: {len(data_manager.get_val_data())} patients")
    print(f"Test set: {len(data_manager.get_test_data())} patients (held out)")

    for fold, (train_fold, val_fold) in enumerate(kfold_splits):
        print("-" * 50); print(f"FOLD {fold + 1}/{CONFIG.K_FOLDS}"); print("-" * 50)
        
        # Convert patient data to format expected by dataset
        train_p = [(p['left_image'], p['right_image']) for p in train_fold]
        val_p = [(p['left_image'], p['right_image']) for p in val_fold]
        train_l = [p['label'] for p in train_fold]
        val_l = [p['label'] for p in val_fold]
        
        # Calculate dynamic normalization channels
        num_image_channels = calculate_input_channels([c for c in CONFIG.CHANNELS_TO_USE if c != 'mask'])
        
        train_g, final_t = get_transforms(True, num_image_channels)
        val_g, _ = get_transforms(False, num_image_channels)
        
        train_ds = PairedIrisDataset(train_p, train_l, CONFIG, g=train_g, f=final_t)
        val_ds = PairedIrisDataset(val_p, val_l, CONFIG, g=val_g, f=final_t)
        
        class_counts = np.bincount(train_l); class_weights = 1. / class_counts
        sample_weights = np.array([class_weights[i] for i in train_l])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
        
        model = SimplerCNN(input_channels=CONFIG.INPUT_CHANNELS, dropout_rate=CONFIG.DROPOUT_RATE).to(CONFIG.DEVICE)
        criterion = FocalLoss(alpha=CONFIG.FOCAL_LOSS_ALPHA, gamma=CONFIG.FOCAL_LOSS_GAMMA).to(CONFIG.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)

        # Early stopping parameters
        best_val_f1 = 0.0
        best_threshold = 0.5
        patience = 8
        patience_counter = 0
        best_model_state = None

        for epoch in range(CONFIG.NUM_EPOCHS):
            # Training phase
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG.NUM_EPOCHS} [Training]", leave=False)
            for inputs, labels in loop:
                inputs, targets = inputs.to(CONFIG.DEVICE), labels.to(CONFIG.DEVICE).float().unsqueeze(1)
                optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, targets)
                loss.backward(); optimizer.step()

            # Validation phase with optimal threshold finding
            model.eval(); val_loss=0.0
            all_probs = []; all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs,labels=inputs.to(CONFIG.DEVICE),labels.to(CONFIG.DEVICE).unsqueeze(1)
                    outputs=model(inputs);loss=criterion(outputs,labels)
                    val_loss+=loss.item()
                    
                    # Collect probabilities for threshold optimization
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_probs.extend(probs.flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
            
            val_loss /= len(val_loader)
            
            # Find optimal threshold for this epoch
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            optimal_threshold, optimal_f1 = find_optimal_threshold(all_labels, all_probs)
            
            print(f"Epoch {epoch+1} -> Val Loss:{val_loss:.4f} | Optimal Threshold:{optimal_threshold:.3f} | F1:{optimal_f1:.4f}")
            
            scheduler.step(optimal_f1)
            
            # Early stopping logic with best model preservation
            if optimal_f1 > best_val_f1:
                best_val_f1 = optimal_f1
                best_threshold = optimal_threshold
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"  -> New best F1: {best_val_f1:.4f} (threshold: {best_threshold:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  -> Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # Restore best model and save with threshold info
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            save_path = CONFIG.MODEL_SAVE_PATH.format(fold + 1)
            
            # Save model with threshold metadata
            torch.save({
                'model_state_dict': best_model_state,
                'optimal_threshold': best_threshold,
                'best_f1_score': best_val_f1,
                'fold': fold + 1
            }, save_path)
            print(f"  -> Best model for Fold {fold+1} saved (F1: {best_val_f1:.4f}, Threshold: {best_threshold:.3f})")
        
        fold_results.append({
            'fold': fold + 1,
            'best_f1': best_val_f1,
            'optimal_threshold': best_threshold
        })
        print(f"\nBest F1 Score for Fold {fold+1}: {best_val_f1:.4f} (Threshold: {best_threshold:.3f})\n")
    
    print("\n" + "="*50); print("K-FOLD CROSS-VALIDATION COMPLETE"); print("="*50)
    
    # Extract F1 scores and thresholds
    f1_scores = [result['best_f1'] for result in fold_results]
    thresholds = [result['optimal_threshold'] for result in fold_results]
    
    mean_f1 = np.mean(f1_scores); std_f1 = np.std(f1_scores)
    mean_threshold = np.mean(thresholds); std_threshold = np.std(thresholds)
    
    print(f"Individual Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: F1={result['best_f1']:.4f}, Threshold={result['optimal_threshold']:.3f}")
    
    print(f"\nSummary Statistics:")
    print(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Average Optimal Threshold: {mean_threshold:.3f} ± {std_threshold:.3f}")
    
    # Save cross-validation results for reproducibility
    cv_results = {
        'fold_results': fold_results,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_threshold': mean_threshold,
        'std_threshold': std_threshold,
        'config': {
            'learning_rate': CONFIG.LEARNING_RATE,
            'batch_size': CONFIG.BATCH_SIZE,
            'channels_used': CONFIG.CHANNELS_TO_USE,
            'input_channels': CONFIG.INPUT_CHANNELS
        }
    }
    
    with open('results/cross_validation_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"\n✅ Training complete with academic rigor:")
    print(f"   - No test set contamination")
    print(f"   - Reproducible seeds set")
    print(f"   - Dynamic threshold optimization")
    print(f"   - Early stopping implemented")
    print(f"   - Spatial attention masking")
    print(f"   - Results saved for validation")