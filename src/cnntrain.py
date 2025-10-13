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
    DIABETIC_IMAGE_DIR='dataset/diabetic/'; CONTROL_IMAGE_DIR='dataset/control/'
    DIABETIC_PANCREAS_MASK_DIR='dataset/pancreas_masks_for_training/diabetic/'; CONTROL_PANCREAS_MASK_DIR='dataset/pancreas_masks_for_training/control/'
    
    MODEL_SAVE_PATH = 'best_f1_model_fold_{}.pth' 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE=128; BATCH_SIZE=4; LEARNING_RATE=2e-5
    NUM_EPOCHS=75 
    
    K_FOLDS = 5 

    # --- MODIFIED: Using all available image channels plus the mask ---
    CHANNELS_TO_USE = ['rgb', 'gray', 'hsv', 'lab', 'mask'] 
    
    # These are calculated automatically based on the list above
    INPUT_CHANNELS = calculate_input_channels(CHANNELS_TO_USE) * 2 
    
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

# --- 4. Data Handling ---
def get_transforms(is_train=True):
    geo_transforms = [A.Resize(Config.IMG_SIZE, Config.IMG_SIZE)]
    # Note: Max pixel value is 255.0, and the number of channels for normalization (10) is a high-end estimate that covers all possible image channels but not the mask. This is fine.
    final_transform = A.Compose([A.Normalize(mean=[0.5]*10, std=[0.5]*10, max_pixel_value=255.0), ToTensorV2()])
    
    if is_train:
        geo_transforms.extend([
            A.HorizontalFlip(p=0.5), A.Rotate(limit=30, p=0.8),
            A.RandomBrightnessContrast(p=0.8), A.GaussNoise(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5)
        ])
    return A.Compose(geo_transforms), final_transform

def process_single_eye(image_path, label, config, geo_transform, final_transform):
    img=cv2.imread(image_path);img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    b,_=os.path.splitext(os.path.basename(image_path));mf=f"{b}_pancreas_roi.png"
    md=config.DIABETIC_PANCREAS_MASK_DIR if label==1 else config.CONTROL_PANCREAS_MASK_DIR
    mp=os.path.join(md,mf);mask=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
    if mask is None: mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if geo_transform: aug=geo_transform(image=img,mask=mask);img,mask=aug['image'],aug['mask']
    
    channels_to_stack = []
    if 'rgb' in config.CHANNELS_TO_USE: channels_to_stack.append(img)
    if 'gray' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[...,np.newaxis])
    if 'hsv' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    if 'lab' in config.CHANNELS_TO_USE: channels_to_stack.append(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    
    combined_image = np.dstack(channels_to_stack) if channels_to_stack else img
    
    if 'mask' in config.CHANNELS_TO_USE:
        final_data = final_transform(image=combined_image)
        image_tensor = final_data['image']
        mask_tensor = ToTensorV2()(image=mask)['image'].float() / 255.0
        return torch.cat([image_tensor, mask_tensor], dim=0)
    else:
        return final_transform(image=combined_image)['image']

class PairedIrisDataset(Dataset):
    def __init__(self,p,l,c,g=None,f=None):self.p,self.l,self.c,self.g,self.f=p,l,c,g,f
    def __len__(self):return len(self.p)
    def __getitem__(self,i):lp,rp=self.p[i];lab=self.l[i];lt=process_single_eye(lp,lab,self.c,self.g,self.f);rt=process_single_eye(rp,lab,self.c,self.g,self.f);return torch.cat([lt,rt],dim=0),torch.tensor(lab,dtype=torch.float32)

# --- 5. Main K-Fold Training Execution ---
if __name__ == '__main__':
    all_p, all_l = [], []
    d, c = CONFIG.DIABETIC_IMAGE_DIR, CONFIG.CONTROL_IMAGE_DIR
    for dr, lab in [(d, 1), (c, 0)]:
        pf = defaultdict(lambda: {'L': None, 'R': None})
        for ip in glob.glob(os.path.join(dr, '*.jpg')):
            fn = os.path.basename(ip); m = re.search(r'(L|R)', fn, re.I)
            if m: pf[fn[:m.start()]][m.group(0).upper()] = ip
        for i, e in pf.items():
            if e['L'] and e['R']: all_p.append((e['L'], e['R'])); all_l.append(lab)
    
    all_p_np = np.array(all_p); all_l_np = np.array(all_l)

    print(f"Found {np.bincount(all_l_np)[0]} Control vs. {np.bincount(all_l_np)[1]} Diabetic pairs.")

    kfold = StratifiedKFold(n_splits=CONFIG.K_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_p_np, all_l_np)):
        print("-" * 50); print(f"FOLD {fold + 1}/{CONFIG.K_FOLDS}"); print("-" * 50)

        train_p, val_p = all_p_np[train_idx], all_p_np[val_idx]
        train_l, val_l = all_l_np[train_idx], all_l_np[val_idx]
        
        train_g, final_t = get_transforms(True); val_g, _ = get_transforms(False)
        
        train_ds = PairedIrisDataset(train_p.tolist(), train_l.tolist(), CONFIG, g=train_g, f=final_t)
        val_ds = PairedIrisDataset(val_p.tolist(), val_l.tolist(), CONFIG, g=val_g, f=final_t)
        
        class_counts = np.bincount(train_l); class_weights = 1. / class_counts
        sample_weights = np.array([class_weights[i] for i in train_l])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
        
        model = SimplerCNN(input_channels=CONFIG.INPUT_CHANNELS, dropout_rate=CONFIG.DROPOUT_RATE).to(CONFIG.DEVICE)
        criterion = FocalLoss(alpha=CONFIG.FOCAL_LOSS_ALPHA, gamma=CONFIG.FOCAL_LOSS_GAMMA).to(CONFIG.DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10)

        metrics = {"Acc":torchmetrics.Accuracy(task="binary").to(CONFIG.DEVICE),"F1":torchmetrics.F1Score(task="binary").to(CONFIG.DEVICE),"P":torchmetrics.Precision(task="binary").to(CONFIG.DEVICE),"R":torchmetrics.Recall(task="binary").to(CONFIG.DEVICE)}
        best_val_f1 = 0.0

        for epoch in range(CONFIG.NUM_EPOCHS):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG.NUM_EPOCHS} [Training]", leave=False)
            for inputs, labels in loop:
                inputs, targets = inputs.to(CONFIG.DEVICE), labels.to(CONFIG.DEVICE).float().unsqueeze(1)
                optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, targets)
                loss.backward(); optimizer.step()

            model.eval(); val_loss=0.0
            for m in metrics.values(): m.reset()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs,labels=inputs.to(CONFIG.DEVICE),labels.to(CONFIG.DEVICE).unsqueeze(1)
                    outputs=model(inputs);loss=criterion(outputs,labels)
                    val_loss+=loss.item();preds=(torch.sigmoid(outputs)>0.5).int()
                    for m in metrics.values(): m.update(preds,labels.data.int())
            
            val_loss /= len(val_loader); results={name:m.compute().item() for name,m in metrics.items()}
            print(f"Epoch {epoch+1} -> Val Loss:{val_loss:.4f} | " + " | ".join([f"{n}:{v:.4f}" for n,v in results.items()]))
            
            scheduler.step(results['F1'])
            
            if results['F1'] > best_val_f1:
                best_val_f1 = results['F1']
                save_path = CONFIG.MODEL_SAVE_PATH.format(fold + 1)
                torch.save(model.state_dict(), save_path)
                print(f"  -> New best model for Fold {fold+1} saved (F1 Score: {best_val_f1:.4f})")
        
        fold_results.append(best_val_f1)
        print(f"\nBest F1 Score for Fold {fold+1}: {best_val_f1:.4f}\n")
    
    print("\n" + "="*50); print("K-FOLD CROSS-VALIDATION COMPLETE"); print("="*50)
    mean_f1 = np.mean(fold_results); std_f1 = np.std(fold_results)
    print(f"Individual Fold F1 Scores: {[f'{score:.4f}' for score in fold_results]}")
    print(f"Average F1 Score: {mean_f1:.4f}"); print(f"Standard Deviation of F1 Scores: {std_f1:.4f}")