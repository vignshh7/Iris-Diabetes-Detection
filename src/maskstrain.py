import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm


class Config:
    DATA_DIR = 'dataset'
    IMAGE_DIR = os.path.join(DATA_DIR, 'imagesformaskstraining')
    MASK_DIR = os.path.join(DATA_DIR, 'masks')
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    ENCODER = 'mobilenet_v2'
    ENCODER_WEIGHTS = 'imagenet'
    IMG_SIZE = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    VALIDATION_SPLIT = 0.15

CONFIG = Config()

# --- 2. Custom Dataset Class (CORRECTED) ---
class IrisDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames
        self.transform = transform

        self.image_paths = []
        self.mask_paths = []
        
        print("--- Checking for image/mask pairs... ---")
        
        # *** THE MAIN FIX IS HERE: Looping over 'image_filenames' ***
        for fname in self.image_filenames:
            base_name, ext = os.path.splitext(fname)
            mask_fname = f"{base_name}_mask.png"
            
            img_path = os.path.join(self.image_dir, fname)
            mask_path = os.path.join(self.mask_dir, mask_fname)

            # Debug print to help find path issues
            # print(f"Checking for IMG: {img_path} | MASK: {mask_path}")

            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                # This will tell you if a specific pair is not found
                print(f"    -> FAILED to find pair for: {fname} (looked for mask: {mask_fname})")

        print(f"--- Found {len(self.image_paths)} valid pairs. ---")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if mask is not None:
            # Convert 0-255 mask to 0-1 class indices for binary segmentation
            mask = (mask > 128).astype(np.uint8)
        else:
            print(f"Warning: Could not read mask for {self.image_paths[idx]}. Creating a blank mask.")
            mask = np.zeros((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

# --- 3. Data Augmentation ---
def get_train_transforms():
    return A.Compose([
        A.Resize(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE),
        A.Rotate(limit=35, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# --- 4. Training & Validation Functions ---
def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=CONFIG.DEVICE)
        targets = targets.to(device=CONFIG.DEVICE).float().unsqueeze(1)
        
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def val_fn(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc="Validating")
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=CONFIG.DEVICE)
            targets = targets.to(device=CONFIG.DEVICE).float().unsqueeze(1)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    print(f"Using device: {CONFIG.DEVICE}")

    if not os.path.exists(CONFIG.IMAGE_DIR) or not os.path.exists(CONFIG.MASK_DIR):
        print(f"Error: Image directory '{CONFIG.IMAGE_DIR}' or Mask directory '{CONFIG.MASK_DIR}' not found. Please check your folder structure and paths in Config.")
        exit()

    all_image_filenames = sorted(os.listdir(CONFIG.IMAGE_DIR))
    
    if not all_image_filenames:
        print(f"Error: No images found in {CONFIG.IMAGE_DIR}")
        exit()

    train_files, val_files = train_test_split(
        all_image_filenames,
        test_size=CONFIG.VALIDATION_SPLIT,
        random_state=42
    )
    
    print(f"Total images found: {len(all_image_filenames)}")
    print(f"Training set size: {len(train_files)}")
    print(f"Validation set size: {len(val_files)}")
    
    train_dataset = IrisDataset(CONFIG.IMAGE_DIR, CONFIG.MASK_DIR, train_files, get_train_transforms())
    val_dataset = IrisDataset(CONFIG.IMAGE_DIR, CONFIG.MASK_DIR, val_files, get_val_transforms())
    
    # Check if the datasets were populated correctly
    if len(train_dataset) == 0:
        print("\nFATAL ERROR: The training dataset is empty. No image/mask pairs were found.")
        print("Please check the 'FAILED' messages above to see why the pairs were not matched.")
        exit()
        
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model for binary segmentation: output classes = 1
    model = smp.Unet(
        encoder_name=CONFIG.ENCODER,
        encoder_weights=CONFIG.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1, 
    ).to(CONFIG.DEVICE)

    # Loss function mode for binary segmentation
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG.DEVICE == "cuda"))

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(CONFIG.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{CONFIG.NUM_EPOCHS} ---")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = val_fn(val_loader, model, loss_fn)
        
        print(f"Average Train Loss: {train_loss:.4f}")
        print(f"Average Val Loss:   {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_iris_model_3class.pth')
            print("=> Saved new best model")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved to 'best_iris_model_2class.pth'")