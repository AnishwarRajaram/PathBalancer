import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torch.nn.functional as F
import random
import numpy as np

# Import your new architecture
from UNet_MobileV3 import MobileUNet 
from NuScenesBevDataset import NuScenesBevDataset

# --- CONFIGURATION ---
VERSION_TAG = "mobile_v1"
MODEL_SAVE_PATH = f"unet_{VERSION_TAG}_weights.pth"
LOG_CSV_PATH = f"training_log_{VERSION_TAG}.csv"
CHECKPOINT_PATH = f"best_loss_{VERSION_TAG}.txt"
BATCH_SIZE = 8  # MobileNet is lighter, so you can likely double your batch size
EPOCHS = 30     # Recommended more epochs for fine-tuning

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    torch.mps.manual_seed(seed)

set_seed(42)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        num_classes = predict.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        predict = F.softmax(predict, dim=1)
        intersection = (predict * target_one_hot).sum(dim=(0, 2, 3))
        union = predict.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# 1. Setup Data
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_dataset_train = NuScenesBevDataset('./processed_data/input_bev', './processed_data/ground_truth', split="train")
full_dataset_val = NuScenesBevDataset('./processed_data/input_bev', './processed_data/ground_truth', split="val")

TOTAL_SAMPLES = 404
TRAIN_SPLIT = int(0.8 * TOTAL_SAMPLES)
train_indices = list(range(0, TRAIN_SPLIT))
val_indices = list(range(TRAIN_SPLIT, TOTAL_SAMPLES))

train_loader = DataLoader(Subset(full_dataset_train, train_indices), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(full_dataset_val, val_indices), batch_size=BATCH_SIZE, shuffle=False)

# 2. Initialize Model & Optimizer
# We use MobileUNet instead of the standard UNet
model = MobileUNet(n_classes=4).to(device)

# AdamW is often preferred for pre-trained backbones
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

weights = torch.tensor([1.0, 5.0, 15.0, 10.0]).to(device)
ce_criterion = nn.CrossEntropyLoss(weight=weights)
dice_criterion = DiceLoss()

# 3. Training Loop
def train_model(epochs=EPOCHS):
    history = []
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            best_v_loss = float(f.read())
            print(f"Resuming {VERSION_TAG}. Record to beat: {best_v_loss:.4f}")
    else:
        best_v_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = ce_criterion(output, target) + dice_criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                v_loss = ce_criterion(val_output, val_target) + dice_criterion(val_output, val_target)
                total_val_loss += v_loss.item()
            
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {current_lr:.6f}")
        
        history.append({'epoch': epoch+1, 'train_loss': avg_train, 'val_loss': avg_val, 'lr': current_lr})

        if avg_val < best_v_loss:
            best_v_loss = avg_val
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            with open(CHECKPOINT_PATH, "w") as f:
                f.write(str(best_v_loss))
            print(f"--> {VERSION_TAG} Saved! (Val Loss: {best_v_loss:.4f})")
    
    pd.DataFrame(history).to_csv(LOG_CSV_PATH, index=False)
    print(f"Results saved to {LOG_CSV_PATH}")

if __name__ == "__main__":
    train_model()