import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from UNetForward import UNet
from NuScenesBevDataset import NuScenesBevDataset
from torch.utils.data import Subset
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torch.nn.functional as F
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    # For Mac (MPS)
    torch.mps.manual_seed(seed)

#to ensure random weights
set_seed(42)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        # target is (B, H, W), predict is (B, C, H, W)
        num_classes = predict.shape[1]
        
        # Convert target to one-hot: (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply Softmax to predictions
        predict = F.softmax(predict, dim=1)
        
        # Calculate Intersection and Union
        intersection = (predict * target_one_hot).sum(dim=(0, 2, 3))
        union = predict.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - mean dice (to minimize it)
        return 1 - dice.mean()

# 1. Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_dataset_train = NuScenesBevDataset('./processed_data/input_bev', './processed_data/ground_truth', split = "train")
full_dataset_val = NuScenesBevDataset('./processed_data/input_bev', './processed_data/ground_truth', split = "val")

TOTAL_SAMPLES = 404
TRAIN_SPLIT = int(0.8 * TOTAL_SAMPLES)

train_indices = list(range(0, TRAIN_SPLIT))
val_indices = list(range(TRAIN_SPLIT, TOTAL_SAMPLES))

train_subset = Subset(full_dataset_train, train_indices)
val_subset = Subset(full_dataset_val, val_indices)

# Batch size 4 is safe for Mac memory; shuffle=True is vital for learning
train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)

# 2. Initialize
model = UNet(n_channels=4, n_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


#The Scheduler
# It monitors 'avg_val' loss. If it doesn't improve for 2 epochs (patience), 
# it cuts the learning rate by half (factor=0.5).
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)




# Weighted Loss: Tell the model that missing a car (1) is 5x worse 
# than missing empty road (0). This stops the model from just predicting "all black".
weights = torch.tensor([1.0, 5.0, 15.0, 10.0]).to(device)
ce_criterion = nn.CrossEntropyLoss(weight=weights)
dice_criterion = DiceLoss()


# 3. Training Loop
def train_model(epochs=3):
    history = []
    checkpoint_path = "best_loss4.txt"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            best_v_loss = float(f.read())
            print(f"Resuming training. Current record to beat: {best_v_loss:.4f}")
    else:
        best_v_loss = float('inf')
    
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc = f"Epoch {epoch+1}/{epochs} [Train]")

        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()      # Clear old gradients
            output = model(data)       # Forward pass
            loss = ce_criterion(output, target) + dice_criterion(output, target) # Calculate error
            loss.backward()            # Backpropagation (Chain Rule)
            optimizer.step()           # Update weights
            
            current_loss = loss.item()
            total_train_loss += current_loss
            train_pbar.set_postfix(loss=f"{current_loss:.4f}")
            
        #print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    
    #validation phase(per epoch)

        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]" , leave = False)

        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                v_loss = ce_criterion(val_output, val_target) + dice_criterion(val_output, val_target)

                total_val_loss += v_loss.item()
                val_pbar.set_postfix(v_loss=f"{v_loss.item():.4f}")
            
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        #tell the scheduler the current validation loss
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | LR: {current_lr:.6f}")
        
        
        history.append({
            'epoch': epoch+1,
            'train_loss': avg_train,
            'val_loss': avg_val,
            'lr': optimizer.param_groups[0]['lr']
        })


        # Save the learned weights if better than the previous one
        if avg_val < best_v_loss:
            best_v_loss = avg_val
            
            # Save the Weights
            torch.save(model.state_dict(), "unet_v4_weights.pth")
            
            # Save the Loss Record to a text file
            with open(checkpoint_path, "w") as f:
                f.write(str(best_v_loss))
                
            print(f"--> New Best Model Saved! (Val Loss: {best_v_loss:.4f})")
    
    pd.DataFrame(history).to_csv("training_log_4.csv", index=False)
    print("Training log saved to training_log_4.csv")

if __name__ == "__main__":
    train_model(epochs=15)