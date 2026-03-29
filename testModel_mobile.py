import torch
import numpy as np
from pathlib import Path
from UNet_MobileV3 import MobileUNet
from NuScenesBevDataset import NuScenesBevDataset
import check_my_tensors as check  # This uses your existing visualization logic
import os
import time

# --- CONFIGURATION ---
MODEL_WEIGHTS = "unet_mobile_v1_weights.pth"
INPUT_DIR = './processed_data/input_bev'
GT_DIR = './processed_data/ground_truth'

# Set these to choose which samples to test
# Enumerated

TOTAL_SAMPLES = 404
TRAIN_END = int(0.8 * TOTAL_SAMPLES)

def calculate_iou(preds, labels, n_classes=4):
    """
    Calculates IoU for each class.
    preds, labels: (Height, Width) arrays of class IDs
    """
    ious = []
    # We flatten the arrays to make comparison easier
    preds = preds.flatten()
    labels = labels.flatten()

    for cls in range(n_classes):
        intersection = np.logical_and(preds == cls, labels == cls).sum()
        union = np.logical_or(preds == cls, labels == cls).sum()
        
        if union == 0:
            # If the class isn't present in ground truth or prediction, skip it
            ious.append(float('nan')) 
        else:
            ious.append(intersection / union)
            
    return ious

def run_test():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Dataset
    # This automatically finds all .npy files you just preprocessed
    dataset = NuScenesBevDataset(INPUT_DIR, GT_DIR)
    
    if len(dataset) < TOTAL_SAMPLES:
        print(f"Warning: Dataset only has {len(dataset)} samples. Adjusting END_IDX.")
        actual_end = len(dataset)
    else:
        actual_end = TOTAL_SAMPLES

    # 3. Initialize and Load Model
    model = MobileUNet(n_classes=4).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print("Successfully loaded trained weights.")
    except FileNotFoundError:
        print("Error: .pth file not found. Did you run train_model() yet?")
        return

    model.eval() # CRITICAL: Sets model to inference mode

    # 4. Loop through the specified range
    cumulative_loss = 0
    all_ious = []
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 15.0, 10.0]).to(device))

    print(f"Calculating Cumulative Loss for indices {TRAIN_END} to {TOTAL_SAMPLES-1}...")

    latencies = []
    
    with torch.no_grad():
        for i in range(TRAIN_END, TOTAL_SAMPLES):
            x_tensor, y_tensor = dataset[i]
            x_tensor = x_tensor.unsqueeze(0).to(device)
            y_tensor = y_tensor.unsqueeze(0).to(device)
            
            start_time = time.perf_counter()

            output = model(x_tensor)

            if device.type == 'mps':
                torch.mps.synchronize()

            end_time = time.perf_counter()

            latencies.append(end_time - start_time)
            loss = criterion(output, y_tensor)
            cumulative_loss += loss.item()

            pred_ids = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            gt_ids = y_tensor.squeeze().cpu().numpy()

            current_iou = calculate_iou(pred_ids, gt_ids)
            all_ious.append(current_iou)

            # Optional: Keep visualization for ONLY the first test sample to spot-check
            if i == TRAIN_END:
                pred_ids = torch.argmax(output, dim=1)
                token = dataset.input_files[i].name.replace("_x.npy", "")
                check.visualize_processed_data(pred_ids, token)
            
    mean_per_class = np.nanmean(all_ious,axis=0)
    overall_miou = np.nanmean(mean_per_class)
    
    avg_test_loss = cumulative_loss / (TOTAL_SAMPLES - TRAIN_END)
    print(f"\n--- FINAL TEST RESULTS ---")
    print(f"Average Cumulative Loss over 50 samples: {avg_test_loss:.4f}")
    

    checkpoint_path = "best_loss.txt"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            best_v_loss = float(f.read())
            print(f"Recorded Best Value loss: {best_v_loss:.4f}")
    else:
        best_v_loss = float('inf')
    print(f"best value loss------>{best_v_loss}\n")

    print("\n--- mIoU TEST RESULTS ---")
    classes = ["Background", "Vehicle", "Pedestrian", "Obstacle"]
    for idx, name in enumerate(classes):
        print(f"{name} IoU: {mean_per_class[idx]:.4f}")
    
    print(f"-------------------------")
    print(f"Final mIoU Score: {overall_miou:.4f}")

    # 5. Calculate Average FPS
    avg_latency = np.mean(latencies)
    fps = 1.0 / avg_latency
    
    print(f"\n--- SPEED PERFORMANCE ---")
    print(f"Average Inference Time: {avg_latency*1000:.2f} ms")
    print(f"Effective Throughput: {fps:.2f} FPS")
    print(f"-------------------------")

if __name__ == "__main__":
    run_test()