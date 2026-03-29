import torch
from UNet_MobileV3 import MobileUNet

import torch.onnx as onnx

def export():
    # 1. Load the model
    device = torch.device("cpu") # Exporting on CPU is safer for compatibility
    model = MobileUNet(n_classes=4)
    model.load_state_dict(torch.load("unet_mobile_v1_weights.pth", map_location=device))
    model.eval()

    # 2. Create Dummy Input (B=1, C=4, H=400, W=400)
    dummy_input = torch.randn(1, 4, 400, 400)

    # 3. Export
    onnx_file = "unet_mobile_v1_weights.onnx"
    print(f"Exporting model to {onnx_file}...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file,
        export_params=True,        # Store trained weights inside the file
        opset_version=12,          # Standard version for most robotics hardware
        do_constant_folding=True,  # Optimizes the graph for speed
        input_names=['input'],     # Name the input for later use
        output_names=['output'],   # Name the output
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Allow variable batch size
    )
    print("Export Complete!")

if __name__ == "__main__":
    export()