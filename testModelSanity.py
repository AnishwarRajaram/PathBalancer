import torch 
import torch.nn as nn
import torch.nn.functional as F 
from UNetForward import UNet as model
import numpy as np
import matplotlib
import check_my_tensors as check

# Defined in your training script, not inside the UNet class
#criterion = nn.CrossEntropyLoss() 
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


sample_token = "356d81f38dd9473ba590f39e266f54e5"

def check_untrained_model(sample_token = sample_token):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    my_model = model(n_channels=3,n_classes=4).to(device)
    my_model.to(device)
    my_model.eval() # Set to evaluation mode
    with torch.no_grad(): # Disable backprop/gradient tracking
        # 1. Load one of the .npy files
        x_data = np.load(f"./processed_data/input_bev/{sample_token}_x.npy")
        
        # 2. Prepare for PyTorch: (H, W, C) -> (1, C, H, W) 
        # The '1' is the batch size
        x_tensor = torch.from_numpy(x_data).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
       
        print(f"Input shape: {x_tensor.shape}")   # Expected: torch.Size([1, 3, 200, 200])
          
                
        with torch.no_grad():
             # 3. Forward Pass
            output = my_model(x_tensor) # Shape: (1, 4, 200, 200)
            print(f"Output shape: {output.shape}") # Expected: torch.Size([1, 4, 200, 200])
            pred_ids = torch.argmax(output, dim = 1)

        # 4. Get the predicted class for each pixel
        # argmax looks at the 4 channels and picks the index of the highest score
        
    return pred_ids

sample_token = "356d81f38dd9473ba590f39e266f54e5"


if __name__ == "__main__":
    check.visualize_processed_data(check_untrained_model(sample_token),sample_token)


# Now visualize 'prediction' using the matplotlib script from earlier