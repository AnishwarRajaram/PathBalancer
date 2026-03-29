import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileUNet(nn.Module):
    def __init__(self, n_classes=4):
        super(MobileUNet, self).__init__()
        
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        
        # Patch first layer for 4 channels
        old_conv = backbone[0][0]
        new_conv = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3, :, :] = torch.mean(old_conv.weight, dim=1)
        backbone[0][0] = new_conv

        # Encoder Stages
        self.stage1 = backbone[0:2]   # Out: 16
        self.stage2 = backbone[2:4]   # Out: 24
        self.stage3 = backbone[4:9]   # Out: 48 (This was likely the 88 channel culprit)
        self.stage4 = backbone[9:13]  # Out: 576 (Bottleneck)
        
        # Decoder - We use flexible convolutions to handle the concat
        # Up 1: 576 (bottleneck) -> 48 (stage 3)
        self.up1 = nn.ConvTranspose2d(576, 576, kernel_size=2, stride=2)
        self.conv_up1 = nn.Sequential(nn.Conv2d(576 + 48, 128, 3, padding=1), nn.ReLU())
        
        # Up 2: 128 -> 24 (stage 2)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_up2 = nn.Sequential(nn.Conv2d(128 + 24, 64, 3, padding=1), nn.ReLU())
        
        # Up 3: 64 -> 16 (stage 1)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up3 = nn.Sequential(nn.Conv2d(64 + 16, 32, 3, padding=1), nn.ReLU())
        
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.stage1(x)  
        x2 = self.stage2(x1) 
        x3 = self.stage3(x2) 
        x4 = self.stage4(x3) 
        
        # Decoder
        u1 = self.up1(x4)
        u1 = F.interpolate(u1, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=True)
        u1 = torch.cat([u1, x3], dim=1) # The 88 channel concat happens here
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = F.interpolate(u2, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=True)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = F.interpolate(u3, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)
        
        output = self.final_up(u3)
        output = F.interpolate(output, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        return self.outc(output)