import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=4):
        super(UNet, self).__init__()
        # Encoder (Downsampling)
        self.inc = UNetBlock(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), UNetBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), UNetBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), UNetBlock(256, 512))
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = UNetBlock(256, 128) # 256 because of concatenation
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = UNetBlock(128, 64)
        
        # Final Output Layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Forward through Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Forward through Decoder with Skip Connections
        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1) # The "U" Skip Connection
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv_up2(u2)
        
        return self.outc(u2)