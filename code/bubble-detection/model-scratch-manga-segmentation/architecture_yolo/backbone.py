import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Conv
from .yolo11_blocks import C3k2, SPPF, C2PSA

class YSNBackbone(nn.Module):
    """
    Backbone of YOLO-Seg-Nano (YSN) 
    """
    def __init__(self, base_channels=16, base_depth=1):
        super().__init__()
        
        # Stem: 1 -> 16 (grayscale input)
        self.stem = Conv(1, base_channels, 3, 2)  # /2
        
        # Stage 2: 16 -> 32 -> 64 (e=0.25)
        self.stage2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),  # /4
            C3k2(base_channels * 2, base_channels * 4, n=base_depth, c3k=False, e=0.25)
        )
        
        # Stage 3: 64 -> 64 -> 128 (e=0.25)
        self.stage3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 4, 3, 2),  # /8
            C3k2(base_channels * 4, base_channels * 8, n=base_depth, c3k=False, e=0.25)
        )
        
        # Stage 4: 128 -> 128 -> 128 (e=0.5)
        self.stage4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 8, 3, 2),  # /16
            C3k2(base_channels * 8, base_channels * 8, n=base_depth, c3k=True, e=0.5)
        )
        
        # Stage 5: 128 -> 256 -> 256 (e=0.5)
        self.stage5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),  # /32
            C3k2(base_channels * 16, base_channels * 16, n=base_depth, c3k=True, e=0.5)
        )
        
        # SPPF after stage 5
        self.sppf = SPPF(base_channels * 16, base_channels * 16, k=5)
        
        # C2PSA after SPPF 
        self.c2psa = C2PSA(base_channels * 16, base_channels * 16, n=1, e=0.5, attn_ratio=0.5, num_heads=4)

    def forward(self, x):
        stem_out = self.stem(x)          # /2
        
        c2_out = self.stage2(stem_out)   # /4, 64 channels
        c3_out = self.stage3(c2_out)     # /8, 128 channels (P3)
        c4_out = self.stage4(c3_out)     # /16, 128 channels (P4)
        c5_out = self.stage5(c4_out)     # /32, 256 channels (P5)
        
        # Apply SPPF then C2PSA on P5
        c5_out = self.sppf(c5_out)
        c5_out = self.c2psa(c5_out)
        
        # Return P3, P4, P5 for neck (at /8, /16, /32 strides)
        # Channels: 128, 128, 256
        return c3_out, c4_out, c5_out


class YSNNeck(nn.Module):
    """
    Neck PANet for YOLOv11n-seg
    Fixed version with proper size matching for odd input dimensions
    """
    def __init__(self, c3_channels, c4_channels, c5_channels):
        super().__init__()
        
        # Top-down pathway
        # P5 (256) → upsample → concat with P4 (128) → 384
        self.pan_c4 = C3k2(c5_channels + c4_channels, c4_channels, n=1, c3k=False, e=0.5)
        
        # P4 (128) → upsample → concat with P3 (128) → 256
        self.pan_c3 = C3k2(c4_channels + c3_channels, c3_channels, n=1, c3k=False, e=0.5)

        # Bottom-up pathway
        # P3 (128) → downsample → concat with P4 (128) → 256
        self.down_c3 = Conv(c3_channels, c3_channels, 3, 2)
        self.pan_n4 = C3k2(c3_channels + c4_channels, c4_channels, n=1, c3k=False, e=0.5)
        
        # N4 (128) → downsample → concat with P5 (256) → 384
        self.down_c4 = Conv(c4_channels, c4_channels, 3, 2)
        self.pan_n5 = C3k2(c4_channels + c5_channels, c5_channels, n=1, c3k=True, e=0.5)

    def forward(self, features):
        c3, c4, c5 = features  # 128, 128, 256 channels at /8, /16, /32 strides
        
        # ==================== Top-down pathway ====================
        # Upsample P5 to match P4 size
        p5_up = F.interpolate(c5, size=c4.shape[2:], mode='nearest')
        p4 = self.pan_c4(torch.cat([p5_up, c4], 1))
        
        # Upsample P4 to match P3 size
        p4_up = F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p3 = self.pan_c3(torch.cat([p4_up, c3], 1))

        # ==================== Bottom-up pathway ====================
        # CRITICAL: For odd input sizes, downsampled features might not match
        
        # Downsample P3
        p3_down = self.down_c3(p3)
        
        # Resize p4 to EXACTLY match p3_down size (handles odd dimensions)
        p4_matched = F.interpolate(p4, size=p3_down.shape[2:], mode='nearest')
        n4 = self.pan_n4(torch.cat([p3_down, p4_matched], 1))
        
        # Downsample N4
        n4_down = self.down_c4(n4)
        
        # Resize c5 to EXACTLY match n4_down size (handles odd dimensions)
        c5_matched = F.interpolate(c5, size=n4_down.shape[2:], mode='nearest')
        n5 = self.pan_n5(torch.cat([n4_down, c5_matched], 1))
        
        # Return P3, N4, N5 for detection head
        # Channels: 128, 128, 256
        return p3, n4, n5