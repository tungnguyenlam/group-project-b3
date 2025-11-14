import torch
import torch.nn as nn
from .blocks import Conv

class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Module):
    """C3k module, which is a stack of Bottlenecks."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.m(x)


class C3k2(nn.Module):
    """C3k2 block, with 2 convolutions by default (YOLOv11 version)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1, 1)

        if c3k:
            self.m = C3k(c_, c_, n, shortcut, g, e=1.0)
        else:
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))
        
        self.cv2 = Conv(2 * c_, c2, 1) 

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = y.chunk(2, 1)
        processed_y2 = self.m(y2)
        concatenated = torch.cat((y1, processed_y2), 1)
     
        return self.cv2(concatenated)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv11"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class PSA(nn.Module):
    """Partial Spatial Attention module for C2PSA"""
    def __init__(self, c, attn_ratio=0.5, num_heads=4):
        super().__init__()
        self.attn_ratio = attn_ratio
        self.num_heads = num_heads
        self.c = c
        
        # Split channels for attention and feedforward
        self.c_attn = int(c * attn_ratio)
        self.c_ff = c - self.c_attn
        
        if self.c_attn > 0:
            # Multi-head attention components
            self.qkv = Conv(self.c_attn, self.c_attn * 3, 1, 1)
            self.proj = Conv(self.c_attn, self.c_attn, 1, 1)
            self.head_dim = self.c_attn // num_heads
        
        if self.c_ff > 0:
            # Feedforward network
            self.ffn = nn.Sequential(
                Conv(self.c_ff, self.c_ff * 2, 1, 1),
                Conv(self.c_ff * 2, self.c_ff, 1, 1)
            )

    def forward(self, x):
        # Split input
        if self.c_attn > 0 and self.c_ff > 0:
            x_attn, x_ff = torch.split(x, [self.c_attn, self.c_ff], dim=1)
        elif self.c_attn > 0:
            x_attn = x
            x_ff = None
        else:
            x_attn = None
            x_ff = x
        
        # Attention branch
        if x_attn is not None:
            B, C, H, W = x_attn.shape
            
            # Generate Q, K, V
            qkv = self.qkv(x_attn)
            q, k, v = torch.chunk(qkv, 3, dim=1)
            
            # Reshape for multi-head attention
            q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
            k = k.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
            v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
            
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            
            # Apply attention to values
            out = (attn @ v).transpose(2, 3).reshape(B, self.c_attn, H, W)
            x_attn = x_attn + self.proj(out)
        
        # Feedforward branch
        if x_ff is not None:
            x_ff = x_ff + self.ffn(x_ff)
        
        # Concatenate results
        if x_attn is not None and x_ff is not None:
            return torch.cat([x_attn, x_ff], dim=1)
        elif x_attn is not None:
            return x_attn
        else:
            return x_ff


class C2PSA(nn.Module):
    """C2PSA block with Partial Spatial Attention for YOLOv11"""
    def __init__(self, c1, c2, n=1, e=0.5, attn_ratio=0.5, num_heads=4):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        
        # Stack of PSA modules
        self.m = nn.Sequential(*(PSA(c_, attn_ratio, num_heads) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

