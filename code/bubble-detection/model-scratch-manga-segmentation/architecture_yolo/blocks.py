import torch.nn as nn

def autopad(k, p=None, d=1):
    """Caculate padding automatically for convolution"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution layer: Conv2d -> BatchNorm2d -> SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    

class ProtoNet(nn.Module):

    def __init__(self, in_channels, num_prototypes=32):
        super().__init__()
        c_ = 128  
        
        self.proto_convs = nn.Sequential(
            Conv(in_channels, c_, k=3),
            nn.ConvTranspose2d(c_, c_, 2, 2, bias=True),  # Learnable upsample
            Conv(c_, c_, k=3),
            Conv(c_, num_prototypes, k=1)
        )

    def forward(self, x):
        return self.proto_convs(x)