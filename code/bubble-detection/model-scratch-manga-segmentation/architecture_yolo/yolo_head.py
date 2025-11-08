import torch.nn as nn
from .blocks import Conv

class YSNHead(nn.Module):
    """Decoupled Head of YSN v11-style to predict box, class, and coefficient mask"""
    def __init__(self, num_classes=1, num_prototypes=32, in_channels_list=(64, 128, 256), reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.reg_max = reg_max

        self.box_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.mask_coef_preds = nn.ModuleList()

        for c in in_channels_list:
            # Box prediction head: output 4 * reg_max channels
            self.box_preds.append(nn.Sequential(Conv(c, c, 3), nn.Conv2d(c, 4 * self.reg_max, 1)))
            # Class prediction head
            self.cls_preds.append(nn.Sequential(Conv(c, c // 2, 3), nn.Conv2d(c // 2, self.num_classes, 1)))
            # Mask coefficient prediction head
            self.mask_coef_preds.append(nn.Sequential(Conv(c, c // 2, 3), nn.Conv2d(c // 2, self.num_prototypes, 1)))

    def forward(self, features):
        box_out, cls_out, mask_coef_out = [], [], []
        for i, x in enumerate(features):
            box_dist = self.box_preds[i](x)
            box_out.append(box_dist)
            cls_out.append(self.cls_preds[i](x))
            mask_coef_out.append(self.mask_coef_preds[i](x))
        return box_out, cls_out, mask_coef_out