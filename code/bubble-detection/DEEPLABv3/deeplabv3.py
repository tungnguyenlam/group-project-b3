import os
import cv2
import PIL 
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

# A function to replace all BatchNorm2d layers with GroupNorm layers
def replace_bn_with_gn(module):
    # iterate through all layers of DeepLabv3 model
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            setattr(module, name, nn.GroupNorm(32, num_features))
        else:
            # if not BatchNorm2d, recursively apply to child modules
            replace_bn_with_gn(child)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model= deeplabv3_resnet50(weights=None, num_classes=1)
replace_bn_with_gn(model)
