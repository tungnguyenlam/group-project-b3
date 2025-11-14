# All data augmentation and preprocessing functions

import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as T

class Compose:
    """A simple version of torchvision.transforms.Compose"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class ToTensor:
    """Converts a PIL Image or numpy.ndarray to tensor"""
    def __call__(self, image, target):
        # Convert image from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        image = F.to_tensor(image) 

        # Make sure all target fields are tensors
        for k, v in target.items():
            if not isinstance(v, torch.Tensor):
                # Convert numpy array or list to tensor
                if isinstance(v, np.ndarray):
                    target[k] = torch.from_numpy(v)
                elif isinstance(v, list):
                     target[k] = torch.tensor(v)

        return image, target
    
class Normalize:
    """Normalizes a tensor image with mean and standard deviation"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Resize:
    """
    Resize images and targets (boxes, masks) while maintaining aspect ratio
    """
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        # Cache scale tensor to avoid recreation
        self._scale_tensor = None

    def get_size(self, image_size):
        w, h = image_size
        min_side = min(w, h)
        max_side = max(w, h)

        scale = self.min_size / min_side
        
        if round(scale * max_side) > self.max_size:
            scale = self.max_size / max_side
        
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return (new_w, new_h), scale

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        
        (new_w, new_h), scale = self.get_size((orig_w, orig_h))

        # Resize image
        image = F.resize(image, (new_h, new_w))

        # Scale bounding boxes
        if "boxes" in target and target["boxes"].numel() > 0:
            # Use cached scale tensor for efficiency
            if self._scale_tensor is None or self._scale_tensor.device != target["boxes"].device:
                self._scale_tensor = torch.tensor([scale, scale, scale, scale], 
                                                  dtype=torch.float32,
                                                  device=target["boxes"].device)
            else:
                self._scale_tensor.fill_(scale)
            
            target["boxes"] = target["boxes"] * self._scale_tensor

        # Resize masks with NEAREST interpolation to preserve binary values
        if "masks" in target and target["masks"].numel() > 0:
            masks = target["masks"]
            
            # More efficient: process all masks at once
            # masks shape: [N, H, W]
            masks_4d = masks.unsqueeze(1).float()  # [N, 1, H, W]
            
            # Resize all masks in one operation
            resized_masks = torch.nn.functional.interpolate(
                masks_4d,
                size=(new_h, new_w),
                mode='nearest'
            )
            
            # Remove channel dimension and convert back to uint8
            target["masks"] = resized_masks.squeeze(1).to(torch.uint8)

        target["size"] = torch.tensor([new_w, new_h])
        
        return image, target
        

class RandomHorizontalFlip:
    """
    Randomly flip image and targets horizontally
    IMPORTANT: Must flip both boxes and masks!
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            # Flip image
            image = F.hflip(image)
            
            # Flip boxes
            if "boxes" in target and target["boxes"].numel() > 0:
                boxes = target["boxes"]
                width = image.shape[-1] if isinstance(image, torch.Tensor) else image.size[0]
                
                # Flip x coordinates: x_new = width - x_old
                boxes_flipped = boxes.clone()
                boxes_flipped[:, 0] = width - boxes[:, 2]  # x1 = width - x2
                boxes_flipped[:, 2] = width - boxes[:, 0]  # x2 = width - x1
                target["boxes"] = boxes_flipped
            
            # Flip masks
            if "masks" in target and target["masks"].numel() > 0:
                # Flip along width dimension (last dimension)
                target["masks"] = torch.flip(target["masks"], dims=[-1])
        
        return image, target


class ColorJitter:
    """Apply color jittering (brightness, contrast) to image only"""
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, image, target):
        # Only apply to image, not masks!
        if isinstance(image, torch.Tensor):
            # Convert to PIL for ColorJitter
            image = F.to_pil_image(image)
            image = self.jitter(image)
            image = F.to_tensor(image)
        else:
            image = self.jitter(image)
        
        return image, target