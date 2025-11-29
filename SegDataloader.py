import torch
from torch.utils.data import Dataset, DataLoader
from SegDataset import MangaBubbleDataset
from torchvision import transforms
import os
import json
import numpy as np
from PIL import Image
import cv2
from pycocotools import mask as mask_utils

def manga_collate_fn(batch):
    images = []
    masks = []
    bboxes = []

    for img, mask, box in batch:
        images.append(img)
        masks.append(mask)
        bboxes.append(box)

    images = torch.stack(images, dim=0)
    return images, masks, bboxes


# --- DataLoader class ---
class MangaBubbleDataLoader:
    def __init__(self, json_file, img_dir, img_size, batch_size=4, shuffle=False, num_workers=2, transform=None):
        self.dataset = MangaBubbleDataset(json_file, img_dir, img_size, transform)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=manga_collate_fn
        )


# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    loader_class = MangaBubbleDataLoader(
        json_file="path/to/annotations.json",
        img_dir="path/to/images",
        img_size=(256, 256),
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    dataloader = loader_class.get_loader()

    for imgs, masks, bboxes in dataloader:
        print(imgs.shape)    # [batch_size, 3, 256, 256]
        print(len(masks))    # batch_size
        print(len(bboxes))   # batch_size
        break