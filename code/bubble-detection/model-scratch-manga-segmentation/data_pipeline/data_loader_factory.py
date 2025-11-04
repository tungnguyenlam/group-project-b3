import torch
from .manga_dataset import MangaDataset
from . import transforms as T

def collate_fn(batch):
    """
    Custom collate function that:
    1. Pads images to same size
    2. Keeps targets as list (don't stack - different number of objects)
    """
    images, targets = tuple(zip(*batch))

    # Find max dimensions in the batch
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    # Pad all images to max dimensions
    padded_images = []
    for img in images:
        # img shape: [C, H, W]
        pad_bottom = max_height - img.shape[1]
        pad_right = max_width - img.shape[2]

        # Pad: (left, right, top, bottom) for last 2 dimensions
        padded_img = torch.nn.functional.pad(
            img, 
            (0, pad_right, 0, pad_bottom), 
            mode='constant', 
            value=0  # Pad with black (0) for images
        )
        padded_images.append(padded_img)

    # Stack images into batch tensor [B, C, H, W]
    images = torch.stack(padded_images, dim=0)

    return images, targets

def get_transforms(is_train):
    transform_list = []
    transform_list.append(T.Resize(min_size=640, max_size=1024))
    # Data augmentation for training only
    if is_train:
        transform_list.append(T.RandomHorizontalFlip(prob=0.5))
        transform_list.append(T.ColorJitter(brightness=0.2, contrast=0.2))

    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(mean=[0.5], std=[0.5]))
    return T.Compose(transform_list)

class DataLoaderFactory:
    def __init__(self, json_dir, image_root):
        self.json_dir = json_dir
        self.image_root = image_root

    def create_data_loader(self, split='train', batch_size=2, shuffle=True, num_workers=8, transforms=None):
        is_train = (split == 'train')

        if transforms is None:
            transforms = get_transforms(is_train=is_train)

        dataset = MangaDataset(
            json_dir=self.json_dir,
            image_root=self.image_root,
            split=split, 
            transforms=get_transforms(is_train=is_train)
        )
    
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
            
        return data_loader