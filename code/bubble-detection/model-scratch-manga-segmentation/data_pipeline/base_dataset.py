# An abstract base class for datasets

from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseVisionDataset(Dataset, ABC):
    def __init__(self, transforms=None):
        self.transforms = transforms

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def get_height_and_width(self, idx):
        """A helper to get image dimensions without loading the image"""
        pass