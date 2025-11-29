from manga_ocr import MangaOcr
from PIL import Image
import gc
import numpy as np
from typing import List, Union
from math import floor, ceil
import torch


class MangaBatchOCRModel:
    """
    Batch processing wrapper for MangaOCR model.
    Processes multiple bounding boxes more efficiently.
    """
    def __init__(self):
        self.mocr = None
        self.device = None

    def load_model(self, device='auto'):
        """
        Load MangaOCR model.
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        if self.mocr is None:
            if device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            
            self.device = device
            self.mocr = MangaOcr()
            
            print(f"MangaOCR model loaded on {self.device}")
        else:
            print("Model is already loaded")

    def predict(self, bboxes: List[List[float]], image: np.ndarray) -> List[str]:
        """
        Predict OCR text for each bounding box in the image.
        Processes boxes in batch for better efficiency.
        
        Args:
            bboxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
            image: numpy array of the full image (RGB format)
            
        Returns:
            List of OCR text strings, one for each bounding box
        """
        if self.mocr is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        if not bboxes or len(bboxes) == 0:
            return []
        
        # Prepare all cropped images
        cropped_images = []
        valid_indices = []
        
        for idx, box in enumerate(bboxes):
            x_min, y_min, x_max, y_max = box
            
            # Convert to integers and crop
            x_min, y_min = floor(x_min), floor(y_min)
            x_max, y_max = ceil(x_max), ceil(y_max)
            
            # Crop the image
            cropped_image = image[y_min:y_max, x_min:x_max, :]
            
            # Skip empty crops
            if cropped_image.size == 0:
                continue
            
            try:
                pil_image = Image.fromarray(cropped_image)
                cropped_images.append(pil_image)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing box {idx}: {e}")
                continue
        
        # Perform OCR on all valid crops
        text_ocr_list = [""] * len(bboxes)
        
        for img_idx, pil_image in enumerate(cropped_images):
            try:
                text = self.mocr(pil_image)
                original_idx = valid_indices[img_idx]
                text_ocr_list[original_idx] = text
            except Exception as e:
                print(f"OCR error: {e}")
                continue
        
        return text_ocr_list
        
    def unload_model(self):
        """Unload model and free memory."""
        if self.mocr is None:
            print("Model is not loaded yet")
        else:
            del self.mocr
            self.mocr = None
            
            gc.collect()
            
            if self.device and 'cuda' in str(self.device):
                torch.cuda.empty_cache()
            elif self.device and 'mps' in str(self.device):
                torch.mps.empty_cache()
            
            print("Model unloaded")
