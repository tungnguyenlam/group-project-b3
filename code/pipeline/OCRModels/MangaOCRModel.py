from manga_ocr import MangaOcr
from PIL import Image
import gc
import numpy as np
from typing import List
from math import floor, ceil


class MangaOCRModel:
    """
    Simple wrapper for MangaOCR model.
    Processes bounding boxes one by one (non-batch mode).
    """
    def __init__(self):
        self.mocr = None

    def load_model(self):
        """Load MangaOCR model."""
        if self.mocr is None:
            self.mocr = MangaOcr()
            print("MangaOCR model loaded")
        else:
            print("Model is already loaded")

    def predict(self, bboxes: List[List[float]], image: np.ndarray) -> List[str]:
        """
        Predict OCR text for each bounding box in the image.
        
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
        
        text_ocr_list = []
        
        # Process each bounding box
        for box in bboxes:
            x_min, y_min, x_max, y_max = box
            
            # Convert to integers and crop
            x_min, y_min = floor(x_min), floor(y_min)
            x_max, y_max = ceil(x_max), ceil(y_max)
            
            # Crop the image
            cropped_image = image[y_min:y_max, x_min:x_max, :]
            
            # Handle empty crops
            if cropped_image.size == 0:
                text_ocr_list.append("")
                continue
            
            # Convert to PIL and perform OCR
            try:
                pil_image = Image.fromarray(cropped_image)
                text = self.mocr(pil_image)
                text_ocr_list.append(text)
            except Exception as e:
                print(f"OCR error for box {box}: {e}")
                text_ocr_list.append("")
        
        return text_ocr_list
        
    def unload_model(self):
        """Unload model and free memory."""
        if self.mocr is None:
            print("Model is not loaded yet")
        else:
            del self.mocr
            self.mocr = None
            gc.collect()
            print("Model unloaded")