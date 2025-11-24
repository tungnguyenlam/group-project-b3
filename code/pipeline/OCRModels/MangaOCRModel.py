from manga_ocr import MangaOcr
from PIL import Image
import numpy as np
import gc


class MangaOCRModel:
    def __init__(self):
        self.mocr = None

    def load_model(self):
        try:
            self.mocr = MangaOcr()
            print("Model load complete")
        except:
            print("Failed to load MangaOCR model")

    def predict(self, bboxes_or_image, image=None):
        """
        Flexible predict method that works with both pipeline and notebook usage.
        
        Usage 1 (Pipeline format - recommended):
            bboxes = [[x_min, y_min, x_max, y_max], ...]
            texts = model.predict(bboxes, image)
            
        Usage 2 (Notebook format - single cropped image):
            cropped_image = image[y:y+h, x:x+w]
            text = model.predict(cropped_image)
        """
        if self.mocr is None:
            raise ValueError("Model has not been loaded successfully")
        
        # Case 1: Single cropped image (notebook usage)
        if image is None:
            cropped = bboxes_or_image
            if not isinstance(cropped, np.ndarray):
                raise TypeError("Image must be a numpy array")
            
            # Perform OCR on single image
            try:
                pil_image = Image.fromarray(cropped)
                text = self.mocr(pil_image)
                return text
            except Exception as e:
                print(f"OCR error: {e}")
                return ""
        
        # Case 2: List of bboxes + full image (pipeline usage)
        bboxes = bboxes_or_image
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        ocr_results = []
        
        # Process each bounding box
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            
            # Convert to integers
            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)
            
            # Crop the image
            cropped = image[y_min:y_max, x_min:x_max]
            
            # Handle empty crops
            if cropped.size == 0:
                ocr_results.append("")
                continue
            
            # Convert to PIL Image and perform OCR
            try:
                pil_image = Image.fromarray(cropped)
                text = self.mocr(pil_image)
                ocr_results.append(text)
            except Exception as e:
                print(f"OCR error for bbox {bbox}: {e}")
                ocr_results.append("")
        
        print(f"OCR completed for {len(bboxes)} text bubbles")
        return ocr_results
    
    def transform_output(self, raw_output):
        # Ensure all outputs are strings
        return [str(text) if text is not None else "" for text in raw_output]

    def unload_model(self):
        if self.mocr == None:
            print("The model is not loaded yet")
        else:
            del self.mocr
            self.mocr = None
            gc.collect()
            
