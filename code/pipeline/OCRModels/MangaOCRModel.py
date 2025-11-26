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

    def predict(self, bboxes, image):
        """
        Predict OCR text for each bounding box in the image.
        
        Args:
            bboxes: List of bounding boxes, each in format [x_min, y_min, x_max, y_max]
            image: numpy array of the full image (RGB format)
            
        Returns:
            List of OCR text strings, one for each bounding box
        """
        if self.mocr is None:
            raise ValueError("Model has not been loaded successfully")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        if not bboxes or len(bboxes) == 0:
            print("No bounding boxes provided")
            return []
        
        ocr_results = []
        
        # Process each bounding box
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            
            # Convert to integers
            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)
            
            # Crop the image
            cropped = image[y_min:y_max, x_min:x_max]
            
            # Handle empty crops
            if cropped.size == 0:
                print(f"Warning: Empty crop for bbox {i}: {bbox}")
                ocr_results.append("")
                continue
            
            # Convert to PIL Image and perform OCR
            try:
                pil_image = Image.fromarray(cropped)
                text = self.mocr(pil_image)
                ocr_results.append(text)
            except Exception as e:
                print(f"OCR error for bbox {i} {bbox}: {e}")
                ocr_results.append("")
        
        print(f"OCR completed for {len(bboxes)} text bubbles")
        return ocr_results
    
    def transform_output(self, raw_output):
        """
        Transform raw OCR output to pipeline format.
        Ensures all outputs are strings.
        
        Args:
            raw_output: List of OCR results from predict()
            
        Returns:
            List of string texts in pipeline format
        """
        return [str(text) if text is not None else "" for text in raw_output]

    def unload_model(self):
        if self.mocr == None:
            print("The model is not loaded yet")
        else:
            del self.mocr
            self.mocr = None
            gc.collect()
            
