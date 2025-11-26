from manga_ocr import MangaOcr
from PIL import Image
import gc
import numpy as np
from math import floor, ceil

def transform_img_to_PIL(img):
    return Image.fromarray(img)

class MangaOCRModel:
    def __init__(self):
        self.mocr = None

    def load_model(self):
        self.mocr = MangaOcr()

    def predict(self, img, bboxes):
        if self.mocr is None:
            raise TypeError("Model is not loaded yet")
        
        image_rgb = np.array(img)
        cropped_image_list = []

        for box in bboxes:
            cropped_image = image_rgb[ floor(box[1]): ceil(box[3]), floor(box[0]): ceil(box[2]), :]
            cropped_image_list.append(cropped_image)

        text_ocr_list = []

        for cropped_img in cropped_image_list:
            if isinstance(cropped_img, Image.Image):
                text = self.mocr(cropped_img)
            else:
                text = self.mocr(transform_img_to_PIL(cropped_img))
            text_ocr_list.append(text)

        return text_ocr_list
        
    def unload_model(self):
        del self.mocr
        gc.collect()
        self.mocr = None
            
