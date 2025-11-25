from manga_ocr import MangaOcr
from PIL import Image
import gc

def transform_img_to_PIL(img):
    return Image.fromarray(img)

class MangaOCRModel:
    def __init__(self):
        self.mocr = None

    def load_model(self):
        self.mocr = MangaOcr()

    def predict(self, img):
        if self.mocr is None:
            raise TypeError("Model is not loaded yet")
        elif isinstance(img, Image.Image):
            return self.mocr(img)
        else:
            return self.mocr(transform_img_to_PIL(img))
        
    def unload_model(self):
        del self.mocr
        gc.collect()
        self.mocr = None
            
