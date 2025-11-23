from transformers import pipeline

class ElanMtJaEnTranslator:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = pipeline('translation', 
                                  model='Mitsua/elan-mt-bt-ja-en',
                                  framework="pt")
        else:
            raise TypeError("Model is already loaded")
        
    def predict(self, text):
        translation = self.model(text)
        return translation[0]["translation_text"]