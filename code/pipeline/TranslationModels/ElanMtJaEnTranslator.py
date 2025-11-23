from transformers import pipeline
from typing import List
from .interfaces import TranslationModel

class ElanMtJaEnTranslator(TranslationModel):
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = pipeline('translation', model='Mitsua/elan-mt-bt-ja-en', framework='pt')
        else:
            raise TypeError("Model is already loaded")
        
    def predict(self, source_texts: List[str]) -> List[str]:  # Fix: Handle list of strings
        """
        Takes a list of source strings and returns a list of translated strings.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Handle both single string and list
        if isinstance(source_texts, str):
            source_texts = [source_texts]
        
        translated_texts = []
        for text in source_texts:
            translation = self.model(text)
            translated_texts.append(translation[0]["translation_text"])
        
        return translated_texts