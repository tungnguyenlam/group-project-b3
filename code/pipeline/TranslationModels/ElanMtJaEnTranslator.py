from transformers import pipeline
from typing import List
import gc

from .Translator import Translator


class ElanMtJaEnTranslator(Translator):
    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self, device='auto', elan_model='tiny'):
        if self.model is None:
            model_map = {
                'bt': 'Mitsua/elan-mt-bt-ja-en',
                'base': 'Mitsua/elan-mt-base-ja-en',
                'tiny': 'Mitsua/elan-mt-tiny-ja-en'
            }

            if elan_model not in model_map:
                raise ValueError(f"Invalid elan model: {elan_model}, please choose from 'bt', 'base', 'tiny'")

            self.model = pipeline('translation', model=model_map[elan_model], framework='pt', device_map=device)
        else:
            print("Model is already loaded")

    def _translate(self, texts: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        translated_texts = []
        for text in texts:
            translation = self.model(text)
            translated_texts.append(translation[0]["translation_text"])

        return translated_texts

    def unload_model(self):
        del self.model
        gc.collect()
        self.model = None
        print("Model unloaded")
