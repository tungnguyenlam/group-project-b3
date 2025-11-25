from transformers import pipeline
from typing import List
import gc
class ElanMtJaEnTranslator:
    def __init__(self):
        self.model = None

    def load_model(self, device='auto', elan_model='tiny'):
        if self.model is None:
            if elan_model == 'bt':
                self.model = pipeline('translation', model='Mitsua/elan-mt-bt-ja-en', framework='pt', device_map=device)
            elif elan_model == 'base':
                self.model = pipeline('translation', model='Mitsua/elan-mt-base-ja-en', framework='pt', device_map=device)
            elif elan_model == 'tiny':
                self.model = pipeline('translation', model='Mitsua/elan-mt-tiny-ja-en', framework='pt', device_map=device)
            else:
                raise ValueError(f"Invalid elan model: {elan_model}, please choose from 'bt', 'base', 'tiny'")
        else:
            print("Model is already loaded")
        
    def predict(self, source_texts: List[str]) -> List[str]:
        """
        Takes a list of source strings and returns a list of translated strings.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if isinstance(source_texts, str):
            source_texts = [source_texts]
        
        translated_texts = []
        for text in source_texts:
            translation = self.model(text)
            translated_texts.append(translation[0]["translation_text"])
        
        return translated_texts

    def unload_model(self):
        del self.model
        gc.collect()
        self.model = None
        print("Model unloaded")
