from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Union
import torch
import gc

class ElanMtJaEnBatchTranslator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self, device='auto', elan_model='tiny'):
        if self.model is None:
            if device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'

            model_map = {
                'bt': 'Mitsua/elan-mt-bt-ja-en',
                'base': 'Mitsua/elan-mt-base-ja-en',
                'tiny': 'Mitsua/elan-mt-tiny-ja-en'
            }
            
            if elan_model not in model_map:
                raise ValueError(f"Invalid elan model: {elan_model}, please choose from 'bt', 'base', 'tiny'")
            
            model_name = model_map[elan_model]
            self.device = device
            
            print(f"Loading {model_name} to {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            
            self.model.eval()
        else:
            print("Model is already loaded")
        
    def predict(self, source_texts: Union[str, List[str]]) -> List[str]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if isinstance(source_texts, str):
            source_texts = [source_texts]
            
        inputs = self.tokenizer(
            source_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs, 
                max_new_tokens=128
            )
            
        translated_texts = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        return translated_texts

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        
        gc.collect()
        
        if self.device and 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        elif self.device and 'mps' in str(self.device):
             torch.mps.empty_cache()

        print("Model unloaded")