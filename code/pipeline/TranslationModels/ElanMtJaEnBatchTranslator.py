from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import torch
import gc

from .Translator import Translator


class ElanMtJaEnBatchTranslator(Translator):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = None

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

            self.model_name = model_map[elan_model]
            self.device = device

            print(f"Loading {self.model_name} to {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

            self.model.eval()
        else:
            print("Model is already loaded")

    def _translate(self, texts: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_new_tokens=128
            )

        return self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

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
