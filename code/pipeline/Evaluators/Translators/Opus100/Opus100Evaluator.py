import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore, SacreBLEUScore
from torchmetrics.text import CHRFScore
from torchmetrics.text.bert import BERTScore
from datasets import load_dataset
from tqdm.auto import tqdm
from tabulate import tabulate
import gc
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

from TranslationEvaluator import TranslationEvaluator
class EvaluationDataset(Dataset):
    def __init__(self, ds, src_lang, tgt_lang):
        super().__init__()
        self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        return {
            "src_text": src_text,
            "tgt_text": tgt_text
        }

class Opus100Evaluator(TranslationEvaluator):
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        
    def get_dataset(self):
        ds_raw = load_dataset('opus100', 'en-ja', split='test', token=self.hf_token)
        return EvaluationDataset(ds_raw, 'ja', 'en')  # Wrap it!
    
    def get_dataloader(self, batch_size=1):
        eval_dataset = self.get_dataset()
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        return eval_dataloader


