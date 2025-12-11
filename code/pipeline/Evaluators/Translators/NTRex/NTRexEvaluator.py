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

class NTRexDataset(Dataset):
    def __init__(self, ds_src, ds_tgt):
        """
        ds_src: Source language dataset (e.g., Japanese)
        ds_tgt: Target language dataset (e.g., English)
        They should be aligned by index.
        """
        self.ds_src = ds_src
        self.ds_tgt = ds_tgt
        
        # Verify that both datasets have the same number of examples
        assert len(ds_src) == len(ds_tgt), "Source and target datasets must have same length"
    
    def __len__(self):
        return len(self.ds_src)
    
    def __getitem__(self, index):
        # Based on your finding: 
        # ds_src (config="ja") has a column 'text'
        # ds_tgt (config="en") has a column 'text'
        return {
            "src_text": self.ds_src[index]['text'],
            "tgt_text": self.ds_tgt[index]['text']
        }

class NTRexEvaluator(TranslationEvaluator):
    def __init__(self, hf_token=None):
        super().__init__(hf_token)
    
    def get_dataset(self):
        ds_jpn = load_dataset("xianf/NTREX","ja", split='train', token=self.hf_token)
        ds_eng = load_dataset("xianf/NTREX","en", split='train', token=self.hf_token)
        return NTRexDataset(ds_jpn, ds_eng)
    
    def get_dataloader(self, batch_size=1):
        ntrex_ds = self.get_dataset()
        ntrex_dataloader = DataLoader(ntrex_ds, batch_size=batch_size, shuffle=False)
        return ntrex_dataloader