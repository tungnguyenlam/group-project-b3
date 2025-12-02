import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
from torchmetrics.text import CHRFScore
from torchmetrics.text.bert import BERTScore
from datasets import load_dataset
from tqdm import tqdm
from tabulate import tabulate
from transformers import pipeline

class NTREXDataset(Dataset):
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

class NTREXEvaluator:
    def __init__(self, hf_token=None, device='cpu'):
        self.hf_token = hf_token
        # Store device globally for the class if needed, 
        # though the evaluate method accepts a device argument as well.
        self.device = device 
        
    def evaluate(self, model, batch_size=8, device='cpu', verbose=False):

        # ---------------------------------------------------------------------
        # CORRECT LOADING STRATEGY FOR xianf/NTREX
        # 1. Load source (Japanese) using config "ja"
        # 2. Load target (English) using config "en"
        # 3. Use split='train' based on your observation of the dataset structure
        # ---------------------------------------------------------------------
        print("Loading xianf/NTREX dataset...")
        
        try:
            ds_jpn = load_dataset("xianf/NTREX", "ja", split='train', token=self.hf_token)
            ds_eng = load_dataset("xianf/NTREX", "en", split='train', token=self.hf_token)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {}

        # Create the paired dataset
        ntrex_ds = NTREXDataset(ds_jpn, ds_eng)
        ntrex_dataloader = DataLoader(ntrex_ds, batch_size=batch_size, shuffle=False)
        
        # Load the translation model
        model.load_model()
        
        source_texts = []
        expected = []
        predicted = []
        
        print("Starting evaluation...")
        for batch in tqdm(ntrex_dataloader):
            batch_src_texts = batch["src_text"]
            batch_tgt_texts = batch["tgt_text"]
            
            # Perform translation
            batch_predicted_texts = model.predict(batch_src_texts)

            if verbose:
                print(f"Batch source texts: {batch_src_texts}")
                print(f"Batch target texts: {batch_tgt_texts}")
                print(f"Batch predicted texts: {batch_predicted_texts}")

            # Collect results
            source_texts.extend(batch_src_texts)
            expected.extend(batch_tgt_texts)
            predicted.extend(batch_predicted_texts)
        
        # Free up memory
        model.unload_model()

        # ---------------------------------------------------------------------
        # COMPUTE METRICS
        # ---------------------------------------------------------------------
        
        # Compute metrics
        metric_cer = CharErrorRate()
        cer = metric_cer(predicted, expected)
        
        metric_wer = WordErrorRate()
        wer = metric_wer(predicted, expected)
        
        metric_bleu = BLEUScore()                               # BLEU expects references as [[ref1, ref2], [ref1]]
        bleu_formatted_refs = [[ref] for ref in expected]       # Since we have 1 ref per item, we wrap it: [ref] -> [[ref]]
        bleu = metric_bleu(predicted, bleu_formatted_refs)

        metric_chrf = CHRFScore(n_char_order=6, n_word_order=0) # n_char_order=6, n_word_order=2 is standard "chrF++"
        chrf = metric_chrf(predicted, bleu_formatted_refs)      # CHRF also expects [[ref]] format
        
        metric_chrf_pp = CHRFScore(n_char_order=6, n_word_order=2) # n_char_order=6, n_word_order=2 is standard "chrF++"
        chrf_pp = metric_chrf_pp(predicted, bleu_formatted_refs)      # CHRF also expects [[ref]] format

        self.bert_scorer = BERTScore(lang="en", rescale_with_baseline=False)
        self.bert_scorer.to(device)
        self.bert_scorer.reset()
        self.bert_scorer.update(predicted, expected)
        bert_results = self.bert_scorer.compute()
        bert_f1 = bert_results['f1'].mean()
        

        metrics_data = [
            ["Character Error Rate (CER)", f"{cer.item():.4f}"],
            ["Word Error Rate (WER)", f"{wer.item():.4f}"],
            ["BLEU Score", f"{bleu.item():.4f}"],
            ["crf Score", f"{chrf.item():.4f}"],
            ["chrF++ Score", f"{chrf_pp.item():.4f}"],     
            ["BERTScore F1", f"{bert_f1.item():.4f}"] 
        ]
        
        print("\n" + '='*60)
        print(f"EVALUATION METRICS ON {len(expected)} EXAMPLES")
        print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))
        print('='*60)
        
        return {
            "cer": cer.item(),
            "wer": wer.item(),
            "bleu": bleu.item(),
            "chrf": chrf.item(),
            "chrf_pp": chrf_pp.item(),
            "bertscore": bert_f1.item()
        }