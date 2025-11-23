import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
from datasets import load_dataset
from tqdm import tqdm
from tabulate import tabulate

try:
    from pipeline.TranslationModels.interfaces import TranslationModel
except ImportError:
    from ....TranslationModels.interfaces import TranslationModel

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

class Opus100Evaluator:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        # Load the dataset
        self.ds_raw_test = load_dataset('opus100', 'en-ja', split='validation', token=hf_token)
        
    def evaluate(self, model: TranslationModel, batch_size=8, device='cpu', verbose=False):
        # Create the evaluation dataset (Text-In, Text-Out)
        val_ds = EvaluationDataset(self.ds_raw_test, 'ja', 'en')
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        source_texts = []
        expected = []
        predicted = []
        
        print("Starting evaluation...")
        for batch in tqdm(val_dataloader):
            batch_src_texts = batch["src_text"]
            batch_tgt_texts = batch["tgt_text"]
            
            # The model handles tokenization internally!
            # We just pass the list of strings.
            batch_predicted_texts = model.predict(batch_src_texts)

            if verbose:
                print(f"Batch source texts: {batch_src_texts}")
                print(f"Batch target texts: {batch_tgt_texts}")
                print(f"Batch predicted texts: {batch_predicted_texts}")

            source_texts.extend(batch_src_texts)
            expected.extend(batch_tgt_texts)
            predicted.extend(batch_predicted_texts)
            
        # Compute metrics
        metric_cer = CharErrorRate()
        cer = metric_cer(predicted, expected)
        
        metric_wer = WordErrorRate()
        wer = metric_wer(predicted, expected)
        
        metric_bleu = BLEUScore()
        bleu = metric_bleu(predicted, [[ref] for ref in expected])  # Wrap each reference in a list
        
        metrics_data = [
            ["Character Error Rate (CER)", f"{cer.item():.4f}"],
            ["Word Error Rate (WER)", f"{wer.item():.4f}"],
            ["BLEU Score", f"{bleu.item():.4f}"]
        ]
        
        print("\n" + '='*60)
        print(f"EVALUATION METRICS ON {len(expected)} EXAMPLES")
        print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))
        print('='*60)
        
        return {
            "cer": cer.item(),
            "wer": wer.item(),
            "bleu": bleu.item()
        }





