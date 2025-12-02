import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
from torchmetrics.text import CHRFScore
from torchmetrics.text.bert import BERTScore
from datasets import load_dataset
from tqdm import tqdm
from tabulate import tabulate

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
        print("Loading Dataset...")
        self.ds_raw_test = load_dataset('opus100', 'en-ja', split='test', token=hf_token)

    def evaluate(self, model, batch_size=8, verbose=False, device='cpu'):
        # Create the evaluation dataset
        val_ds = EvaluationDataset(self.ds_raw_test, 'ja', 'en')
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model.load_model()
        
        source_texts = []
        expected = []
        predicted = []
        
        print(f"Starting evaluation on {self.device}...")
        for batch in tqdm(val_dataloader):
            batch_src_texts = batch["src_text"]
            batch_tgt_texts = batch["tgt_text"]
            
            # Predict
            batch_predicted_texts = model.predict(batch_src_texts)

            if verbose:
                print(f"Src: {batch_src_texts[0]}")
                print(f"Tgt: {batch_tgt_texts[0]}")
                print(f"Pred: {batch_predicted_texts[0]}")

            source_texts.extend(batch_src_texts)
            expected.extend(batch_tgt_texts)
            predicted.extend(batch_predicted_texts)

        model.unload_model()
            
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