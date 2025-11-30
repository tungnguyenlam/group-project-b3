import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore, SacreBLEUScore
from torchmetrics.text import CHRFScore
from torchmetrics.text.bert import BERTScore
from datasets import load_dataset
from tqdm.auto import tqdm
from tabulate import tabulate
from transformers import pipeline
import gc

class FloresDataset(Dataset):
    def __init__(self, ds_src, ds_tgt):
        """
        ds_src: Japanese dataset
        ds_tgt: English dataset (or target language)
        They should be aligned by index
        """
        self.ds_src = ds_src
        self.ds_tgt = ds_tgt
        assert len(ds_src) == len(ds_tgt), "Source and target datasets must have same length"
    
    def __len__(self):
        return len(self.ds_src)
    
    def __getitem__(self, index):
        return {
            "src_text": self.ds_src[index]['text'],
            "tgt_text": self.ds_tgt[index]['text']
        }

class FloresPlusEvaluator:
    def __init__(self, hf_token=None):
        self.hf_token = hf_token
        
    def evaluate(self, model, batch_size=1, device='cpu', verbose=True):

        ds_jpn = load_dataset("openlanguagedata/flores_plus", "jpn_Jpan", split = 'devtest')
        ds_eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split = 'devtest')
        flores_ds = FloresDataset(ds_jpn, ds_eng)
        flores_dataloader = DataLoader(flores_ds, batch_size=batch_size, shuffle=False)
        
        model.load_model()
        
        source_texts = []
        expected = []
        predicted = []
        
        print("Starting evaluation...")
        for i, batch in enumerate(tqdm(flores_dataloader)):
            batch_src_texts = batch["src_text"]
            batch_tgt_texts = batch["tgt_text"]
            
            batch_predicted_texts = model.predict(batch_src_texts)

            if verbose:
                if i % 20 == 0:
                    print(f"Batch {i} source texts: {batch_src_texts}")
                    print(f"Batch {i} target texts: {batch_tgt_texts}")
                    print(f"Batch {i} predicted texts: {batch_predicted_texts}")

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

        sacre_bleu = SacreBLEUScore()
        sacre_bleu_score = sacre_bleu(predicted, bleu_formatted_refs)

        metric_chrf = CHRFScore(n_char_order=6, n_word_order=0)
        chrf = metric_chrf(predicted, bleu_formatted_refs)
        
        metric_chrf_pp = CHRFScore(n_char_order=6, n_word_order=2) # n_char_order=6, n_word_order=2 is standard "chrF++"
        chrf_pp = metric_chrf_pp(predicted, bleu_formatted_refs)      # CHRF also expects [[ref]] format

        self.bert_scorer = BERTScore(lang="en", rescale_with_baseline=False, device=device, verbose=verbose)
        self.bert_scorer.reset()
        self.bert_scorer.update(predicted, expected)
        bert_results = self.bert_scorer.compute()
        bert_f1 = bert_results['f1'].mean()

        if verbose:
            metrics_data = [
                ["Character Error Rate (CER)", f"{cer.item():.4f}"],
                ["Word Error Rate (WER)", f"{wer.item():.4f}"],
                ["BLEU Score", f"{bleu.item():.4f}"],
                ["SacreBLEU Score", f"{sacre_bleu_score.item():.4f}"],
                ["chrF Score", f"{chrf.item():.4f}"],
                ["chrF++ Score", f"{chrf_pp.item():.4f}"],
                ["BERTScore F1", f"{bert_f1.item():.4f}"] 
            ]
            
            print("\n" + '='*60)
            print(f"EVALUATION METRICS ON {len(expected)} EXAMPLES")
            print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))
            print('='*60)

        del metric_cer, metric_wer, metric_bleu, sacre_bleu, metric_chrf, self.bert_scorer
        gc.collect()

        return {
            "cer": cer.item(),
            "wer": wer.item(),
            "bleu": bleu.item(),
            "sacrebleu": sacre_bleu_score.item(),
            "chrf": chrf.item(),
            "chrf_pp": chrf_pp.item(),
            "bertscore": bert_f1.item()
        }





