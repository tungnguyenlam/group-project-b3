# code/pipeline/Evaluators/evals_utils/TranslationEvaluator.py

from abc import abstractmethod
from typing import Dict, List, Optional, Any
import gc
from tqdm.auto import tqdm
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore, SacreBLEUScore, CHRFScore
from Evaluator import Evaluator
try:
    from evals_utils.utils import save_output_to_json, save_metrics_to_json, print_tabulate_results
except ImportError:
    from ..evals_utils.utils import save_output_to_json, save_metrics_to_json, print_tabulate_results

class TranslationEvaluator(Evaluator):
    """
    Abstract base class for all translation evaluators.
    
    Provides common functionality for computing translation metrics:
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - BLEU Score
    - SacreBLEU Score
    - chrF Score
    - chrF++ Score
    
    Subclasses must implement:
    - get_dataset(): Load and return the specific evaluation dataset
    - get_dataloader(): Create and return a DataLoader for the dataset
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        super().__init__()
        self.hf_token = hf_token
    
    @abstractmethod
    def get_dataset(self) -> Any:
        """
        Load and return the evaluation dataset.
        
        Returns:
            Dataset object compatible with PyTorch DataLoader.
        """
        pass
    
    @abstractmethod
    def get_dataloader(self, batch_size: int = 1) -> Any:
        """
        Create and return a DataLoader for the evaluation dataset.
        
        Args:
            batch_size: Batch size for evaluation.
            
        Returns:
            PyTorch DataLoader.
        """
        pass

    def save_output_to_json(self, source_texts: List[str], expected: List[str], predicted: List[str], file_dir: str):
        save_output_to_json(source_texts, expected, predicted, file_dir)

    def save_metrics_to_json(self, metrics: Dict[str, float], file_dir: str):
        save_metrics_to_json(metrics, file_dir)

    def print_tabulate_results(self, metrics: Dict[str, float]):
        print_tabulate_results(metrics)
    
    def compute_metrics(
        self, 
        predicted: List[str], 
        expected: List[str], 
        device: str = 'auto'
    ) -> Dict[str, float]:
        """
        Compute all translation metrics.
        
        Args:
            predicted: List of predicted translations.
            expected: List of ground truth translations.
            device: Device for BERTScore computation.
            
        Returns:
            Dictionary with all metric values.
        """
        # CER
        metric_cer = CharErrorRate()
        cer = metric_cer(predicted, expected)
        
        # WER
        metric_wer = WordErrorRate()
        wer = metric_wer(predicted, expected)
        
        # Format references for BLEU-style metrics: [[ref1], [ref2], ...]
        bleu_formatted_refs = [[ref] for ref in expected]
        
        # BLEU
        metric_bleu = BLEUScore()
        bleu = metric_bleu(predicted, bleu_formatted_refs)
        
        # SacreBLEU
        sacre_bleu = SacreBLEUScore()
        sacre_bleu_score = sacre_bleu(predicted, bleu_formatted_refs)
        
        # chrF
        metric_chrf = CHRFScore(n_char_order=6, n_word_order=0)
        chrf = metric_chrf(predicted, bleu_formatted_refs)
        
        # chrF++
        metric_chrf_pp = CHRFScore(n_char_order=6, n_word_order=2)
        chrf_pp = metric_chrf_pp(predicted, bleu_formatted_refs)
    
        
        # Cleanup
        del metric_cer, metric_wer, metric_bleu, sacre_bleu, metric_chrf, metric_chrf_pp
        gc.collect()
        
        return {
            "cer": cer.item(),
            "wer": wer.item(),
            "bleu": bleu.item(),
            "sacrebleu": sacre_bleu_score.item(),
            "chrf": chrf.item(),
            "chrf_pp": chrf_pp.item()
        }

    def run_inference(self, model, dataloader, save_dir: Optional[str] = None, save_steps: int = 50):
        source_texts = []
        expected = []
        predicted = []
        
        for i, batch in enumerate(tqdm(dataloader)):
            batch_src_texts = batch["src_text"]
            batch_tgt_texts = batch["tgt_text"]
            batch_predicted_texts = model.predict(batch_src_texts)
            source_texts.extend(batch_src_texts)
            expected.extend(batch_tgt_texts)
            predicted.extend(batch_predicted_texts)

            if save_dir is not None and i % save_steps == 0:
                self.save_output_to_json(source_texts, expected, predicted, save_dir)
            
        return source_texts, expected, predicted
    
    def evaluate(
        self,
        model: Any,
        batch_size: int = 1,
        device: str = 'auto',
        verbose: bool = True,
        save_steps: int = 50,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a translation model.
        
        Args:
            model: Translation model with predict(List[str]) -> List[str] method.
                   Must also have load_model() and unload_model() methods.
            batch_size: Batch size for evaluation.
            device: Device for computation.
            verbose: If True, print progress and sample predictions.
            save_steps: Save intermediate results every N steps.
            save_dir: Directory to save outputs. If None, results are not saved.
            
        Returns:
            Dictionary with metric values.
        """
        from tqdm.auto import tqdm
        
        dataloader = self.get_dataloader(batch_size)
        
        model.load_model(device=device)
        
        source_texts, expected, predicted = self.run_inference(model, dataloader, save_dir, save_steps)
        
        model.unload_model()
        
        # Compute metrics
        metrics = self.compute_metrics(predicted, expected, device)
        metrics["model_name"] = getattr(model, 'model_name', 'unknown')
        
        # Print and save results
        if verbose:
            self.print_tabulate_results(metrics)
        
        if save_dir is not None:
            self.save_output_to_json(source_texts, expected, predicted, save_dir)
            self.save_metrics_to_json(metrics, save_dir)
        
        return metrics