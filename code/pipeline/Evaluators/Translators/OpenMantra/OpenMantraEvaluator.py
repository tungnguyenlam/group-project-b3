import sys
import os
import json
import gc
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore, SacreBLEUScore, CHRFScore
from torchmetrics.text.bert import BERTScore
from tqdm.auto import tqdm
from tabulate import tabulate


class OpenMantraDataset(Dataset):
    """
    Dataset class for OpenMantra manga translation dataset.
    
    Each item returns:
        - image: PIL Image or torch.Tensor (depending on transform)
        - image_info: Dictionary containing metadata about the image
        - bubbles: List of dictionaries containing bubble/text annotations
    
    Args:
        root_dir: Path to the root directory of the OpenMantra dataset
        annotation_file: Name of the annotation JSON file (default: 'annotation.json')
        transform: Optional transform to apply to the images
        language: Language of images to load ('ja' for Japanese, default)
        book_titles: Optional list of book titles to filter (None = all books)
    """
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str = "annotation.json",
        transform: Optional[Any] = None,
        language: str = "ja",
        book_titles: Optional[List[str]] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.language = language
        
        # Load annotations
        annotation_path = self.root_dir / annotation_file
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Filter books if specified
        if book_titles is not None:
            self.annotations = [
                book for book in self.annotations 
                if book['book_title'] in book_titles
            ]
        
        # Flatten the dataset: create a list of (book_info, page_info) tuples
        self.samples: List[Tuple[Dict, Dict]] = []
        for book in self.annotations:
            book_title = book['book_title']
            for page in book['pages']:
                self.samples.append((
                    {'book_title': book_title},
                    page
                ))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Union[Image.Image, torch.Tensor], Dict, List[Dict]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing:
                - image: The manga page image (PIL Image or Tensor if transform applied)
                - image_info: Dictionary with image metadata:
                    - book_title: Title of the manga book
                    - page_index: Page number in the book
                    - image_path: Path to the image file
                    - image_size: Tuple of (width, height)
                    - frames: List of frame bounding boxes (if available)
                - bubbles: List of dictionaries, each containing:
                    - xmin, ymin, xmax, ymax: Bounding box coordinates
                    - text_ja: Japanese text
                    - text_en: English translation
                    - text_zh: Chinese translation
        """
        book_info, page_info = self.samples[idx]
        
        # Get image path
        image_paths = page_info.get('image_paths', {})
        relative_image_path = image_paths.get(self.language, '')
        image_path = self.root_dir / relative_image_path
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_size = image.size  # (width, height)
        
        # Build image info dictionary
        image_info = {
            'book_title': book_info['book_title'],
            'page_index': page_info.get('page_index', -1),
            'image_path': str(image_path),
            'image_size': image_size,
            'frames': page_info.get('frame', [])
        }
        
        # Get bubble annotations (text annotations) and convert bbox format
        raw_bubbles = page_info.get('text', [])
        bubbles = []
        for bubble in raw_bubbles:
            # Convert from (x, y, w, h) to (xmin, ymin, xmax, ymax)
            converted_bubble = {
                'xmin': bubble['x'],
                'ymin': bubble['y'],
                'xmax': bubble['x'] + bubble['w'],
                'ymax': bubble['y'] + bubble['h'],
                'text_ja': bubble.get('text_ja', ''),
                'text_en': bubble.get('text_en', ''),
                'text_zh': bubble.get('text_zh', '')
            }
            bubbles.append(converted_bubble)
        
        # Apply transform if specified
        if self.transform is not None:
            image = self.transform(image)
        
        return image, image_info, bubbles
    
    def get_book_titles(self) -> List[str]:
        """Get list of all book titles in the dataset."""
        return list(set(book['book_title'] for book in self.annotations))
    
    def get_pages_by_book(self, book_title: str) -> List[int]:
        """Get list of all page indices for a specific book."""
        indices = []
        for idx, (book_info, page_info) in enumerate(self.samples):
            if book_info['book_title'] == book_title:
                indices.append(idx)
        return indices


class OpenMantraImageDataset(Dataset):
    """
    PyTorch Dataset wrapper for OpenMantra that returns source/target text lists per image.
    
    Each sample returns all Japanese texts from a single manga page as a list,
    along with the corresponding English translations as ground truth.
    The order of bubbles is preserved - this is important for evaluation.
    
    Args:
        openmantra_dataset: An OpenMantraDataset instance.
        source_lang: Source language key ('text_ja', 'text_en', or 'text_zh'). Default: 'text_ja'
        target_lang: Target language key ('text_ja', 'text_en', or 'text_zh'). Default: 'text_en'
        filter_empty_pages: If True, skip pages with no valid bubble texts. Default: True
    """
    
    def __init__(
        self,
        openmantra_dataset: OpenMantraDataset,
        source_lang: str = 'text_ja',
        target_lang: str = 'text_en',
        filter_empty_pages: bool = True
    ):
        self.openmantra_dataset = openmantra_dataset
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Build index of valid pages
        self.valid_indices: List[int] = []
        
        for idx in range(len(openmantra_dataset)):
            _, _, bubbles = openmantra_dataset[idx]
            
            # Check if page has at least one valid bubble
            has_valid = any(
                bubble.get(source_lang, '').strip() and bubble.get(target_lang, '').strip()
                for bubble in bubbles
            )
            
            if has_valid or not filter_empty_pages:
                self.valid_indices.append(idx)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, index: int) -> Tuple[List[str], List[str]]:
        """
        Get source and target text lists for a single page.
        
        Args:
            index: Index of the sample
            
        Returns:
            Tuple containing:
                - src_texts: List of source language texts (in bubble order)
                - tgt_texts: List of target language texts (in bubble order)
        """
        actual_idx = self.valid_indices[index]
        _, _, bubbles = self.openmantra_dataset[actual_idx]
        
        # Extract texts from all bubbles, preserving order
        src_texts = []
        tgt_texts = []
        
        for bubble in bubbles:
            src_text = bubble.get(self.source_lang, '').strip()
            tgt_text = bubble.get(self.target_lang, '').strip()
            
            if src_text and tgt_text:
                src_texts.append(src_text)
                tgt_texts.append(tgt_text)
        
        return src_texts, tgt_texts


class OpenMantraEvaluator:
    """
    Evaluator for translation models using the OpenMantra manga translation dataset.
    
    Processes one page at a time (batch_size invariant for translation model).
    Each page contains multiple text bubbles with preserved order.
    
    Metrics computed:
        - Character Error Rate (CER)
        - Word Error Rate (WER)
        - BLEU Score
        - SacreBLEU Score
        - chrF Score
        - chrF++ Score
        - BERTScore F1
    
    Args:
        openmantra_root: Path to the OpenMantra dataset root directory.
        source_lang: Source language key. Default: 'text_ja' (Japanese)
        target_lang: Target language key. Default: 'text_en' (English)
        book_titles: Optional list of book titles to filter. None = use all books.
    """
    
    def __init__(self,
        openmantra_root: str,
        source_lang: str = 'text_ja',
        target_lang: str = 'text_en',
        book_titles: Optional[List[str]] = None
    ):
        self.openmantra_root = openmantra_root
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.book_titles = book_titles
    
    def evaluate(
        self,
        model,
        model_name: str,
        device: str = 'cpu',
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a translation model on the OpenMantra dataset.
        
        Args:
            model: Translation model with predict(List[str]) -> List[str] method.
                   Model should already be loaded before calling evaluate.
            device: Device for BERTScore computation. Default: 'cpu'
            verbose: If True, print sample predictions. Default: True
            
        Returns:
            Dictionary with metric values
        """
        # Load dataset
        base_dataset = OpenMantraDataset(
            root_dir=self.openmantra_root,
            book_titles=self.book_titles
        )
        eval_dataset = OpenMantraImageDataset(
            base_dataset,
            source_lang=self.source_lang,
            target_lang=self.target_lang
        )
        
        all_expected = []
        all_predicted = []
        
        print(f"Starting OpenMantra evaluation...")
        print(f"Dataset size: {len(eval_dataset)} pages")
        
        for i in tqdm(range(len(eval_dataset))):
            src_texts, tgt_texts = eval_dataset[i]
            
            # Translate all bubbles from this page
            predicted_texts = model.predict(src_texts)
            
            if verbose and i % 10 == 0:
                print(f"\n--- Page {i} ---")
                for src, tgt, pred in zip(src_texts, tgt_texts, predicted_texts):
                    print(f"  Src: {src}")
                    print(f"  Tgt: {tgt}")
                    print(f"  Pred: {pred}")
                    print()
            
            all_expected.extend(tgt_texts)
            all_predicted.extend(predicted_texts)
        
        # Compute metrics
        metric_cer = CharErrorRate()
        cer = metric_cer(all_predicted, all_expected)
        
        metric_wer = WordErrorRate()
        wer = metric_wer(all_predicted, all_expected)
        
        # BLEU expects references as [[ref1], [ref2], ...]
        bleu_formatted_refs = [[ref] for ref in all_expected]
        
        metric_bleu = BLEUScore()
        bleu = metric_bleu(all_predicted, bleu_formatted_refs)
        
        sacre_bleu = SacreBLEUScore()
        sacre_bleu_score = sacre_bleu(all_predicted, bleu_formatted_refs)
        
        metric_chrf = CHRFScore(n_char_order=6, n_word_order=0)
        chrf = metric_chrf(all_predicted, bleu_formatted_refs)
        
        metric_chrf_pp = CHRFScore(n_char_order=6, n_word_order=2)
        chrf_pp = metric_chrf_pp(all_predicted, bleu_formatted_refs)
        
        bert_scorer = BERTScore(lang="en", rescale_with_baseline=False, device=device, verbose=verbose)
        bert_scorer.reset()
        bert_scorer.update(all_predicted, all_expected)
        bert_results = bert_scorer.compute()
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
            
            print("\n" + '=' * 60)
            print(f"OPENMANTRA EVALUATION METRICS ON {model_name} ON {len(all_expected)} EXAMPLES")
            print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))
            print('=' * 60)
        
        # Cleanup
        del metric_cer, metric_wer, metric_bleu, sacre_bleu, metric_chrf, metric_chrf_pp, bert_scorer
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
