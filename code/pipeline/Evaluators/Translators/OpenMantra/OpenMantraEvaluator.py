import json
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
from tqdm.auto import tqdm
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

from evals_utils.utils import save_output_to_json, save_metrics_to_json, print_tabulate_results
from TranslationEvaluator import TranslationEvaluator

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


class OpenMantraEvaluator(TranslationEvaluator):
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

    def get_dataset(self):
        base_dataset = OpenMantraDataset(
            root_dir=self.openmantra_root,
            book_titles=self.book_titles
        )
        eval_dataset = OpenMantraImageDataset(
            base_dataset,
            source_lang=self.source_lang,
            target_lang=self.target_lang
        )
        return eval_dataset

    def get_dataloader(self, batch_size=1):
        eval_dataset = self.get_dataset()
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        return eval_dataloader

    def run_inference(self, model, dataloader, save_dir: Optional[str] = None, save_steps: int = 50):
        source_texts = []
        expected = []
        predicted = []
        
        dataset = self.get_dataset()  # Use dataset directly, not dataloader
        
        for i in tqdm(range(len(dataset))):
            src_texts, tgt_texts = dataset[i]  # Returns clean lists directly
            
            predicted_texts = model.predict(src_texts)
            source_texts.extend(src_texts)
            expected.extend(tgt_texts)
            predicted.extend(predicted_texts)

            if save_dir is not None and i % save_steps == 0:
                save_output_to_json(source_texts, expected, predicted, save_dir)
        return source_texts, expected, predicted