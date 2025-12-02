import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate
from tqdm import tqdm
from tabulate import tabulate
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2


class MangaOCRDataset(Dataset):
    """
    Dataset for MangaOCR evaluation using parsed JSON annotations.
    """
    def __init__(self, image_paths: List[str], boxes_list: List[List], 
                 ground_truth_texts: List[List[str]], bbox_type: str = "text"):
        """
        Args:
            image_paths: List of paths to manga images
            boxes_list: List of bounding boxes for each image, 
                       each element is a list of boxes [[x_min, y_min, x_max, y_max], ...]
            ground_truth_texts: List of ground truth texts for each image,
                               each element is a list of text strings corresponding to boxes
            bbox_type: Type of bbox used - "text" or "bubble"
        """
        self.image_paths = image_paths
        self.boxes_list = boxes_list
        self.ground_truth_texts = ground_truth_texts
        self.bbox_type = bbox_type
        
        # Validate that all lists have same length
        assert len(image_paths) == len(boxes_list) == len(ground_truth_texts), \
            "All input lists must have the same length"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # Load image on-the-fly to save memory
        image_path = self.image_paths[index]
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return {
            "image": img_rgb,
            "image_path": image_path,
            "boxes": self.boxes_list[index],
            "ground_truth": self.ground_truth_texts[index]
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length boxes and texts.
    Keeps boxes and ground_truth as lists instead of trying to stack them as tensors.
    """
    images = [item["image"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    boxes = [item["boxes"] for item in batch]
    ground_truth = [item["ground_truth"] for item in batch]
    
    return {
        "image": images,
        "image_path": image_paths,
        "boxes": boxes,
        "ground_truth": ground_truth
    }


class MangaOCREvaluator:
    def __init__(self, device='cpu'):
        """
        Evaluator for MangaOCR models using Character Error Rate (CER) and Word Error Rate (WER).
        
        Args:
            device: Device to use for computation ('cpu', 'cuda', or 'mps')
        """
        self.device = device
        
    def evaluate(self, ocr_model, dataset, batch_size=1, verbose=False, bbox_type="text"):
        """
        Evaluate OCR model on the given dataset.
        
        Args:
            ocr_model: OCR model with load_model(), predict(), and unload_model() methods
            dataset: MangaOCRDataset instance containing images, boxes, and ground truth
            batch_size: Batch size for DataLoader (default=1, recommended for manga images)
            verbose: If True, print detailed predictions
            bbox_type: Type of bbox being evaluated ("text" or "bubble")
            
        Returns:
            Dictionary containing CER and WER metrics
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        ocr_model.load_model()
        
        all_predicted = []
        all_expected = []
        
        print(f"Starting OCR evaluation with {bbox_type} bboxes...")
        for batch in tqdm(dataloader):
            batch_images = batch["image"]
            batch_boxes = batch["boxes"]
            batch_ground_truth = batch["ground_truth"]
            
            # Process each item in the batch
            for i in range(len(batch_images)):
                image = batch_images[i]
                boxes = batch_boxes[i]
                ground_truth = batch_ground_truth[i]
                
                # Predict OCR for each box
                predicted_texts = ocr_model.predict(boxes, image)
                
                if verbose:
                    print(f"\nImage {len(all_predicted) + 1}:")
                    print(f"  Number of boxes: {len(boxes)}")
                    print(f"  Predictions: {predicted_texts}")
                    print(f"  Ground truth: {ground_truth}")

                all_predicted.extend(predicted_texts)
                all_expected.extend(ground_truth)
        
        ocr_model.unload_model()
        
        # Filter out empty predictions and ground truth for fair evaluation
        filtered_predicted = []
        filtered_expected = []
        for pred, exp in zip(all_predicted, all_expected):
            if exp and exp.strip():  # Only include if ground truth is not empty
                filtered_predicted.append(pred if pred else "")
                filtered_expected.append(exp)
        
        if len(filtered_predicted) == 0:
            print("Warning: No valid text pairs found for evaluation")
            return {
                "cer": 1.0,
                "wer": 1.0,
                "num_samples": 0,
                "bbox_type": bbox_type
            }
        
        # Compute metrics
        metric_cer = CharErrorRate()
        cer = metric_cer(filtered_predicted, filtered_expected)
        
        metric_wer = WordErrorRate()
        wer = metric_wer(filtered_predicted, filtered_expected)
        
        # Display results
        metrics_data = [
            ["BBox Type", bbox_type.upper()],
            ["Character Error Rate (CER)", f"{cer.item():.4f}"],
            ["Word Error Rate (WER)", f"{wer.item():.4f}"],
            ["Number of Text Samples", f"{len(filtered_predicted)}"]
        ]
        
        print("\n" + '='*60)
        print(f"OCR EVALUATION METRICS ({bbox_type.upper()} BBOX)")
        print(tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="heavy_outline"))
        print('='*60)
        
        return {
            "cer": cer.item(),
            "wer": wer.item(),
            "num_samples": len(filtered_predicted),
            "bbox_type": bbox_type
        }
    
    def compare_bbox_types(self, ocr_model, json_path: str, images_dir: str, 
                          batch_size=1, verbose=False, max_images=None):
        """
        Compare OCR performance using text bbox vs bubble bbox.
        
        Args:
            ocr_model: OCR model with load_model(), predict(), and unload_model() methods
            json_path: Path to processed JSON annotation file
            images_dir: Directory containing manga images
            batch_size: Batch size for DataLoader
            verbose: If True, print detailed predictions
            max_images: Maximum number of images to evaluate (None for all)
            
        Returns:
            Dictionary with comparison results
        """
        # Load both datasets
        text_data = self.load_manga109_annotations(json_path, images_dir, bbox_type="text")
        bubble_data = self.load_manga109_annotations(json_path, images_dir, bbox_type="bubble")
        
        # Limit number of images if specified
        if max_images is not None:
            text_data = {k: v[:max_images] if isinstance(v, list) else v for k, v in text_data.items()}
            bubble_data = {k: v[:max_images] if isinstance(v, list) else v for k, v in bubble_data.items()}
        
        # Create datasets
        text_dataset = MangaOCRDataset(
            text_data["image_paths"],
            text_data["boxes_list"],
            text_data["ground_truth_texts"],
            bbox_type="text"
        )
        
        bubble_dataset = MangaOCRDataset(
            bubble_data["image_paths"],
            bubble_data["boxes_list"],
            bubble_data["ground_truth_texts"],
            bbox_type="bubble"
        )
        
        # Evaluate with text bbox
        print("\n" + "="*60)
        print("EVALUATING WITH TEXT BBOX")
        print("="*60)
        text_metrics = self.evaluate(ocr_model, text_dataset, batch_size, verbose, bbox_type="text")
        
        # Evaluate with bubble bbox
        print("\n" + "="*60)
        print("EVALUATING WITH BUBBLE BBOX")
        print("="*60)
        bubble_metrics = self.evaluate(ocr_model, bubble_dataset, batch_size, verbose, bbox_type="bubble")
        
        # Compare results
        comparison_data = [
            ["", "Text BBox", "Bubble BBox", "Difference"],
            ["CER", f"{text_metrics['cer']:.4f}", f"{bubble_metrics['cer']:.4f}", 
             f"{bubble_metrics['cer'] - text_metrics['cer']:.4f}"],
            ["WER", f"{text_metrics['wer']:.4f}", f"{bubble_metrics['wer']:.4f}",
             f"{bubble_metrics['wer'] - text_metrics['wer']:.4f}"],
            ["Samples", str(text_metrics['num_samples']), str(bubble_metrics['num_samples']), ""]
        ]
        
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print(tabulate(comparison_data[1:], headers=comparison_data[0], tablefmt="heavy_outline"))
        print("="*60)
        
        return {
            "text_bbox": text_metrics,
            "bubble_bbox": bubble_metrics
        }
    
    @staticmethod
    def load_manga109_annotations(json_path: str, images_dir: str, bbox_type: str = "text") -> Dict:
        """
        Load annotations from processed JSON file and prepare data for evaluation.
        Compatible with link_segment_to_text.py output structure.
        
        Handles:
            - category_id=5 (speech_bubble): Has bubble_bbox and linked texts
            - category_id=10 (unlinked_text): Standalone text annotations
        
        Args:
            json_path: Path to processed JSON annotation file
            images_dir: Directory containing manga images
            bbox_type: Type of bbox to use:
                - "text": Use text_bbox (from text annotations)
                - "bubble": Use bubble_bbox (from speech bubble segmentation)
            
        Returns:
            Dictionary containing:
                - image_paths: List of image file paths
                - boxes_list: List of bounding boxes per image
                - ground_truth_texts: List of ground truth text per image
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        images_dir = Path(images_dir)
        
        # Create mapping from image_id to image info
        image_id_to_info = {img['id']: img for img in json_data['images']}
        
        # Group annotations by image_id
        image_annotations = {}
        for ann in json_data['annotations']:
            # Only process annotations with text
            if not ann.get('texts') or len(ann['texts']) == 0:
                continue
            
            # Skip if no valid text content
            if not any(text.strip() for text in ann['texts']):
                continue
            
            # For bubble bbox type, only include speech bubbles (category_id=5) with bubble_bbox
            if bbox_type == "bubble":
                if ann.get('category_id') != 5:
                    continue
                if not ann.get('bubble_bbox') or len(ann['bubble_bbox']) == 0:
                    continue
                
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Prepare output lists
        image_paths = []
        boxes_list = []
        ground_truth_texts = []
        
        # Get manga name from images_dir to filter relevant images
        manga_name = images_dir.name
        
        for image_id in sorted(image_annotations.keys()):
            if image_id not in image_id_to_info:
                continue
                
            image_info = image_id_to_info[image_id]
            file_name = image_info['file_name']
            
            # Filter: only process images belonging to the target manga
            # file_name format: "MangaName/xxx.jpg"
            if '/' in file_name:
                file_manga_name, base_name = file_name.split('/', 1)
                if file_manga_name != manga_name:
                    continue
                # Use only the base filename since images_dir already includes manga name
                image_path = images_dir / base_name
            else:
                image_path = images_dir / file_name
            
            if not image_path.exists():
                continue
            
            annotations = image_annotations[image_id]
            boxes = []
            texts = []
            
            for ann in annotations:
                bbox = None
                
                # Choose bbox based on type
                if bbox_type == "bubble":
                    # Use bubble bbox format: [xmin, ymin, xmax, ymax]
                    if ann.get('bubble_bbox') and len(ann['bubble_bbox']) == 4:
                        bbox = ann['bubble_bbox']
                else:
                    # Use text bbox
                    if ann.get('text_bbox') and len(ann['text_bbox']) == 4:
                        # text_bbox format: [xmin, ymin, xmax, ymax]
                        bbox = ann['text_bbox']
                    else:
                        # Fallback to standard bbox format: [x, y, w, h] -> [xmin, ymin, xmax, ymax]
                        if ann.get('bbox') and len(ann['bbox']) == 4:
                            x, y, w, h = ann['bbox']
                            bbox = [x, y, x + w, y + h]
                
                if bbox is None:
                    continue
                
                boxes.append(bbox)
                
                # Concatenate all texts in this annotation
                text = ' '.join(ann['texts'])
                texts.append(text)
            
            if boxes:  # Only add if there are valid boxes
                image_paths.append(str(image_path))
                boxes_list.append(boxes)
                ground_truth_texts.append(texts)
        
        return {
            "image_paths": image_paths,
            "boxes_list": boxes_list,
            "ground_truth_texts": ground_truth_texts
        }
