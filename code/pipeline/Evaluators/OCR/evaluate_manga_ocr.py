#!/usr/bin/env python3
"""
CLI script to evaluate MangaOCR on a single manga volume.
Compares performance using text bbox vs bubble bbox.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR / 'code'))

from MangaOCREvaluator import ParseAnnotation, MangaOCREvaluator
from pipeline.OCRModels.MangaOCRModel import MangaOCRModel
import torch


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MangaOCR on a single manga volume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on AisazuNihaIrarenai (first 10 pages)
  python evaluate_manga_ocr.py --manga-name AisazuNihaIrarenai --max-images 10
  
  # Evaluate on all pages with verbose output
  python evaluate_manga_ocr.py --manga-name AisazuNihaIrarenai --verbose
  
  # Only parse annotations without evaluation
  python evaluate_manga_ocr.py --manga-name AisazuNihaIrarenai --parse-only
  
  # Evaluate using only text bbox
  python evaluate_manga_ocr.py --manga-name AisazuNihaIrarenai --bbox-type text
        """
    )
    
    parser.add_argument(
        '--manga-name',
        type=str,
        required=True,
        help='Name of manga volume to evaluate (e.g., AisazuNihaIrarenai)'
    )
    
    parser.add_argument(
        '--xml-dir',
        type=Path,
        default=Path('data/Manga109_released_2023_12_07/annotations'),
        help='Directory containing XML annotations (default: data/Manga109_released_2023_12_07/annotations)'
    )
    
    parser.add_argument(
        '--images-dir',
        type=Path,
        default=Path('data/Manga109_released_2023_12_07/images'),
        help='Directory containing manga images (default: data/Manga109_released_2023_12_07/images)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/MangaOCR/jsons_processed'),
        help='Output directory for parsed JSON (default: data/MangaOCR/jsons_processed)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for evaluation (default: 1)'
    )
    
    parser.add_argument(
        '--bbox-type',
        type=str,
        choices=['text', 'bubble', 'compare'],
        default='compare',
        help='Type of bbox to evaluate: text, bubble, or compare both (default: compare)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed predictions during evaluation'
    )
    
    parser.add_argument(
        '--parse-only',
        action='store_true',
        help='Only parse XML to JSON without evaluation'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps', 'auto'],
        default='auto',
        help='Device to use for computation (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path.cwd()
    xml_path = base_dir / args.xml_dir / f"{args.manga_name}.xml"
    manga_images_dir = base_dir / args.images_dir / args.manga_name
    output_dir = base_dir / args.output_dir
    
    # Check if files exist
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        return 1
    
    if not manga_images_dir.exists():
        print(f"Error: Images directory not found: {manga_images_dir}")
        return 1
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print(f"MangaOCR Evaluation - {args.manga_name}")
    print("="*70)
    print(f"Device: {device}")
    print(f"XML: {xml_path}")
    print(f"Images: {manga_images_dir}")
    print(f"Max images: {args.max_images if args.max_images else 'All'}")
    print("="*70)
    
    # Step 1: Parse XML to JSON
    print("\n[1/2] Parsing XML annotations...")
    parser = ParseAnnotation(
        xml_path=str(xml_path),
        images_dir=str(manga_images_dir),
        output_dir=str(output_dir)
    )
    json_output_path = parser.parse_and_save()
    
    if args.parse_only:
        print("\n✓ Parse complete. Exiting (--parse-only flag set)")
        return 0
    
    # Step 2: Evaluate OCR
    print("\n[2/2] Evaluating OCR...")
    ocr_model = MangaOCRModel()
    evaluator = MangaOCREvaluator(device=device)
    
    if args.bbox_type == 'compare':
        # Compare both bbox types
        results = evaluator.compare_bbox_types(
            ocr_model=ocr_model,
            json_path=str(json_output_path),
            images_dir=str(manga_images_dir),
            batch_size=args.batch_size,
            verbose=args.verbose,
            max_images=args.max_images
        )
        
        # Save results to file
        results_file = output_dir / f"{args.manga_name}_comparison_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"MangaOCR Evaluation Results - {args.manga_name}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Text BBox CER: {results['text_bbox']['cer']:.4f}\n")
            f.write(f"Text BBox WER: {results['text_bbox']['wer']:.4f}\n")
            f.write(f"Text BBox Samples: {results['text_bbox']['num_samples']}\n\n")
            f.write(f"Bubble BBox CER: {results['bubble_bbox']['cer']:.4f}\n")
            f.write(f"Bubble BBox WER: {results['bubble_bbox']['wer']:.4f}\n")
            f.write(f"Bubble BBox Samples: {results['bubble_bbox']['num_samples']}\n\n")
            f.write(f"CER Difference: {results['bubble_bbox']['cer'] - results['text_bbox']['cer']:.4f}\n")
            f.write(f"WER Difference: {results['bubble_bbox']['wer'] - results['text_bbox']['wer']:.4f}\n")
        
        print(f"\n✓ Results saved to: {results_file}")
        
    else:
        # Evaluate single bbox type
        from MangaOCREvaluator import MangaOCRDataset
        
        data = evaluator.load_manga109_annotations(
            json_path=str(json_output_path),
            images_dir=str(manga_images_dir),
            bbox_type=args.bbox_type
        )
        
        # Limit images if specified
        if args.max_images is not None:
            data = {
                k: v[:args.max_images] if isinstance(v, list) else v 
                for k, v in data.items()
            }
        
        dataset = MangaOCRDataset(
            data["image_paths"],
            data["boxes_list"],
            data["ground_truth_texts"],
            bbox_type=args.bbox_type
        )
        
        results = evaluator.evaluate(
            ocr_model=ocr_model,
            dataset=dataset,
            batch_size=args.batch_size,
            verbose=args.verbose,
            bbox_type=args.bbox_type
        )
        
        # Save results to file
        results_file = output_dir / f"{args.manga_name}_{args.bbox_type}_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"MangaOCR Evaluation Results - {args.manga_name}\n")
            f.write(f"BBox Type: {args.bbox_type}\n")
            f.write("="*70 + "\n\n")
            f.write(f"CER: {results['cer']:.4f}\n")
            f.write(f"WER: {results['wer']:.4f}\n")
            f.write(f"Samples: {results['num_samples']}\n")
        
        print(f"\n✓ Results saved to: {results_file}")
    
    print("\n✓ Evaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
