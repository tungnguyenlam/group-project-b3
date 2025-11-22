#!/usr/bin/env python3
"""
Script to link segmentation polygons with text annotations.
For speech bubbles (category_id=5), this script adds text_ids and texts fields
by finding all text bounding boxes that are completely contained within each polygon.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
from shapely.geometry import Polygon, Point
import argparse


def load_xml_annotations(xml_path: Path) -> Dict[int, List[Dict]]:
    """
    Load text annotations from XML file organized by page index.
    
    Returns:
        Dictionary mapping page_index -> list of text annotations
        Each text annotation contains: id, xmin, ymin, xmax, ymax, text
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    page_texts = {}
    
    for page in root.findall('.//page'):
        page_index = int(page.get('index'))
        texts = []
        
        for text_elem in page.findall('text'):
            text_data = {
                'id': text_elem.get('id'),
                'xmin': int(text_elem.get('xmin')),
                'ymin': int(text_elem.get('ymin')),
                'xmax': int(text_elem.get('xmax')),
                'ymax': int(text_elem.get('ymax')),
                'text': text_elem.text if text_elem.text else ''
            }
            texts.append(text_data)
        
        page_texts[page_index] = texts
    
    return page_texts


def point_in_polygon(x: float, y: float, polygon_coords: List[float]) -> bool:
    """
    Check if a point (x, y) is inside a polygon.
    
    Args:
        x, y: Point coordinates
        polygon_coords: Flat list of [x1, y1, x2, y2, ..., xn, yn]
    
    Returns:
        True if point is inside polygon
    """
    # Convert flat list to list of (x, y) tuples
    points = [(polygon_coords[i], polygon_coords[i+1]) 
              for i in range(0, len(polygon_coords), 2)]
    
    polygon = Polygon(points)
    point = Point(x, y)
    
    return polygon.contains(point)


def bbox_in_polygon(text_bbox: Dict, polygon_coords: List[float]) -> bool:
    """
    Check if a text bounding box is completely inside a polygon.
    
    Args:
        text_bbox: Dictionary with xmin, ymin, xmax, ymax
        polygon_coords: Flat list of polygon coordinates
    
    Returns:
        True if all four corners of the bbox are inside the polygon
    """
    xmin, ymin = text_bbox['xmin'], text_bbox['ymin']
    xmax, ymax = text_bbox['xmax'], text_bbox['ymax']
    
    # Check all four corners
    corners = [
        (xmin, ymin),  # top-left
        (xmax, ymin),  # top-right
        (xmin, ymax),  # bottom-left
        (xmax, ymax)   # bottom-right
    ]
    
    for x, y in corners:
        if not point_in_polygon(x, y, polygon_coords):
            return False
    
    return True


def link_texts_to_segments(json_data: Dict, page_texts: Dict[int, List[Dict]]) -> Dict:
    """
    Add text_ids and texts fields to annotations with category_id=5.
    
    Args:
        json_data: The loaded JSON data with images and annotations
        page_texts: Dictionary mapping page_index -> list of text annotations
    
    Returns:
        Modified json_data with text_ids and texts added to relevant annotations
    """
    # Create mapping from image_id to page_index
    # Assuming file_name format is "BookName/XXX.jpg" where XXX is the page number
    image_id_to_page = {}
    for img in json_data['images']:
        file_name = img['file_name']
        # Extract page number from filename (e.g., "ARMS/003.jpg" -> 3)
        page_num_str = Path(file_name).stem
        try:
            page_index = int(page_num_str)
            image_id_to_page[img['id']] = page_index
        except ValueError:
            print(f"Warning: Could not extract page index from {file_name}")
            continue
    
    # Process each annotation
    for ann in json_data['annotations']:
        # Only process speech bubbles
        if ann['category_id'] != 5:
            continue
        
        image_id = ann['image_id']
        if image_id not in image_id_to_page:
            ann['text_ids'] = []
            ann['texts'] = []
            continue
        
        page_index = image_id_to_page[image_id]
        if page_index not in page_texts:
            ann['text_ids'] = []
            ann['texts'] = []
            continue
        
        # Get polygon coordinates (first segmentation)
        if not ann['segmentation'] or len(ann['segmentation']) == 0:
            ann['text_ids'] = []
            ann['texts'] = []
            continue
        
        polygon_coords = ann['segmentation'][0]
        
        # Find all texts that fit inside this polygon
        matching_texts = []
        for text_data in page_texts[page_index]:
            if bbox_in_polygon(text_data, polygon_coords):
                matching_texts.append(text_data)
        
        # Sort texts by xmax descending (right to left)
        matching_texts.sort(key=lambda t: t['xmax'], reverse=True)
        
        # Add to annotation
        ann['text_ids'] = [t['id'] for t in matching_texts]
        ann['texts'] = [t['text'] for t in matching_texts]
    
    return json_data


def process_manga(json_path: Path, xml_path: Path, output_path: Path):
    """
    Process a single manga: link segments to texts and save output.
    
    Args:
        json_path: Path to processed JSON file with polygons
        xml_path: Path to XML file with text annotations
        output_path: Path to save output JSON
    """
    print(f"Processing {json_path.stem}...")
    
    # Load data
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    page_texts = load_xml_annotations(xml_path)
    
    # Link texts to segments
    json_data = link_texts_to_segments(json_data, page_texts)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    speech_bubbles = [ann for ann in json_data['annotations'] if ann['category_id'] == 5]
    bubbles_with_text = [ann for ann in speech_bubbles if len(ann.get('text_ids', [])) > 0]
    
    print(f"  Total speech bubbles: {len(speech_bubbles)}")
    print(f"  Bubbles with text: {len(bubbles_with_text)}")
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Link segmentation polygons with text annotations'
    )
    parser.add_argument(
        '--json-dir',
        type=Path,
        default=Path('data/MangaSegmentation/jsons_processed'),
        help='Directory containing processed JSON files'
    )
    parser.add_argument(
        '--xml-dir',
        type=Path,
        default=Path('data/Manga109_released_2023_12_07/annotations'),
        help='Directory containing XML annotation files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/MangaOCR'),
        help='Directory to save output JSON files'
    )
    parser.add_argument(
        '--manga-name',
        type=str,
        help='Process only this manga (without extension). If not specified, process all.'
    )
    
    args = parser.parse_args()
    
    # Get list of JSON files to process
    if args.manga_name:
        json_files = [args.json_dir / f"{args.manga_name}.json"]
    else:
        json_files = sorted(args.json_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each manga
    success_count = 0
    error_count = 0
    
    for json_path in json_files:
        manga_name = json_path.stem
        xml_path = args.xml_dir / f"{manga_name}.xml"
        output_path = args.output_dir / f"{manga_name}.json"
        
        if not xml_path.exists():
            print(f"Warning: XML file not found for {manga_name}, skipping...")
            error_count += 1
            continue
        
        try:
            process_manga(json_path, xml_path, output_path)
            success_count += 1
        except Exception as e:
            print(f"Error processing {manga_name}: {e}")
            error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")


if __name__ == '__main__':
    main()
