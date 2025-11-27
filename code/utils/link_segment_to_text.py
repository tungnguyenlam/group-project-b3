#!/usr/bin/env python3
"""
Script to link segmentation polygons with text annotations.
Logic changes:
1. Calculate bubble bbox from polygon (foremost min/max).
2. Check if text center point is inside bubble bbox -> link it.
3. Keep empty bubbles.
4. Keep unassigned texts (create new annotations for them).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


def load_xml_annotations(xml_path: Path) -> Dict[int, List[Dict]]:
    """
    Load text annotations from XML file organized by page index.
    Returns: Dictionary mapping page_index -> list of text annotations
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


def is_center_in_bbox(text_bbox: Dict, bubble_bbox: List[float]) -> bool:
    """
    Check if the center point of the text bbox is inside the bubble bbox.
    bubble_bbox format: [xmin, ymin, xmax, ymax]
    """
    # Calculate center of text
    text_cx = (text_bbox['xmin'] + text_bbox['xmax']) / 2
    text_cy = (text_bbox['ymin'] + text_bbox['ymax']) / 2
    
    b_xmin, b_ymin, b_xmax, b_ymax = bubble_bbox
    
    # Check if center is within bubble bbox limits
    return (b_xmin <= text_cx <= b_xmax) and (b_ymin <= text_cy <= b_ymax)


def link_texts_to_segments(json_data: Dict, page_texts: Dict[int, List[Dict]]) -> Dict:
    """
    1. Calculate bubble BBox from polygon.
    2. Link texts if center is inside bubble BBox.
    3. Create new annotations for texts that are not assigned to any bubble.
    """
    # Mapping image_id -> page_index
    image_id_to_page = {}
    page_to_image_id = {} # Reverse mapping for creating new annotations
    
    for img in json_data['images']:
        file_name = img['file_name']
        page_num_str = Path(file_name).stem
        try:
            page_index = int(page_num_str)
            image_id_to_page[img['id']] = page_index
            page_to_image_id[page_index] = img['id']
        except ValueError:
            print(f"Warning: Could not extract page index from {file_name}")
            continue
    
    # Track which texts have been assigned to a bubble
    # structure: {page_index: {text_id, ...}}
    assigned_texts = {p_idx: set() for p_idx in page_texts}
    
    # Process existing bubble annotations
    for ann in json_data['annotations']:
        # Only process speech bubbles (category_id 5)
        if ann['category_id'] != 5:
            continue
            
        image_id = ann['image_id']
        if image_id not in image_id_to_page:
            continue
            
        page_index = image_id_to_page[image_id]
        if page_index not in page_texts:
            ann['text_ids'] = []
            ann['texts'] = []
            ann['bubble_bbox'] = []
            continue
            
        # Get polygon coordinates
        if not ann.get('segmentation') or len(ann['segmentation']) == 0:
            ann['text_ids'] = []
            ann['texts'] = []
            ann['bubble_bbox'] = []
            continue
            
        polygon_coords = ann['segmentation'][0]
        
        # 1. Calculate Bubble BBox from Polygon (foremost xmin, xmax, ymin, ymax)
        xs = polygon_coords[0::2]
        ys = polygon_coords[1::2]
        
        if not xs or not ys:
            ann['text_ids'] = []
            ann['texts'] = []
            ann['bubble_bbox'] = []
            continue

        b_xmin, b_xmax = min(xs), max(xs)
        b_ymin, b_ymax = min(ys), max(ys)
        
        # Format: [xmin, ymin, xmax, ymax] used for checking
        bubble_bbox_absolute = [b_xmin, b_ymin, b_xmax, b_ymax]
        
        # Store in annotation (you can choose to store [x,y,w,h] in 'bbox' or this custom field)
        # Updating standard 'bbox' to ensure it matches the polygon tight box
        ann['bbox'] = [b_xmin, b_ymin, b_xmax - b_xmin, b_ymax - b_ymin]
        ann['bubble_bbox'] = bubble_bbox_absolute
        
        # 2. Find matching texts (Center inside Bubble BBox)
        matching_texts = []
        for text_data in page_texts[page_index]:
            if is_center_in_bbox(text_data, bubble_bbox_absolute):
                matching_texts.append(text_data)
                # Mark as assigned
                assigned_texts[page_index].add(text_data['id'])
        
        # Sort texts by xmax descending (right to left reading order assumption)
        matching_texts.sort(key=lambda t: t['xmax'], reverse=True)
        
        ann['text_ids'] = [t['id'] for t in matching_texts]
        ann['texts'] = [t['text'] for t in matching_texts]

    # 3. Handle unassigned texts (Texts that didn't fit in any bubble)
    # Generate new annotations for them
    
    # Find max annotation ID to avoid duplicates
    max_ann_id = 0
    if json_data['annotations']:
        max_ann_id = max(ann['id'] for ann in json_data['annotations'])
    
    new_annotations = []
    
    for page_index, texts in page_texts.items():
        if page_index not in page_to_image_id:
            continue
            
        image_id = page_to_image_id[page_index]
        used_ids = assigned_texts.get(page_index, set())
        
        for text_data in texts:
            if text_data['id'] not in used_ids:
                # This text was not assigned to any bubble. Create a new annotation.
                max_ann_id += 1
                
                width = text_data['xmax'] - text_data['xmin']
                height = text_data['ymax'] - text_data['ymin']
                
                new_ann = {
                    'id': max_ann_id,
                    'image_id': image_id,
                    'category_id': 10,  # Assign category 10 for "Unlinked Text"
                    'bbox': [text_data['xmin'], text_data['ymin'], width, height],
                    'area': width * height,
                    'segmentation': [], # No polygon for raw text box
                    'iscrowd': 0,
                    'text_ids': [text_data['id']],
                    'texts': [text_data['text']],
                    'is_unlinked_text': True # Flag to identify these easily later
                }
                new_annotations.append(new_ann)
    
    # Append new annotations to the main list
    if new_annotations:
        print(f"  Adding {len(new_annotations)} unlinked text annotations.")
        json_data['annotations'].extend(new_annotations)
    
    return json_data


def process_manga(json_path: Path, xml_path: Path, output_path: Path):
    """Process a single manga."""
    print(f"Processing {json_path.stem}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    page_texts = load_xml_annotations(xml_path)
    
    json_data = link_texts_to_segments(json_data, page_texts)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    speech_bubbles = [ann for ann in json_data['annotations'] if ann['category_id'] == 5]
    bubbles_with_text = [ann for ann in speech_bubbles if len(ann.get('text_ids', [])) > 0]
    unlinked_texts = [ann for ann in json_data['annotations'] if ann.get('category_id') == 10]
    
    print(f"  Total speech bubbles: {len(speech_bubbles)}")
    print(f"  Bubbles with text: {len(bubbles_with_text)}")
    print(f"  Unlinked texts added: {len(unlinked_texts)}")
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Link segmentation polygons with text annotations')
    parser.add_argument('--json-dir', type=Path, default=Path('data/MangaSegmentation/jsons_processed'))
    parser.add_argument('--xml-dir', type=Path, default=Path('data/Manga109_released_2023_12_07/annotations'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/MangaOCR/jsons_processed'))
    parser.add_argument('--manga-name', type=str, help='Process only this manga')
    
    args = parser.parse_args()
    
    if args.manga_name:
        json_files = [args.json_dir / f"{args.manga_name}.json"]
    else:
        json_files = sorted(args.json_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_path in json_files:
        manga_name = json_path.stem
        xml_path = args.xml_dir / f"{manga_name}.xml"
        output_path = args.output_dir / f"{manga_name}.json"
        
        if not xml_path.exists():
            continue
            
        try:
            process_manga(json_path, xml_path, output_path)
        except Exception as e:
            print(f"Error processing {manga_name}: {e}")

if __name__ == '__main__':
    main()