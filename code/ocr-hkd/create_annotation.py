import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# Third-party imports
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = os.getcwd() 
JSON_DIR = os.path.join(BASE_DIR, 'data', 'MangaSegmentation', 'jsons_processed')
IMAGE_ROOT_DIR = os.path.join(BASE_DIR, 'data', 'Manga109_released_2023_12_07', 'images')
ANNOTATION_DIR = os.path.join(BASE_DIR, 'data', 'Manga109_released_2023_12_07', 'annotations')

# Output file
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, 'data', 'manga109_ocr_dataset.json')

# Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
ALLOWED_CATEGORY_IDS = [5]  # ID 5 is 'balloon' in Manga109 

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_path_exists(path, description):
    """Verifies that a directory exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{description}] not found at: {path}")

def get_split_mapping(image_root_dir, random_seed):
    """
    Reproduces the exact train/test split used in YOLO training.
    """
    all_books = [d for d in os.listdir(image_root_dir) 
                 if os.path.isdir(os.path.join(image_root_dir, d))]
    all_books.sort() 
    
    train_titles, test_titles = train_test_split(
        all_books, 
        test_size=TEST_SIZE, 
        random_state=random_seed
    )
    return set(train_titles), set(test_titles)

def get_text_sort_key(text_obj):
    """
    Sorts text regions based on Japanese reading order:
    1. Right to Left (Decreasing X)
    2. Top to Bottom (Increasing Y)
    """
    bbox = text_obj['bbox']
    return (-bbox[2], bbox[1]) 

def check_point_in_box(point, box):
    """Checks if a point (x, y) is inside a bounding box."""
    px, py = point
    b_xmin, b_ymin, b_xmax, b_ymax = box
    return b_xmin <= px <= b_xmax and b_ymin <= py <= b_ymax

# =============================================================================
# CORE PROCESSING LOGIC
# =============================================================================

def process_book(book_name, split_type, json_data_map):
    """
    Matches XML text regions with JSON bubble masks for a single book.
    """
    book_results = []
    
    # 1. Load XML Annotation (Text Source)
    xml_path = os.path.join(ANNOTATION_DIR, f"{book_name}.xml")
    if not os.path.exists(xml_path): 
        return []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []
    
    # Iterate through pages
    for page in root.findall(".//page"):
        page_index = int(page.get("index"))
        img_filename = f"{page_index:03d}.jpg"
        
        # Construct the lookup key using the Book Name and Filename
        # Logic: "BookName/003.jpg"
        lookup_key = f"{book_name}/{img_filename}"
        
        # --- A. Parse Text Regions (XML) ---
        all_texts = []
        for text_node in page.findall(".//text"):
            try:
                xmin = int(text_node.get("xmin"))
                ymin = int(text_node.get("ymin"))
                xmax = int(text_node.get("xmax"))
                ymax = int(text_node.get("ymax"))
                content = text_node.text.strip() if text_node.text else ""
                
                if not content: continue
                
                all_texts.append({
                    "bbox": (xmin, ymin, xmax, ymax),
                    "center": ((xmin + xmax) / 2, (ymin + ymax) / 2),
                    "content": content,
                    "matched": False
                })
            except: continue
            
        # --- B. Parse Bubble Regions (JSON) ---
        json_record = json_data_map.get(lookup_key)
        all_bubbles = []
        
        if json_record:
            for ann in json_record.get("annotations", []):
                # Filter by Category ID
                if ann.get('category_id') not in ALLOWED_CATEGORY_IDS: 
                    continue 
                
                # Extract Segmentation Mask
                seg = ann.get("segmentation", [])
                
                # Calculate BBox from Polygon
                all_xs, all_ys = [], []
                for poly in seg:
                    if len(poly) < 4: continue
                    all_xs.extend(poly[0::2])
                    all_ys.extend(poly[1::2])
                
                if not all_xs: continue
                bbox = (min(all_xs), min(all_ys), max(all_xs), max(all_ys))
                
                all_bubbles.append({
                    "bbox": bbox,
                    "mask": seg,
                    "texts_inside": []
                })

        # --- C. Mapping Logic (Point inside Box) ---
        for text in all_texts:
            for bubble in all_bubbles:
                if check_point_in_box(text['center'], bubble['bbox']):
                    bubble['texts_inside'].append(text)
                    text['matched'] = True
                    break 
        
        # --- D. Generate Output ---
        
        # 1. Bubbles containing text (Blue)
        for bubble in all_bubbles:
            if bubble['texts_inside']:
                sorted_texts = sorted(bubble['texts_inside'], key=get_text_sort_key)
                combined_text = "".join([t['content'] for t in sorted_texts])
                
                book_results.append({
                    "img_path": lookup_key,
                    "bbox": bubble['bbox'],
                    "mask": bubble['mask'], # Export Mask
                    "text": combined_text,
                    "type": "bubble",
                    "split": split_type
                })
        
        # 2. Orphan Text (Red)
        for text in all_texts:
            if not text['matched']:
                book_results.append({
                    "img_path": lookup_key,
                    "bbox": text['bbox'],
                    "mask": [], # Empty mask for orphan text
                    "text": text['content'],
                    "type": "text_orphan",
                    "split": split_type
                })
                
    return book_results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("--- Starting Final Annotation Generation Process ---")
    
    # Validation
    try:
        check_path_exists(JSON_DIR, "JSON Directory")
        check_path_exists(IMAGE_ROOT_DIR, "Image Root Directory")
        check_path_exists(ANNOTATION_DIR, "XML Annotation Directory")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # [Step 1] Cache JSONs with Path Correction
    print("\n[Step 1/4] Caching Segmentation JSONs (with Path Correction)...")
    json_map = {} 
    
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    if not json_files:
        print("Error: No JSON files found.")
        return

    for jf in tqdm(json_files, desc="Loading JSONs"):
        # MUST SET: JSON filename as the Book Name
        book_name_from_file = os.path.splitext(jf)[0]
        
        try:
            with open(os.path.join(JSON_DIR, jf), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for img in data.get('images', []):
                    # Extract only the filename (e.g., "003.jpg")
                    raw_filename = img['file_name']
                    filename_only = Path(raw_filename).name 
                    
                    # Force the correct key format: "BookName/003.jpg"
                    corrected_key = f"{book_name_from_file}/{filename_only}"
                    
                    # Attach annotations
                    img_id = img['id']
                    anns = [a for a in data.get('annotations', []) if a['image_id'] == img_id]
                    img['annotations'] = anns
                    
                    # Store in map
                    json_map[corrected_key] = img
        except Exception as e:
            print(f"Warning: Failed to load {jf}: {e}")

    print(f"Cached metadata for {len(json_map)} total pages.")

    # [Step 2] Determine Split
    print("\n[Step 2/4] Determining Train/Test Split...")
    train_books, test_books = get_split_mapping(IMAGE_ROOT_DIR, RANDOM_SEED)
    print(f"Train Books: {len(train_books)} | Test Books: {len(test_books)}")

    # [Step 3] Process Books
    print("\n[Step 3/4] Processing Books (Mapping XML <-> JSON)...")
    final_dataset = []
    all_books = sorted(list(train_books) + list(test_books))
    
    for book in tqdm(all_books, desc="Processing"):
        split = "train" if book in train_books else "test"
        items = process_book(book, split, json_map)
        final_dataset.extend(items)

    # [Step 4] Save Output
    print(f"\n[Step 4/4] Saving results...")
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    # Summary Statistics
    bubbles = [x for x in final_dataset if x['type'] == 'bubble']
    orphans = [x for x in final_dataset if x['type'] == 'text_orphan']
    
    print("\n" + "="*40)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("="*40)
    print(f"Output File:      {OUTPUT_JSON_PATH}")
    print(f"Total Samples:    {len(final_dataset)}")
    print(f" - Bubbles (Blue): {len(bubbles)}")
    print(f" - Orphans (Red):  {len(orphans)}")
    print(f"Train/Test Split: {len([x for x in final_dataset if x['split']=='train'])} / {len([x for x in final_dataset if x['split']=='test'])}")
    
    if len(bubbles) > 0:
        print("\nVerification: Bubbles detected successfully.")
    else:
        print("\nWARNING: No bubbles detected. Please check Category IDs again.")

if __name__ == "__main__":
    main()
    
# MUST BE AT ROOT DIRECTORY TO RUN THIS CODE
# Run: python -m code.ocr-hkd.create_annotation 