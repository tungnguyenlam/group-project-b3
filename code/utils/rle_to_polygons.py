import os
import json
import cv2
import pycocotools.mask as maskUtils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys

END_WITH_LOCAL = 'bubble-segmentation-final-deep-learning'

os.environ['PATH'] = f"/root/.cargo/bin:{os.environ['PATH']}"

BASE_DIR = os.getcwd()
print(f"BASE_DIR: {BASE_DIR}")

if not (BASE_DIR.endswith('/content') or BASE_DIR.endswith(END_WITH_LOCAL)):
    raise ValueError(f"Expected to be in .../{END_WITH_LOCAL} or .../content directory, but got: {BASE_DIR}")

ORIGINAL_JSON_DIR = os.path.join(BASE_DIR,'data','MangaSegmentation/jsons')
PROCESSED_JSON_DIR = os.path.join(BASE_DIR,'data','MangaSegmentation/jsons_processed')

def process_single_annotation(ann):
    """
    Process a single annotation, converting RLE to polygons if needed.
    Returns the processed annotation or None if it should be skipped.
    """
    processed_ann = ann.copy()
    segmentation_data = processed_ann['segmentation']
    
    # If the segmentation is an RLE dictionary, convert it
    if isinstance(segmentation_data, dict):
        binary_mask = maskUtils.decode(segmentation_data)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # A valid polygon requires at least 3 points (6 values)
            if contour.size >= 6:
                polygons.append(contour.flatten().tolist())
        
        # If no valid polygons were found, we skip this annotation
        if not polygons:
            return None
        
        processed_ann['segmentation'] = polygons
    
    return processed_ann

def process_single_json(args):
    """
    Process a single JSON file. This function is called by each worker process.
    """
    json_file, original_dir, processed_dir = args
    
    original_path = os.path.join(original_dir, json_file)
    
    with open(original_path, 'r') as f:
        data = json.load(f)
    
    # Process all annotations with a progress bar
    processed_annotations = []
    for ann in tqdm(data['annotations'], desc=f"Processing {json_file}", leave=False, position=1):
        processed_ann = process_single_annotation(ann)
        if processed_ann is not None:
            processed_annotations.append(processed_ann)
    
    # Replace the old annotations with our newly processed list
    data['annotations'] = processed_annotations
    
    # Save the new, clean JSON file to the processed directory
    processed_path = os.path.join(processed_dir, json_file)
    with open(processed_path, 'w') as f:
        json.dump(data, f)
    
    return json_file

def preprocess_all_jsons():
    """
    Reads all original JSONs, converts all RLE annotations to polygons,
    and saves them as new files. This is a one-time operation.
    Uses multiprocessing to utilize all CPU cores.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(PROCESSED_JSON_DIR, exist_ok=True)
    
    json_files = [f for f in os.listdir(ORIGINAL_JSON_DIR) if f.endswith('.json')]
    
    num_cores = cpu_count()
    print(f"Found {len(json_files)} JSON files to process using {num_cores} CPU cores.")
    
    # Prepare arguments for each worker
    args_list = [(json_file, ORIGINAL_JSON_DIR, PROCESSED_JSON_DIR) for json_file in json_files]
    
    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_single_json, args_list), 
                  total=len(json_files), 
                  desc="Processing JSON files",
                  position=0))
    
    print(f"\nProcessing complete. New JSON files are saved in '{PROCESSED_JSON_DIR}'")

if __name__ == '__main__':
    preprocess_all_jsons()