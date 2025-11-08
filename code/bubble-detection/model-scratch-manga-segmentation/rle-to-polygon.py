import os
import json
import cv2
import pycocotools.mask as maskUtils
from tqdm import tqdm

# --- Configuration ---
# Directory containing the original JSON files
ORIGINAL_JSON_DIR = 'MangaSegmentation/jsons'
# Directory where the new, processed JSON files will be saved
PROCESSED_JSON_DIR = 'MangaSegmentation/jsons_processed'

def preprocess_all_jsons():
    """
    Reads all original JSONs, converts all RLE annotations to polygons,
    and saves them as new files. This is a one-time operation.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(PROCESSED_JSON_DIR, exist_ok=True)
    
    json_files = [f for f in os.listdir(ORIGINAL_JSON_DIR) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON files to process.")

    for json_file in tqdm(json_files, desc="Processing JSON files"):
        original_path = os.path.join(ORIGINAL_JSON_DIR, json_file)
        
        with open(original_path, 'r') as f:
            data = json.load(f)
        
        # We will build a new list of annotations
        processed_annotations = []
        
        # Use tqdm for a progress bar on the annotations within each file
        for ann in tqdm(data['annotations'], desc=f"  -> Processing '{json_file}'", leave=False):
            
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
                    continue
                
                processed_ann['segmentation'] = polygons
            
            processed_annotations.append(processed_ann)
            
        # Replace the old annotations with our newly processed list
        data['annotations'] = processed_annotations
        
        # Save the new, clean JSON file to the processed directory
        processed_path = os.path.join(PROCESSED_JSON_DIR, json_file)
        with open(processed_path, 'w') as f:
            json.dump(data, f)
            
    print(f"\nProcessing complete. New JSON files are saved in '{PROCESSED_JSON_DIR}'")

if __name__ == '__main__':
    preprocess_all_jsons()