#!/usr/bin/env python3
"""
Step 3: Split and Prepare YOLO Dataset
========================================
This script handles:
- Loading data records from Step 1-2
- Grouping data by manga series (prevent data leakage)
- Splitting into train/val sets (80/20)
- Converting to YOLO instance segmentation format
- Copying images and creating label files

Prerequisites:
- Step 1-2 must be completed successfully
- data_records.json must exist

Outputs:
- YOLO_data/images/train/ : Training images
- YOLO_data/images/val/   : Validation images
- YOLO_data/labels/train/ : Training labels
- YOLO_data/labels/val/   : Validation labels
- .pipeline_state/s3_complete.json: Checkpoint file
"""

import os
import json
import shutil
import random
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
import shutil

# ===================================================================
# Configuration
# ===================================================================

END_WITH_LOCAL = 'bubble-segmentation-final-deep-learning'

BASE_DIR = os.getcwd()
print(f"BASE_DIR: {BASE_DIR}")

# Simple validation
if not (BASE_DIR.endswith('/content') or BASE_DIR.endswith(END_WITH_LOCAL)):
    raise ValueError(f"Expected to be in .../{END_WITH_LOCAL} or .../content directory, but got: {BASE_DIR}")

# Paths
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'YOLO_data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'code', 'bubble-detection', 'YOLO', '.pipeline_state')
DATA_RECORDS_FILE = os.path.join(BASE_DIR, 'code', 'bubble-detection', 'YOLO', 'data_records.json')

# ===================================================================
# Checkpoint System
# ===================================================================

def load_checkpoint(step_name):
    """Load checkpoint state. Returns None if not found or invalid."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{step_name}_complete.json")
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        if checkpoint.get("status") != "complete":
            return None
        
        return checkpoint
    except Exception as e:
        print(f"⚠ Error loading checkpoint: {e}")
        return None

def save_checkpoint(step_name, outputs):
    """Save checkpoint state."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "status": "complete",
        "outputs": outputs
    }
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{step_name}_complete.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"✓ Checkpoint saved: {checkpoint_file}")

# ===================================================================
# Validation Functions
# ===================================================================

def check_prerequisites():
    """Check if Step 1-2 was completed successfully."""
    print("\n" + "="*60)
    print("Checking Prerequisites")
    print("="*60)
    
    # Check for Step 1-2 checkpoint
    checkpoint = load_checkpoint("s1_2")
    if not checkpoint:
        print("⚠ Warning: Step 1-2 checkpoint not found!")
        print("  It's recommended to run: python s1_2_data_preparation.py")
        print("  Continuing anyway...\n")
    else:
        print("✓ Step 1-2 checkpoint found")
        print(f"  Timestamp: {checkpoint['timestamp']}")
    
    # Check for data records file
    if not os.path.exists(DATA_RECORDS_FILE):
        print(f"⚠ Warning: Data records file not found: {DATA_RECORDS_FILE}")
        print("  It's recommended to run: python s1_2_data_preparation.py")
        print("  Continuing anyway...\n")
        return []  # Return empty list instead of raising error
    
    print(f"✓ Data records file found: {DATA_RECORDS_FILE}")
    
    # Load and validate data
    try:
        with open(DATA_RECORDS_FILE, 'r') as f:
            data_records = json.load(f)
        
        if not data_records:
            print("⚠ Warning: Data records file is empty!")
            print("  Please check your data preparation step")
            return []
        
        print(f"✓ Loaded {len(data_records)} data records")
        print("\n✓ All prerequisites satisfied\n")
        
        return data_records
        
    except Exception as e:
        print(f"⚠ Warning: Could not load data records: {e}")
        print("  Continuing with empty dataset...\n")
        return []

def check_partial_work():
    """
    Check if there's partial work in the dataset directory.
    Returns True only if valid complete work is found.
    """
    print("\n" + "="*60)
    print("Checking for Existing Work")
    print("="*60)
    
    # Check if directories exist
    train_img_dir = os.path.join(DATASET_DIR, 'images/train')
    train_lbl_dir = os.path.join(DATASET_DIR, 'labels/train')
    val_img_dir = os.path.join(DATASET_DIR, 'images/val')
    val_lbl_dir = os.path.join(DATASET_DIR, 'labels/val')
    
    dirs_exist = any(os.path.exists(d) and os.listdir(d) for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir])
    
    if not dirs_exist:
        print("✓ No existing work found - will create new dataset\n")
        return False
    
    # Directories exist - check if we have a valid checkpoint
    checkpoint = load_checkpoint("s3")
    
    if checkpoint:
        # Validate the checkpoint matches current state
        expected_train = checkpoint['outputs'].get('train_images', 0)
        expected_val = checkpoint['outputs'].get('val_images', 0)
        
        actual_train = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
        actual_val = len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 0
        
        if actual_train == expected_train and actual_val == expected_val:
            print("✓ Found valid completed work with matching data")
            print(f"  Training images: {actual_train}")
            print(f"  Validation images: {actual_val}")
            return True
        else:
            print("⚠ Found work but checkpoint doesn't match")
            print(f"  Expected - Train: {expected_train}, Val: {expected_val}")
            print(f"  Found    - Train: {actual_train}, Val: {actual_val}")
    else:
        print("⚠ Found work but no valid checkpoint")
    
    # Clean up incomplete/invalid work
    print("\n🧹 Cleaning up incomplete work...")
    for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"  Deleted: {directory}")
    print("✓ Cleanup complete - will create fresh dataset\n")
    return False

# ===================================================================
# Multiprocessing Functions
# ===================================================================

def process_single_record(args):
    """Process a single record (for multiprocessing)."""
    record, split_type, dataset_dir = args
    
    try:
        original_img_path = record['file_name']
        img_height = record['height']
        img_width = record['width']
        
        # Check if image exists
        if not os.path.exists(original_img_path):
            return False, f"Image not found: {original_img_path}", 0
        
        # Create unique identifier to avoid filename collisions
        manga_title = Path(original_img_path).parts[-2]
        img_stem = Path(original_img_path).stem
        img_identifier = f"{manga_title}_{img_stem}"
        
        # 1. Copy image to train/val directory
        dest_img_path = os.path.join(dataset_dir, f'images/{split_type}', f"{img_identifier}.jpg")
        shutil.copy2(original_img_path, dest_img_path)
        
        # 2. Create corresponding label .txt file
        label_path = os.path.join(dataset_dir, f'labels/{split_type}', f"{img_identifier}.txt")
        
        # 3. Write normalized polygon coordinates
        annotation_count = 0
        with open(label_path, 'w') as f:
            for ann in record.get('annotations', []):
                segmentation = ann.get('segmentation')
                if not segmentation:
                    continue
                
                # Each object can have multiple polygons
                for poly in segmentation:
                    # Normalize polygon coordinates
                    normalized_poly = []
                    for i in range(0, len(poly), 2):
                        x = poly[i] / img_width
                        y = poly[i+1] / img_height
                        normalized_poly.extend([x, y])
                    
                    # Write in format: class_id x1 y1 x2 y2 ...
                    if normalized_poly:
                        f.write(f"0 {' '.join(map(str, normalized_poly))}\n")
                        annotation_count += 1
        
        return True, None, annotation_count
        
    except Exception as e:
        return False, str(e), 0

# ===================================================================
# Main Processing Functions
# ===================================================================

def split_data(data_records):
    """
    Split data into train/val sets, grouped by manga series to prevent data leakage.
    """
    print("\n" + "="*60)
    print("STEP 3A: Splitting Data by Manga Series")
    print("="*60)
    
    # Group data by manga title
    print("\nGrouping data by manga series...")
    grouped_data = defaultdict(list)
    for record in data_records:
        manga_name = Path(record['file_name']).parts[-2]
        grouped_data[manga_name].append(record)
    
    print(f"✓ Found {len(grouped_data)} unique manga series")
    
    # Display distribution
    series_sizes = [(name, len(records)) for name, records in grouped_data.items()]
    series_sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 series by image count:")
    for name, count in series_sizes[:5]:
        print(f"  - {name}: {count} images")
    
    # Split manga titles (80/20)
    manga_titles = list(grouped_data.keys())
    train_titles, val_titles = train_test_split(manga_titles, test_size=0.2, random_state=42)
    
    print(f"\n✓ Split into {len(train_titles)} training series and {len(val_titles)} validation series")
    
    # Reconstruct train/val lists
    train_data = [record for title in train_titles for record in grouped_data[title]]
    val_data = [record for title in val_titles for record in grouped_data[title]]
    
    # Shuffle
    random.Random(42).shuffle(train_data)
    random.Random(42).shuffle(val_data)
    
    print(f"✓ Final training set: {len(train_data)} images")
    print(f"✓ Final validation set: {len(val_data)} images")
    
    return train_data, val_data

def process_dataset_split_segmentation(data_split, split_type):
    """
    Process a dataset split and convert to YOLO instance segmentation format.
    Uses multiprocessing for faster processing.
    """
    print(f"\n" + "="*60)
    print(f"STEP 3B: Converting {split_type.upper()} Split to YOLO Format")
    print("="*60)
    
    # Prepare arguments for multiprocessing
    tasks = [(record, split_type, DATASET_DIR) for record in data_split]
    
    # Use half the CPU cores to avoid overwhelming the system
    num_workers = max(1, cpu_count() // 2)
    
    print(f"Processing with {num_workers} workers...")
    
    total_annotations = 0
    skipped_images = 0
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_record, tasks),
            total=len(tasks),
            desc=f"Processing {split_type} images"
        ))
    
    # Process results
    for success, error, ann_count in results:
        if success:
            total_annotations += ann_count
        else:
            skipped_images += 1
            if error:
                print(f"⚠ {error}")
    
    if skipped_images > 0:
        print(f"⚠ Skipped {skipped_images} images (not found or errors)")
    
    return len(data_split) - skipped_images, total_annotations

def create_dataset_directories():
    """Create the required directory structure for YOLO dataset."""
    print("\nCreating dataset directories...")
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            dir_path = os.path.join(DATASET_DIR, f'{subdir}/{split}')
            os.makedirs(dir_path, exist_ok=True)
    print("✓ Directories created\n")

def verify_dataset():
    """Verify the created dataset."""
    print("\n" + "="*60)
    print("Verifying Dataset")
    print("="*60)
    
    train_img_count = len(os.listdir(os.path.join(DATASET_DIR, 'images/train')))
    train_lbl_count = len(os.listdir(os.path.join(DATASET_DIR, 'labels/train')))
    val_img_count = len(os.listdir(os.path.join(DATASET_DIR, 'images/val')))
    val_lbl_count = len(os.listdir(os.path.join(DATASET_DIR, 'labels/val')))
    
    print(f"\nDataset on disk:")
    print(f"  Training   - Images: {train_img_count}, Labels: {train_lbl_count}")
    print(f"  Validation - Images: {val_img_count}, Labels: {val_lbl_count}")
    
    # Check for mismatches
    issues = []
    if train_img_count != train_lbl_count:
        issues.append(f"Training image/label mismatch: {train_img_count} vs {train_lbl_count}")
    if val_img_count != val_lbl_count:
        issues.append(f"Validation image/label mismatch: {val_img_count} vs {val_lbl_count}")
    
    if issues:
        print("\n⚠ Verification issues found:")
        for issue in issues:
            print(f"  - {issue}")
        raise ValueError("Dataset verification failed!")
    
    print("\n✓ Verification successful: All images have corresponding labels")
    
    return {
        "train_images": train_img_count,
        "train_labels": train_lbl_count,
        "val_images": val_img_count,
        "val_labels": val_lbl_count
    }

# ===================================================================
# Main Execution
# ===================================================================

def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("# Pipeline Step 3: Split and Prepare YOLO Dataset")
    print("#"*60)
    
    # Check if valid work exists
    work_is_complete = check_partial_work()
    
    if work_is_complete:
        # Valid completed work found - skip execution
        print("\n✓ Step 3 already completed!")
        checkpoint = load_checkpoint("s3")
        print(f"  Timestamp: {checkpoint['timestamp']}")
        print(f"  Training images: {checkpoint['outputs']['train_images']}")
        print(f"  Validation images: {checkpoint['outputs']['val_images']}")
        print("\nSkipping to next step...")
        return
    
    # No valid work found or partial work cleaned - proceed with execution
    print("\n" + "="*60)
    print("Starting Fresh Dataset Creation")
    print("="*60)
    
    # Check prerequisites (now returns empty list instead of raising error)
    data_records = check_prerequisites()
    
    # Check if we have any data to process
    if not data_records:
        print("\n" + "="*60)
        print("⚠ No data records available!")
        print("="*60)
        print("\nCannot proceed without data records.")
        print("Please run: python s1_2_data_preparation.py")
        print("\nExiting gracefully...")
        return
    
    # Create directories
    create_dataset_directories()
    
    # Split data
    train_data, val_data = split_data(data_records)
    
    # Process training split
    train_count, train_ann_count = process_dataset_split_segmentation(train_data, 'train')
    print(f"\n✓ Training split complete:")
    print(f"  - Images: {train_count}")
    print(f"  - Annotations: {train_ann_count}")
    
    # Process validation split
    val_count, val_ann_count = process_dataset_split_segmentation(val_data, 'val')
    print(f"\n✓ Validation split complete:")
    print(f"  - Images: {val_count}")
    print(f"  - Annotations: {val_ann_count}")
    
    # Verify dataset
    stats = verify_dataset()
    
    # Save checkpoint
    save_checkpoint("s3", outputs={
        "train_images": stats["train_images"],
        "train_labels": stats["train_labels"],
        "train_annotations": train_ann_count,
        "val_images": stats["val_images"],
        "val_labels": stats["val_labels"],
        "val_annotations": val_ann_count,
        "dataset_dir": DATASET_DIR
    })
    
    print("\n" + "="*60)
    print("✓ Step 3 Complete!")
    print("="*60)
    print(f"\nYOLO Dataset created at: {DATASET_DIR}")
    print(f"Training:   {stats['train_images']} images, {train_ann_count} annotations")
    print(f"Validation: {stats['val_images']} images, {val_ann_count} annotations")
    print(f"\nReady for Step 4: Creating YAML configuration file")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error in Step 3: {e}")
        import traceback
        traceback.print_exc()
        exit(1)