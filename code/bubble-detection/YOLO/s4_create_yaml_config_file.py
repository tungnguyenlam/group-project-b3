#!/usr/bin/env python3
"""
Step 4: Create YAML Configuration File
=======================================
This script handles:
- Validating that Step 3 was completed successfully
- Creating dataset.yaml configuration for YOLOv8
- Validating the configuration file

Prerequisites:
- Step 3 must be completed successfully
- YOLOv8_data directory structure must exist

Outputs:
- data/YOLOv8_data/dataset.yaml: Configuration file
- .pipeline_state/s4_complete.json: Checkpoint file
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime

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
YAML_PATH = os.path.join(DATASET_DIR, 'dataset.yaml')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'code', 'bubble-detection', 'YOLO', '.pipeline_state')

# Target category
TARGET_CATEGORY_NAME = "balloon"

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
    """Check if Step 3 was completed successfully."""
    print("\n" + "="*60)
    print("Checking Prerequisites")
    print("="*60)
    
    # Check for Step 3 checkpoint
    checkpoint = load_checkpoint("s3")
    if not checkpoint:
        raise RuntimeError(
            "❌ Step 3 has not been completed!\n"
            "Please run: python s3_split_prepare_yolo_dataset.py"
        )
    
    print("✓ Step 3 checkpoint found")
    print(f"  Timestamp: {checkpoint['timestamp']}")
    print(f"  Training images: {checkpoint['outputs']['train_images']}")
    print(f"  Validation images: {checkpoint['outputs']['val_images']}")
    
    # Check dataset directories exist and contain files
    required_dirs = [
        os.path.join(DATASET_DIR, 'images/train'),
        os.path.join(DATASET_DIR, 'images/val'),
        os.path.join(DATASET_DIR, 'labels/train'),
        os.path.join(DATASET_DIR, 'labels/val')
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"❌ Required directory not found: {dir_path}")
        
        if not os.listdir(dir_path):
            raise ValueError(f"❌ Directory is empty: {dir_path}")
        
        print(f"✓ Found: {dir_path} ({len(os.listdir(dir_path))} files)")
    
    print("\n✓ All prerequisites satisfied\n")
    
    return checkpoint

def validate_yaml_file():
    """Validate the existing YAML file if it exists."""
    if not os.path.exists(YAML_PATH):
        return False
    
    try:
        with open(YAML_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'names']
        for field in required_fields:
            if field not in config:
                print(f"⚠ YAML missing required field: {field}")
                return False
        
        # Validate paths exist
        base_path = config.get('path')
        if not os.path.exists(base_path):
            print(f"⚠ YAML path doesn't exist: {base_path}")
            return False
        
        # Validate train/val paths
        train_path = os.path.join(base_path, config.get('train', ''))
        val_path = os.path.join(base_path, config.get('val', ''))
        
        if not os.path.exists(train_path):
            print(f"⚠ Training path doesn't exist: {train_path}")
            return False
        
        if not os.path.exists(val_path):
            print(f"⚠ Validation path doesn't exist: {val_path}")
            return False
        
        # Validate class names
        if not isinstance(config.get('names'), dict):
            print("⚠ YAML 'names' field is not a dictionary")
            return False
        
        print("✓ Existing YAML file is valid")
        return True
        
    except Exception as e:
        print(f"⚠ Error validating YAML: {e}")
        return False

# ===================================================================
# Main Functions
# ===================================================================

def create_yaml_config():
    """Create the dataset.yaml configuration file for YOLOv8."""
    print("\n" + "="*60)
    print("STEP 4: Creating dataset.yaml Configuration")
    print("="*60)
    
    # Create configuration dictionary
    dataset_config = {
        'path': os.path.abspath(DATASET_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: TARGET_CATEGORY_NAME
        }
    }
    
    # Save YAML file
    print(f"\nWriting configuration to: {YAML_PATH}")
    with open(YAML_PATH, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    print("✓ YAML file created successfully")
    
    # Display content
    print("\n" + "-"*60)
    print("YAML Content:")
    print("-"*60)
    print(yaml.dump(dataset_config, default_flow_style=False, sort_keys=False))
    print("-"*60)
    
    return dataset_config

def verify_yaml_config(config):
    """Verify the created YAML configuration."""
    print("\n" + "="*60)
    print("Verifying YAML Configuration")
    print("="*60)
    
    # Check all paths are accessible
    base_path = config['path']
    train_path = os.path.join(base_path, config['train'])
    val_path = os.path.join(base_path, config['val'])
    
    checks = [
        ("Base path", base_path),
        ("Training images path", train_path),
        ("Validation images path", val_path),
    ]
    
    for name, path in checks:
        if os.path.exists(path):
            file_count = len(os.listdir(path)) if os.path.isdir(path) else "N/A"
            print(f"✓ {name}: {path}")
            if file_count != "N/A":
                print(f"  Files: {file_count}")
        else:
            raise FileNotFoundError(f"❌ {name} not found: {path}")
    
    # Verify class names
    print(f"\n✓ Class configuration:")
    for class_id, class_name in config['names'].items():
        print(f"  - Class {class_id}: {class_name}")
    
    print("\n✓ YAML configuration verified successfully")

# ===================================================================
# Main Execution
# ===================================================================

def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("# Pipeline Step 4: Create YAML Configuration File")
    print("#"*60)
    
    # Check if work is already done
    checkpoint = load_checkpoint("s4")
    if checkpoint and validate_yaml_file():
        print("\n✓ Step 4 already completed!")
        print(f"  Timestamp: {checkpoint['timestamp']}")
        print(f"  YAML file: {checkpoint['outputs']['yaml_path']}")
        print("\nSkipping to next step...")
        return
    
    if checkpoint and not validate_yaml_file():
        print("\n⚠ Checkpoint exists but YAML file is invalid or missing")
        print("Regenerating YAML file...\n")
    
    # Check prerequisites
    s3_checkpoint = check_prerequisites()
    
    # Create YAML configuration
    config = create_yaml_config()
    
    # Verify configuration
    verify_yaml_config(config)
    
    # Save checkpoint
    save_checkpoint("s4", outputs={
        "yaml_path": YAML_PATH,
        "dataset_dir": DATASET_DIR,
        "class_name": TARGET_CATEGORY_NAME,
        "train_images": s3_checkpoint['outputs']['train_images'],
        "val_images": s3_checkpoint['outputs']['val_images']
    })
    
    print("\n" + "="*60)
    print("✓ Step 4 Complete!")
    print("="*60)
    print(f"\nYAML configuration saved to: {YAML_PATH}")
    print(f"Dataset path: {config['path']}")
    print(f"Class: {TARGET_CATEGORY_NAME}")
    print(f"\nReady for Step 5: Model training")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error in Step 4: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
