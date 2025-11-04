#!/usr/bin/env python3
"""
Step 6: Evaluate YOLOv8 Model
==============================
This script handles:
- Validating that Step 5 (training) was completed successfully
- Loading the best trained model
- Running comprehensive validation on the test set
- Generating and displaying detailed metrics reports

Prerequisites:
- Step 5 must be completed successfully
- Trained model weights must exist

Outputs:
- Validation metrics (printed to console)
- Updated results in training directory with validation plots

Note: This step does NOT use checkpointing - it always runs fresh.
      This allows for re-evaluation after model updates.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

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
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'code', 'bubble-detection', 'YOLO', '.pipeline_state')
PROJECT_NAME = os.path.join(BASE_DIR, 'models', 'bubble-detection', 'YOLO')

# Dynamic run name helper
def get_latest_run_name(project_dir, base_name='balloon_seg_run'):
    """Find the latest run directory."""
    if not os.path.exists(project_dir):
        return None
    
    existing_runs = [d for d in os.listdir(project_dir) 
                     if os.path.isdir(os.path.join(project_dir, d)) and d.startswith(base_name)]
    
    if not existing_runs:
        return None
    
    # Extract numbers and find the latest
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run.replace(base_name, ''))
            run_numbers.append((num, run))
        except ValueError:
            continue
    
    if not run_numbers:
        return None
    
    # Return the run with the highest number
    latest_run = max(run_numbers, key=lambda x: x[0])[1]
    return latest_run

# Get the latest run name
RUN_NAME = get_latest_run_name(PROJECT_NAME) or 'balloon_seg_run1'

# ===================================================================
# Utility Functions
# ===================================================================

def load_training_info():
    """Load training information from Step 5."""
    training_info_file = os.path.join(CHECKPOINT_DIR, 's5_training_info.json')
    
    if not os.path.exists(training_info_file):
        return None
    
    try:
        with open(training_info_file, 'r') as f:
            info = json.load(f)
        return info
    except Exception as e:
        print(f"⚠ Error loading training info: {e}")
        return None

def find_best_model():
    """Find the best model from training results."""
    # Try to get from training info first
    training_info = load_training_info()
    if training_info and 'training_info' in training_info:
        best_model_path = training_info['training_info'].get('best_model')
        if best_model_path and os.path.exists(best_model_path):
            return best_model_path
    
    # Fallback: search in expected location
    expected_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')
    if os.path.exists(expected_path):
        return expected_path
    
    # Last resort: search for any best.pt in project directory
    if os.path.exists(PROJECT_NAME):
        for root, dirs, files in os.walk(PROJECT_NAME):
            if 'best.pt' in files:
                return os.path.join(root, 'best.pt')
    
    return None

# ===================================================================
# Validation Functions
# ===================================================================

def check_prerequisites():
    """Check if Step 5 was completed successfully and model exists."""
    print("\n" + "="*60)
    print("Checking Prerequisites")
    print("="*60)
    
    # Check for training info
    training_info = load_training_info()
    if not training_info:
        print("⚠ No training info found from Step 5")
        print("  Looking for trained model in default location...")
    else:
        print("✓ Training info found")
        print(f"  Training completed: {training_info['timestamp']}")
        print(f"  Device used: {training_info['device']}")
        print(f"  Epochs: {training_info['epochs']}")
    
    # Find best model
    best_model_path = find_best_model()
    if not best_model_path:
        raise FileNotFoundError(
            "❌ Trained model not found!\n"
            "Please run Step 5 first: python s5_train_model.py\n"
            f"Expected location: {os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')}"
        )
    
    print(f"✓ Best model found: {best_model_path}")
    print(f"✓ Using run: {RUN_NAME}")
    
    # Check model file size
    model_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    print("\n✓ All prerequisites satisfied\n")
    
    return best_model_path

# ===================================================================
# Evaluation Functions
# ===================================================================

def evaluate_model(model_path):
    """Run comprehensive evaluation on the trained model."""
    print("\n" + "="*60)
    print("STEP 6: Evaluating Model Performance")
    print("="*60)
    
    # Load the best model
    print(f"\nLoading model from: {model_path}")
    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")
    
    # Run validation
    print("\n" + "="*60)
    print("Running Validation on Test Set...")
    print("="*60)
    print("\nThis may take a few minutes...\n")
    
    try:
        metrics = model.val(
            split='val',
            project=PROJECT_NAME,
            name=RUN_NAME,
            exist_ok=True,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("✓ Validation Completed Successfully!")
        print("="*60)
        
        return metrics
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Validation Failed!")
        print("="*60)
        raise RuntimeError(f"Validation error: {e}")

def print_metrics_report(metrics):
    """Print a comprehensive, well-formatted metrics report."""
    print("\n" + "#"*60)
    print("# COMPREHENSIVE EVALUATION REPORT")
    print("#"*60)
    
    if not hasattr(metrics, 'results_dict') or not metrics.results_dict:
        print("\n⚠ No metrics available to display")
        return
    
    print(f"\nValidation results saved to: {metrics.save_dir}\n")
    
    # Group metrics by category
    box_metrics = {}
    mask_metrics = {}
    other_metrics = {}
    
    for key, value in metrics.results_dict.items():
        # Clean the key
        clean_key = key.replace('metrics/', '').strip()
        
        # Categorize
        if '(B)' in clean_key:
            final_key = clean_key.replace('(B)', '').strip()
            box_metrics[final_key] = value
        elif '(M)' in clean_key:
            final_key = clean_key.replace('(M)', '').strip()
            mask_metrics[final_key] = value
        else:
            other_metrics[clean_key] = value
    
    # Print function
    def print_metric_group(title, metric_dict, emoji="📊"):
        print(f"\n{emoji} {title}")
        print("-" * 60)
        if not metric_dict:
            print("  (No metrics found)")
            return
        
        for key in sorted(metric_dict.keys()):
            value = metric_dict[key]
            # Format value based on magnitude
            if isinstance(value, (int, float)):
                if value >= 1000:
                    formatted_value = f"{value:,.2f}"
                elif value < 0.01 and value != 0:
                    formatted_value = f"{value:.6f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            print(f"  • {key:<20}: {formatted_value}")
    
    # Print grouped metrics
    print_metric_group("Bounding Box Detection Metrics", box_metrics, "📦")
    print_metric_group("Instance Segmentation Metrics", mask_metrics, "🎭")
    print_metric_group("Other Metrics", other_metrics, "📈")
    
    # Summary of key metrics
    print("\n" + "="*60)
    print("KEY PERFORMANCE INDICATORS")
    print("="*60)
    
    key_metrics = {
        "Box mAP@50": box_metrics.get("mAP50", "N/A"),
        "Box mAP@50-95": box_metrics.get("mAP50-95", "N/A"),
        "Mask mAP@50": mask_metrics.get("mAP50", "N/A"),
        "Mask mAP@50-95": mask_metrics.get("mAP50-95", "N/A"),
        "Precision (Mask)": mask_metrics.get("precision", "N/A"),
        "Recall (Mask)": mask_metrics.get("recall", "N/A")
    }
    
    for metric_name, value in key_metrics.items():
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}"
            # Add performance indicator
            if "mAP" in metric_name or "precision" in metric_name or "Recall" in metric_name:
                if value >= 0.9:
                    indicator = "🟢 Excellent"
                elif value >= 0.7:
                    indicator = "🟡 Good"
                elif value >= 0.5:
                    indicator = "🟠 Fair"
                else:
                    indicator = "🔴 Needs Improvement"
            else:
                indicator = ""
        else:
            formatted_value = str(value)
            indicator = ""
        
        print(f"  • {metric_name:<20}: {formatted_value:>10}  {indicator}")
    
    print("\n" + "#"*60)

# ===================================================================
# Main Execution
# ===================================================================

def main():
    """Main execution function."""
    print("\n" + "#"*60)
    print("# Pipeline Step 6: Evaluate YOLOv8 Model")
    print("#"*60)
    
    print("\nNote: This step always runs fresh (no checkpointing).")
    print("This allows re-evaluation after model updates.\n")
    
    # Check prerequisites and get model path
    model_path = check_prerequisites()
    
    # Evaluate model
    metrics = evaluate_model(model_path)
    
    # Print comprehensive report
    print_metrics_report(metrics)
    
    print("\n" + "="*60)
    print("✓ Step 6 Complete!")
    print("="*60)
    print("\n🎉 Full pipeline completed successfully!")
    print("\nAll training and evaluation results are available in:")
    print(f"  {os.path.join(PROJECT_NAME, RUN_NAME)}")
    print("\nKey files:")
    print("  • weights/best.pt       - Best model weights")
    print("  • results.csv           - Training metrics over time")
    print("  • results.png           - Training curves visualization")
    print("  • confusion_matrix.png  - Classification performance")
    print("  • *_curve.png          - Precision/Recall curves")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error in Step 6: {e}")
        import traceback
        traceback.print_exc()
        exit(1)