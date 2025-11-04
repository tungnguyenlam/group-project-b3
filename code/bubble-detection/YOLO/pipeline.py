#!/usr/bin/env python3
"""
YOLOv8 Balloon Detection Training Pipeline
===========================================
Master orchestrator script that runs all pipeline steps sequentially:
  1-2. Data Preparation
  3.   Split and Prepare YOLO Dataset
  4.   Create YAML Configuration
  5.   Train Model
  6.   Evaluate Model

Usage:
  python pipeline.py                    # Run all steps
  python pipeline.py --force            # Force re-run all steps (ignore checkpoints)
  python pipeline.py --start-from 3     # Start from specific step
  python pipeline.py --steps 1 2 3 4    # Run specific steps only

Features:
  - Automatic checkpoint validation
  - Color-coded console output
  - Step timing and progress tracking
  - Graceful error handling
  - Summary report at the end
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import gc 

# ===================================================================
# Configuration
# ===================================================================

END_WITH_LOCAL = 'bubble-segmentation-final-deep-learning'

BASE_DIR = os.getcwd()

# Simple validation
if not (BASE_DIR.endswith('/content') or BASE_DIR.endswith(END_WITH_LOCAL)):
    print(f"❌ Error: Expected to be in .../{END_WITH_LOCAL} or .../content directory")
    print(f"   Current directory: {BASE_DIR}")
    sys.exit(1)

# Pipeline steps
PIPELINE_STEPS = [
    {
        "id": "1-2",
        "name": "Data Preparation",
        "script": "s1_2_data_preparation.py",
        "description": "Validate directories, load JSONs, filter balloon annotations"
    },
    {
        "id": "3",
        "name": "Split and Prepare Dataset",
        "script": "s3_split_prepare_yolo_dataset.py",
        "description": "Group by manga series, train/val split, YOLO format conversion"
    },
    {
        "id": "4",
        "name": "Create YAML Configuration",
        "script": "s4_create_yaml_config_file.py",
        "description": "Generate dataset.yaml configuration file"
    },
    {
        "id": "5",
        "name": "Train Model",
        "script": "s5_train_model.py",
        "description": "Train YOLOv8 segmentation model with device auto-detection"
    },
    {
        "id": "6",
        "name": "Evaluate Model",
        "script": "s6_eval_model.py",
        "description": "Run validation and generate comprehensive metrics report"
    }
]

# Directory containing step scripts
SCRIPT_DIR = os.path.join(BASE_DIR, 'code', 'bubble-detection', 'YOLO')

# ===================================================================
# ANSI Color Codes for Pretty Output
# ===================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a colored header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}\n")

def print_step_header(step_num, step_name):
    """Print a step header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─'*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}Step {step_num}: {step_name}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─'*70}{Colors.ENDC}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")

# ===================================================================
# Pipeline Execution Functions
# ===================================================================

def run_step(step, force=False):
    """
    Run a single pipeline step.
    
    Returns:
        tuple: (success: bool, duration: float, message: str)
    """
    script_path = os.path.join(SCRIPT_DIR, step['script'])
    
    if not os.path.exists(script_path):
        return False, 0, f"Script not found: {script_path}"
    
    print_info(f"Description: {step['description']}")
    print_info(f"Script: {step['script']}")
    
    # Build command
    cmd = [sys.executable, script_path]
    if force:
        cmd.append('--force')
    
    print(f"\nExecuting: {' '.join(cmd)}\n")
    
    # Run the step
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        duration = time.time() - start_time
        del result  # Clean up subprocess result
        del cmd  # Clean up command list
        return True, duration, "Completed successfully"
    
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        error_code = e.returncode
        del e  # Clean up exception object
        del cmd  # Clean up command list
        return False, duration, f"Failed with exit code {error_code}"
    
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        del e  # Clean up exception object
        del cmd  # Clean up command list
        return False, duration, f"Error: {error_msg}"

def format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def print_summary(results):
    """Print a summary of all pipeline steps."""
    print_header("PIPELINE EXECUTION SUMMARY")
    
    total_duration = sum(r['duration'] for r in results)
    success_count = sum(1 for r in results if r['success'])
    
    print(f"{'Step':<8} {'Status':<12} {'Duration':<12} {'Message':<40}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*40}")
    
    for result in results:
        status = f"{Colors.GREEN}✓ Success{Colors.ENDC}" if result['success'] else f"{Colors.RED}✗ Failed{Colors.ENDC}"
        duration = format_duration(result['duration'])
        print(f"{result['step_id']:<8} {status:<21} {duration:<12} {result['message']:<40}")
    
    print(f"\n{'-'*70}")
    print(f"Total Steps: {len(results)}")
    print(f"Successful: {Colors.GREEN}{success_count}{Colors.ENDC}")
    print(f"Failed: {Colors.RED}{len(results) - success_count}{Colors.ENDC}")
    print(f"Total Duration: {Colors.BOLD}{format_duration(total_duration)}{Colors.ENDC}")
    
    if success_count == len(results):
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All pipeline steps completed successfully!{Colors.ENDC}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Pipeline completed with errors{Colors.ENDC}")
    
    # Clean up temporary variables
    del total_duration, success_count

# ===================================================================
# Main Pipeline Orchestration
# ===================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Balloon Detection Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                    # Run all steps
  python pipeline.py --force            # Force re-run all steps
  python pipeline.py --start-from 3     # Start from step 3
  python pipeline.py --steps 1 2 3      # Run specific steps only
        """
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run all steps (ignore checkpoints)'
    )
    
    parser.add_argument(
        '--start-from',
        type=int,
        metavar='N',
        help='Start from step N (1-6)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        metavar='N',
        help='Run specific steps only (e.g., --steps 1 2 3)'
    )
    
    return parser.parse_args()

def main():
    """Main pipeline orchestration function."""
    args = parse_arguments()
    
    # Print banner
    print_header("YOLOv8 Balloon Detection Training Pipeline")
    print(f"Working Directory: {Colors.BOLD}{BASE_DIR}{Colors.ENDC}")
    print(f"Start Time: {Colors.BOLD}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    
    # Determine which steps to run
    if args.steps:
        # Run specific steps
        steps_to_run = [step for step in PIPELINE_STEPS if int(step['id'].split('-')[0]) in args.steps]
        print_info(f"Running specific steps: {', '.join(str(s) for s in args.steps)}")
    elif args.start_from:
        # Start from specific step
        steps_to_run = [step for step in PIPELINE_STEPS if int(step['id'].split('-')[0]) >= args.start_from]
        print_info(f"Starting from step {args.start_from}")
    else:
        # Run all steps
        steps_to_run = PIPELINE_STEPS
        print_info("Running all pipeline steps")
    
    if args.force:
        print_warning("Force mode enabled - all checkpoints will be ignored")
    
    # Clean up args after extracting needed info
    force_mode = args.force
    del args
    
    # Execute pipeline
    results = []
    pipeline_start_time = time.time()
    
    for i, step in enumerate(steps_to_run, 1):
        print_step_header(f"{i}/{len(steps_to_run)} (Step {step['id']})", step['name'])
        
        success, duration, message = run_step(step, force=force_mode)
        
        results.append({
            'step_id': step['id'],
            'step_name': step['name'],
            'success': success,
            'duration': duration,
            'message': message
        })
        
        # Clean up temporary variables for this iteration
        del success, duration, message
        
        if results[-1]['success']:
            print_success(f"Step {step['id']} completed in {format_duration(results[-1]['duration'])}")
        else:
            print_error(f"Step {step['id']} failed: {results[-1]['message']}")
            print_error("Pipeline stopped due to error")
            break
    
    # Clean up loop variables and temporary data
    del steps_to_run, force_mode, pipeline_start_time
    
    # Print summary
    print_summary(results)
    
    # Determine exit status and clean up
    all_success = all(r['success'] for r in results)
    del results
    gc.collect()  # Force garbage collection before exit
    
    # Exit with appropriate code
    if all_success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
