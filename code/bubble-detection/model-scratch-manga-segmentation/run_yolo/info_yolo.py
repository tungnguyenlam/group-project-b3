import argparse
import sys 
from collections import defaultdict
from architecture_yolo.yolo_seg import YOLOSeg

def print_model_info(model, output_stream=sys.stdout):

    print("--- Model Architecture ---", file=output_stream)
    print(model, file=output_stream)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n--- Parameter Count ---", file=output_stream)
    print(f"Total Parameters: {total_params:,}", file=output_stream)
    print(f"Trainable Parameters: {trainable_params:,}", file=output_stream)
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}", file=output_stream)

    
    layer_counts = defaultdict(int)

    for module in model.modules():
        layer_name = module.__class__.__name__
        layer_counts[layer_name] += 1
        

    print("\n--- Layer Count Breakdown ---", file=output_stream)
    print("Detailed breakdown of nn.Module components:", file=output_stream)
    for layer_type, count in layer_counts.items():
        if layer_type not in ["YOLOSeg", "Sequential", "ModuleList"]:
            print(f"  - {layer_type}: {count}", file=output_stream)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print YOLO-Seg model architecture and parameter count.")
    parser.add_argument(
        "--num-classes", 
        type=int, 
        default=2, 
        help="Number of classes (including background). Default is 2."
    )
 
    parser.add_argument(
        "--output-file",
        type=str,
        default="model_yolo_architecture.txt",
        help="File path to save the model architecture information."
    )
    args = parser.parse_args()

    # Create model
    model = YOLOSeg(num_classes=args.num_classes)
    
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            print("\n" + "="*60, file=f)
            print("YOLOv11n-seg (from scratch) MODEL INFORMATION", file=f)
            print("="*60, file=f)

            print_model_info(model, output_stream=f)
            
            print("="*60 + "\n", file=f)
        
        print(f"Model architecture and info successfully saved to '{args.output_file}'")

    except IOError as e:
        print(f"Error writing to file {args.output_file}: {e}")