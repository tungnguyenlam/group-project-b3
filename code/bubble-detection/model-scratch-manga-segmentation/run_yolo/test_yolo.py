import torch
from torch.amp import autocast
import argparse
import os
import sys
import json
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.data_loader_factory import DataLoaderFactory
from architecture_yolo.yolo_seg import YOLOSeg
from post_process_yolo.processor import PostProcessorV2

def build_coco_ground_truth(data_loader):
    """Constructs a COCO ground truth object from the dataset records."""
    coco_gt = COCO()
    images, annotations = [], []
    ann_id = 1
    
    print("Building COCO ground truth annotations...")
    for record in tqdm(data_loader.dataset.records, desc="Processing GT"):
        image_id = record['image_id']
        
        images.append({
            'id': int(image_id),
            'file_name': record['file_name_for_debug'],
            'height': record['height'],
            'width': record['width']
        })
        
        for ann in record['annotations']:
            bbox = ann['bbox']
            h, w = record['height'], record['width']
            segmentation = ann['segmentation']
            
            try:
                if isinstance(segmentation, dict) and 'counts' in segmentation:
                    rle = segmentation
                elif isinstance(segmentation, str):
                    rle = {'counts': segmentation.encode('utf-8'), 'size': [h, w]}
                elif isinstance(segmentation, list):
                    if not segmentation or not segmentation[0]: continue
                    if not isinstance(segmentation[0], list): segmentation = [segmentation]
                    rles = mask_util.frPyObjects(segmentation, h, w)
                    rle = mask_util.merge(rles)
                else: continue
                
                area = float(mask_util.area(rle))
                
                annotations.append({
                    'id': ann_id, 'image_id': int(image_id), 'category_id': 1,
                    'bbox': [float(x) for x in bbox], 'area': area,
                    'segmentation': rle, 'iscrowd': 0
                })
                ann_id += 1
            except Exception as e:
                print(f"Warning: Could not process annotation in {record['file_name_for_debug']}: {e}")
                continue
    
    dataset_info = {'info': {}, 'images': images, 'annotations': annotations, 
                    'categories': [{'id': 1, 'name': 'text_bubble'}]}
    coco_gt.dataset = dataset_info
    coco_gt.createIndex()
    
    print(f"Built COCO GT with {len(images)} images and {len(annotations)} annotations")
    return coco_gt

@torch.inference_mode()
def evaluate_with_coco(model, data_loader, post_processor, device):
    """
    Memory-safe and GPU-accelerated evaluation function.
    It processes each image in the batch sequentially after inference
    to avoid storing large, upsampled mask tensors in VRAM.
    """
    model.eval()
    
    coco_gt = build_coco_ground_truth(data_loader)
    all_coco_results = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")
    
    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]

        # 1. Run model inference on the GPU for the whole batch
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            preds = model(images)
            input_shape = images.shape[-2:]
            results = post_processor(preds, input_shape)

        # 2. Process each image's results sequentially to manage memory
        for i in range(batch_size):
            res = results[i]
            tgt = targets[i]
            num_preds = res['boxes'].shape[0]

            if num_preds == 0:
                continue

            # 3. Get scaling info for the current image
            orig_size = tgt['orig_size'].cpu().numpy()
            orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
            size_after_resize = tgt['size'].cpu().numpy()
            w_after_resize, h_after_resize = int(size_after_resize[0]), int(size_after_resize[1])

            # 4. Perform cropping and scaling for boxes on GPU
            res['boxes'][:, [0, 2]] = res['boxes'][:, [0, 2]].clamp(0, w_after_resize)
            res['boxes'][:, [1, 3]] = res['boxes'][:, [1, 3]].clamp(0, h_after_resize)
            
            scale_w = orig_w / w_after_resize
            scale_h = orig_h / h_after_resize
            scale_tensor = torch.tensor([scale_w, scale_h, scale_w, scale_h], device=device)
            scaled_boxes = res['boxes'] * scale_tensor
            
            # Move boxes and scores to CPU early
            scaled_boxes_cpu = scaled_boxes.cpu().numpy()
            scores_cpu = res['scores'].cpu().numpy()
            image_id = tgt['image_id'].item()
            
            # 5. Crop and upscale masks in small chunks on GPU, then immediately convert to RLE on CPU
            cropped_masks = res['masks'][:, :h_after_resize, :w_after_resize]
            
            resize_batch_size = args.resize_batch_size  # Process N masks at a time
            for start_idx in range(0, num_preds, resize_batch_size):
                end_idx = min(start_idx + resize_batch_size, num_preds)
                
                # Upsample a small chunk of masks on GPU
                mask_chunk_gpu = torch.nn.functional.interpolate(
                    cropped_masks[start_idx:end_idx].unsqueeze(1).float(),
                    size=(orig_h, orig_w),
                    mode='nearest'
                ).squeeze(1)
                
                # Binarize on GPU, then move to CPU and convert to numpy
                mask_chunk_np = (mask_chunk_gpu > 0.5).cpu().numpy().astype(np.uint8)

                # Convert to RLE and format for COCO on CPU
                for j in range(mask_chunk_np.shape[0]):
                    global_idx = start_idx + j
                    
                    x1, y1, x2, y2 = scaled_boxes_cpu[global_idx]
                    bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    
                    rle = mask_util.encode(np.asfortranarray(mask_chunk_np[j]))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    
                    all_coco_results.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox_coco,
                        "score": float(scores_cpu[global_idx]),
                        "segmentation": rle,
                    })

        progress_bar.set_postfix(predictions=len(all_coco_results))

    # Clean up memory after the loop
    del results, preds, images
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if not all_coco_results:
        print("\nWarning: No predictions were generated. Cannot evaluate.")
        return None

    # === Run COCO Evaluation ===
    print("\n" + "="*60)
    print("Running COCO Evaluation...")
    print("="*60)
    
    coco_dt = coco_gt.loadRes(all_coco_results)
    
    results_dict = {}
    
    for iou_type in ["bbox", "segm"]:
        print(f"\n--- Evaluating IoU type: {iou_type} ---")
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        results_dict[iou_type] = {'AP': stats[0], 'AP50': stats[1], 'AP75': stats[2],
                                'APs': stats[3], 'APm': stats[4], 'APl': stats[5]}
    
    return results_dict


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("YOLO-Seg Evaluation using pycocotools")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if not os.path.exists(args.model_path):
        print(f"\nERROR: Model file not found at {args.model_path}")
        return
    
    print(f"Model path: {args.model_path}")
    print(f"Confidence threshold: {args.conf_thresh}")
    print(f"NMS threshold: {args.nms_thresh}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mask batch size: {args.mask_batch_size}")
    print(f"{'='*60}\n")
    
    loader_factory = DataLoaderFactory(json_dir=args.json_dir, image_root=args.image_root)
    val_loader = loader_factory.create_data_loader(
        split='validation', batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    model = YOLOSeg(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model weights loaded successfully")
    
    post_processor = PostProcessorV2(
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        mask_batch_size=args.mask_batch_size
    )
    
    results_dict = evaluate_with_coco(model, val_loader, post_processor, device)
    
    if results_dict and args.save_results:
        output_path = os.path.splitext(args.model_path)[0] + '_coco_eval.json'
        
        save_data = {
            'model_path': args.model_path,
            'config': vars(args),
            'metrics': results_dict,
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nEvaluation metrics saved to: {output_path}")


if __name__ == "__main__":
    cwd = os.getcwd()
    if not cwd.endswith('bubble-segmentation-final-deep-learning'):
        raise ValueError('To run this you should be in the bubble-segmentation-final-deep-learning directory')
    JSON_DIR = os.path.join(cwd, 'data', 'MangaSegmentation', 'jsons_processed')
    IMAGE_ROOT = os.path.join(cwd, 'data', 'Manga109_released_2023_12_07', 'images')
    MODEL_DIR = os.path.join(cwd, 'models', 'bubble-detection', 'model-scratch-manga-segmentation', 'yolo_outputs')
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, 'yolo_best.pth')
    parser = argparse.ArgumentParser(description="YOLO-Seg Evaluation with COCOeval")
    parser.add_argument("--json-dir", type=str, default=JSON_DIR)
    parser.add_argument("--image-root", type=str, default=IMAGE_ROOT)
    parser.add_argument("--model-path", type=str, help="Path to the trained .pth model file.", default=MODEL_PATH)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="Reduce this if you encounter Out-Of-Memory errors.")
    parser.add_argument("--resize-batch-size", type=int, default=16, help="Process N mask at a time for resizing mask")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--conf-thresh", type=float, default=0.98, help="Confidence threshold for post-processing.")
    parser.add_argument("--nms-thresh", type=float, default=0.9, help="NMS IoU threshold for post-processing.")
    parser.add_argument("--mask-batch-size", type=int, default=4, help="Batch size for mask processing during inference (critical for VRAM management).")
    parser.add_argument("--save-results", action='store_true', help="Save evaluation metrics to a JSON file.")
    
    args = parser.parse_args()
    main(args)