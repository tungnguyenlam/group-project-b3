import torch
import torch.optim as optim
import argparse
import os
import sys
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
import torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline.data_loader_factory import DataLoaderFactory, get_transforms
from architecture_yolo.yolo_seg import YOLOSeg
from post_process_yolo.evaluation_metric import MetricsCalculator
from post_process_yolo.processor import PostProcessorV2
from post_process_yolo.loss import YOLOv11Loss
from post_process_yolo.debug import debug_predictions_vs_gt, check_assigner_outputs, verify_decoding

import time
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


@torch.inference_mode()
def evaluate_yolo_custom(model, data_loader, post_processor, device, args):
    """
    Fixed evaluation function with proper coordinate transformation.
    """
    model.eval()
    
    print("\n" + "="*60)
    print("Starting Custom Evaluation")
    print("="*60)
    
    # Initialize metrics calculators
    bbox_metrics = MetricsCalculator(device=device, conf_threshold=args.conf_thresh)
    segm_metrics = MetricsCalculator(device=device, conf_threshold=args.conf_thresh)

    y_true = []
    y_pred = []
    
    total_images = 0
    total_predictions = 0
    total_gts = 0
    
    progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")
    
    for images, targets in progress_bar:
        # Move to device
        images = images.to(device, non_blocking=True)

        targets_on_device = []
        for t in targets:
            processed_target = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    processed_target[k] = v.to(device, non_blocking=True)
                else:
                    processed_target[k] = v 
            targets_on_device.append(processed_target)

        targets = targets_on_device
        
        # Forward pass
        use_amp = device.type == 'cuda'
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
            
            # Post-process at padded input size
            input_shape = images.shape[-2:]  # [H_pad, W_pad]
            results = post_processor(preds, input_shape)
        
        # ============================================================
        # Transform predictions to original image space
        # ============================================================
        for i in range(len(results)):
            # Get original image size (before resize and padding)
            orig_size = targets[i]['orig_size'].cpu().numpy()  # [W, H]
            orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
            
            # Get size after resize (before padding)
            size_after_resize = targets[i]['size'].cpu().numpy()  # [W, H]
            w_after_resize, h_after_resize = int(size_after_resize[0]), int(size_after_resize[1])
            
            # Step 1: Crop to remove padding
            if results[i]['boxes'].numel() > 0:
                # Clamp boxes to valid range after resize
                results[i]['boxes'][:, [0, 2]] = results[i]['boxes'][:, [0, 2]].clamp(0, w_after_resize)
                results[i]['boxes'][:, [1, 3]] = results[i]['boxes'][:, [1, 3]].clamp(0, h_after_resize)
            
            if results[i]['masks'].numel() > 0:
                # Crop masks to remove padding
                results[i]['masks'] = results[i]['masks'][:, :h_after_resize, :w_after_resize]
            
            # Step 2: Scale from resized size to original size
            if results[i]['boxes'].numel() > 0:
                scale_w = orig_w / w_after_resize
                scale_h = orig_h / h_after_resize
                
                scale_tensor = torch.tensor(
                    [scale_w, scale_h, scale_w, scale_h],
                    device=results[i]['boxes'].device,
                    dtype=torch.float32
                )
                results[i]['boxes'] = results[i]['boxes'] * scale_tensor
            
            if results[i]['masks'].numel() > 0:
                # Resize masks to original size
                masks_resized = torch.nn.functional.interpolate(
                    results[i]['masks'].unsqueeze(1).float(),
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                
                # Binarize
                results[i]['masks'] = (masks_resized > 0.5).to(torch.uint8)
            
            # Check if GT is already in original size
            gt_boxes = targets[i]['boxes']
            gt_masks = targets[i]['masks']
            
            # GT boxes should already be in resized coordinates from transform
            # We need to scale them back to original
            if gt_boxes.numel() > 0:
                scale_w = orig_w / w_after_resize
                scale_h = orig_h / h_after_resize
                
                scale_tensor_gt = torch.tensor(
                    [scale_w, scale_h, scale_w, scale_h],
                    device=gt_boxes.device,
                    dtype=torch.float32
                )
                targets[i]['boxes'] = gt_boxes * scale_tensor_gt
            
            if gt_masks.numel() > 0:
                # Resize GT masks to original size
                gt_masks_resized = torch.nn.functional.interpolate(
                    gt_masks.unsqueeze(1).float(),
                    size=(orig_h, orig_w),
                    mode='nearest'  # Use nearest for GT to preserve exact boundaries
                ).squeeze(1)
                
                targets[i]['masks'] = gt_masks_resized.to(torch.uint8)


            # Gather info for Confusion matrix (only bbox)
            iou_threshold = 0.5 
        
            background_id = args.num_classes 

            gt_labels = targets[i]['labels']
            pred_labels = results[i]['labels']
            pred_boxes = results[i]['boxes']
            gt_boxes = targets[i]['boxes']

            if gt_labels.numel() == 0 or pred_labels.numel() == 0:
                continue
            
            # Use a boolean array to keep track of matched objects
            gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=device)
            pred_matched = torch.zeros(pred_boxes.shape[0], dtype=torch.bool, device=device)

            if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
                # Compute the IoU matrix between all predictions and the true labels
                iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)

                # Browse through each prediction to find a match
                for pred_idx in range(pred_boxes.shape[0]):
                    # Tìm nhãn thật có IoU cao nhất với dự đoán hiện tại
                    best_gt_iou, best_gt_idx = iou_matrix[pred_idx].max(0)

                    # If IoU > threshold and the real label is not matched
                    if best_gt_iou > iou_threshold and not gt_matched[best_gt_idx]:
                        # Check if the class matches
                        if pred_labels[pred_idx] == gt_labels[best_gt_idx]:
                            # This is a TRUE POSITIVE (TP)
                            gt_matched[best_gt_idx] = True
                            pred_matched[pred_idx] = True
                            y_true.append(gt_labels[best_gt_idx].item())
                            y_pred.append(pred_labels[pred_idx].item())

            # After finding all the matching pairs, handle the errors.
            # 1. FALSE POSITIVES: The predictions do not match any of the real labels
            unmatched_preds_indices = torch.where(~pred_matched)[0]
            for pred_idx in unmatched_preds_indices:
                # True label is background, but model predicts a certain class
                y_true.append(background_id)
                y_pred.append(pred_labels[pred_idx].item())

            # 2. FALSE NEGATIVES: The real labels are not predicted by any
            unmatched_gts_indices = torch.where(~gt_matched)[0]
            for gt_idx in unmatched_gts_indices:
                # True label is a certain class, but the model predicts the background (missing)
                y_true.append(gt_labels[gt_idx].item())
                y_pred.append(background_id)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Add to metrics (now both pred and GT are in original image space)
        bbox_metrics.add_batch(results, targets, iou_type='bbox')
        segm_metrics.add_batch(results, targets, iou_type='segm')
        
        # Update statistics
        total_images += len(images)
        total_predictions += sum(len(r['boxes']) for r in results)
        total_gts += sum(len(t['boxes']) for t in targets)
        
        progress_bar.set_postfix(
            images=total_images,
            preds=total_predictions,
            gts=total_gts,
            avg_pred=f"{total_predictions/max(total_images, 1):.1f}"
        )
    
    # Compute metrics
    print("\n" + "="*60)
    print("Computing Metrics...")
    print("="*60)
    
    bbox_results = bbox_metrics.compute_ap()
    segm_results = segm_metrics.compute_ap()
    
    # Print results
    print("\n" + "="*60)
    print("BBOX METRICS:")
    print("="*60)
    print(f"  mAP@0.5:0.95: {bbox_results['ap']:.4f}")
    print(f"  mAP@0.5:     {bbox_results['ap50']:.4f}")
    print(f"  mAP@0.75:    {bbox_results['ap75']:.4f}")
    print(f"  Precision:   {bbox_results['precision']:.4f}")
    print(f"  Recall:      {bbox_results['recall']:.4f}")
    print(f"  F1-Score:    {bbox_results['f1']:.4f}")
    
    print("\n" + "="*60)
    print("SEGMENTATION METRICS:")
    print("="*60)
    print(f"  mAP@0.5:0.95: {segm_results['ap']:.4f}")
    print(f"  mAP@0.5:     {segm_results['ap50']:.4f}")
    print(f"  mAP@0.75:    {segm_results['ap75']:.4f}")
    print(f"  Precision:   {segm_results['precision']:.4f}")
    print(f"  Recall:      {segm_results['recall']:.4f}")
    print(f"  F1-Score:    {segm_results['f1']:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"  Total images:       {total_images}")
    print(f"  Total predictions:  {total_predictions}")
    print(f"  Total ground truth: {total_gts}")
    print(f"  Avg predictions/image: {total_predictions/max(total_images, 1):.2f}")
    print("="*60 + "\n")
    
    return segm_results, bbox_results, (y_true, y_pred)


def save_confusion_matrix(y_true, y_pred, output_dir, num_classes, epoch):
    if not y_true or not y_pred:
        print("NOT have info for confusion matrix.")
        return

    # num_classes=2 means: background (0) + 1 object class (1)
    # num_classes_obj = num_classes - 1  # Number of real class object = 1
    background_id = num_classes  # 2
    
    # Real labels: [1] (class object) + [2] (background when miss)
    class_labels = list(range(1, num_classes))  # [1] 
    
    # Labels to plot (including background)
    plot_labels = [f'class_{i} (bubble)' for i in class_labels] + ['background']  
    # ['class_1 (bubble)', 'background']
    
    # Labels to calculate confusion matrix (including background)
    all_labels_ids = class_labels + [background_id]  # [1, 2]
    
    cm = confusion_matrix(y_true, y_pred, labels=all_labels_ids)
    
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=plot_labels, yticklabels=plot_labels)
    plt.title(f'Confusion Matrix - Epoch {epoch} (Un-normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_unnormalized_epoch_{epoch}.png'), dpi=300)
    plt.close()

    
    row_sums = cm.sum(axis=1)
    cm_normalized = np.zeros_like(cm, dtype=float) 
    non_zero_rows = row_sums > 0
    cm_normalized[non_zero_rows] = cm[non_zero_rows].astype(float) / row_sums[non_zero_rows, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=plot_labels, yticklabels=plot_labels)
    plt.title(f'Confusion Matrix - Epoch {epoch} (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_normalized_epoch_{epoch}.png'), dpi=300)
    plt.close()
    
    print(f"Saved confusion matrix for epoch {epoch} to {output_dir}")



# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_one_epoch(model, optimizer, data_loader, device, loss_fn, epoch_num, scheduler, scaler, accumulation_steps):
    model.train()
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num} Training")
    
    optimizer.zero_grad()
    running_losses = {'total': 0.0, 'cls': 0.0, 'box': 0.0, 'mask': 0.0, 'dfl': 0.0}
    num_batches = 0

    for i, (images, targets) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)

        use_amp = device.type == 'cuda'
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
            loss_dict = loss_fn(preds, targets)
            total_loss = loss_dict['total_loss'] / accumulation_steps
        
        scaler.scale(total_loss).backward()

        running_losses['total'] += loss_dict['total_loss'].item()
        running_losses['cls'] += loss_dict['loss_cls'].item()
        running_losses['box'] += loss_dict['loss_box'].item()
        running_losses['mask'] += loss_dict['loss_mask'].item()
        running_losses['dfl'] += loss_dict['loss_dfl'].item()
        num_batches += 1

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()

        # Update progress bar less frequently (every 10 iterations)
        if i % 10 == 0:
            if device.type == 'cuda':
                vram_allocated_mb = torch.cuda.memory_allocated(device=device) / (1024 * 1024)
                vram_peak_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
            else:
                vram_allocated_mb = vram_peak_mb = 0

            current_lr = optimizer.param_groups[0]['lr']
            
            # Display average loss
            avg_total = running_losses['total'] / max(1, num_batches)
            avg_cls = running_losses['cls'] / max(1, num_batches)
            avg_box = running_losses['box'] / max(1, num_batches)
            avg_mask = running_losses['mask'] / max(1, num_batches)
            
            progress_bar.set_postfix(
                loss=f"{avg_total:.4f}",
                cls=f"{avg_cls:.4f}",
                box=f"{avg_box:.4f}",
                mask=f"{avg_mask:.4f}",
                lr=f"{current_lr:.1e}",
                vram_mb=f"{vram_allocated_mb:.1f}/{vram_peak_mb:.1f}"
            )

    avg_losses = {k: v / max(1, num_batches) for k, v in running_losses.items()}

    return avg_losses


def main(args):
    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 for Ampere GPUs 
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    loader_factory = DataLoaderFactory(json_dir=args.json_dir, image_root=args.image_root)
    train_transforms = get_transforms(is_train=True)
    train_loader = loader_factory.create_data_loader(
        split='train', 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        transforms=train_transforms
    )

    val_transforms = get_transforms(is_train=False) 
    val_loader = loader_factory.create_data_loader(
        split='validation',
        batch_size=args.eval_batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        transforms=val_transforms
    )

    model = YOLOSeg(num_classes=args.num_classes).to(device)
    
    # Use channels_last memory format for better performance on newer GPUs
    if device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 7:
        model = model.to(memory_format=torch.channels_last)
        print("Using channels_last memory format")
    
    loss_fn = YOLOv11Loss(model, box_weight=args.box_weight, cls_weight=args.cls_weight, 
                          mask_weight=args.mask_weight, dfl_weight=args.dfl_weight)
    
    # Use fused AdamW for faster updates
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        fused=True if device.type == 'cuda' else False
    )

    scaler = GradScaler()
    
    accumulation_steps = args.accumulation_steps

    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    post_processor = PostProcessorV2(
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        mask_batch_size=args.mask_batch_size,
        max_det=50
    )
    
    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Eval Batch size: {args.eval_batch_size}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * accumulation_steps}")
    print(f"  Number of workers: {args.num_workers}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"{'='*60}\n")
    
    
    if args.debug: 
        print("\n" + "="*60)
        print("RUNNING DEBUG CHECKS")
        print("="*60)

        # Check 1: Visualize predictions
        print("\n1. Checking prediction coordinate alignment...")
        debug_predictions_vs_gt(model, val_loader, post_processor, device, num_samples=2)

        # Check 2: Check assigner
        print("\n2. Checking Task-Aligned Assigner...")
        check_assigner_outputs(model, train_loader, device, num_batches=3)

        # Check 3: Verify decoding
        print("\n3. Running critical verification...")
        verify_decoding(model, train_loader, device)

        input("\nPress Enter to continue with training, or Ctrl+C to exit...")

    best_map = 0.0

    # Create csv file
    csv_path = os.path.join(args.output_dir, "result.csv")
    csv_columns = [
        'epoch', 'training_time',
        'train_loss_total', 'train_loss_cls', 'train_loss_box', 'train_loss_mask', 'train_loss_dfl',
        'val_bbox_map_0.5:0.95', 'val_bbox_map_0.5', 'val_bbox_map_0.75', 'val_bbox_precision', 'val_bbox_recall', 'val_bbox_f1',
        'val_segm_map_0.5:0.95', 'val_segm_map_0.5', 'val_segm_map_0.75', 'val_segm_precision', 'val_segm_recall', 'val_segm_f1'
    ]

    results_df = pd.DataFrame(columns=csv_columns)
    results_df.to_csv(csv_path, index=False)

    print("\nStarting training and validation...")
    for epoch in range(1, args.num_epochs + 1):

        start_time = time.time()
        avg_losses = train_one_epoch(model, optimizer, train_loader, device, loss_fn, epoch, scheduler, scaler, accumulation_steps)
        training_time = time.time() - start_time

        # Create dictionary to save data of this epoch 
        log_data = {
            'epoch': epoch,
            'training_time': round(training_time, 2),
            'train_loss_total': avg_losses['total'],
            'train_loss_cls': avg_losses['cls'],
            'train_loss_box': avg_losses['box'],
            'train_loss_mask': avg_losses['mask'],
            'train_loss_dfl': avg_losses['dfl'],
        }

        if epoch % args.epoch_val == 0:
            print(f"\n--- Running validation for epoch {epoch} ---")
            
            segm_results, bbox_results, cm_data = evaluate_yolo_custom(model, val_loader, post_processor, device, args)
            save_confusion_matrix(cm_data[0], cm_data[1], args.output_dir, args.num_classes, epoch)

            # Update log
            log_data.update({
                'val_bbox_map_0.5:0.95': bbox_results['ap'],
                'val_bbox_map_0.5': bbox_results['ap50'],
                'val_bbox_map_0.75': bbox_results['ap75'],
                'val_bbox_precision': bbox_results['precision'],
                'val_bbox_recall': bbox_results['recall'],
                'val_bbox_f1': bbox_results['f1'],
                'val_segm_map_0.5:0.95': segm_results['ap'],
                'val_segm_map_0.5': segm_results['ap50'],
                'val_segm_map_0.75': segm_results['ap75'],
                'val_segm_precision': segm_results['precision'],
                'val_segm_recall': segm_results['recall'],
                'val_segm_f1': segm_results['f1']
            })

            current_map = segm_results['ap']

            if current_map > 0:
                print(f"Validation mAP@.5:.95 (segm): {current_map:.4f}")

                if current_map > best_map:
                    best_map = current_map
                    best_checkpoint_path = os.path.join(args.output_dir, "yolo_best.pth")
                    torch.save(model.state_dict(), best_checkpoint_path)
                    print(f"New best model saved to {best_checkpoint_path} with mAP: {best_map:.4f}")
        
        # Convert log_data -> DataFrame and write to file
        pd.DataFrame([log_data]).to_csv(csv_path, mode='a', header=False, index=False)

        # Clear cache periodically
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("\nTraining finished successfully!")
    print(f"All training results have been saved to: {csv_path}")


if __name__ == "__main__":
    cwd = os.getcwd()
    if not cwd.endswith('bubble-segmentation-final-deep-learning'):
        raise ValueError('To run this you should be in the bubble-segmentation-final-deep-learning directory')
    JSON_DIR = os.path.join(cwd, 'data', 'MangaSegmentation', 'jsons_processed')
    IMAGE_ROOT = os.path.join(cwd, 'data', 'Manga109_released_2023_12_07', 'images')
    OUTPUT_DIR = os.path.join(cwd, 'models', 'bubble-detection', 'model-scratch-manga-segmentation', 'yolo_outputs')
    parser = argparse.ArgumentParser(description="YOLO-Seg Training Script")
    parser.add_argument("--json-dir", type=str, default=JSON_DIR)
    parser.add_argument("--image-root", type=str, default=IMAGE_ROOT)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes (counted background)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Batch size for evaluation to save VRAM")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--box-weight", type=float, default=7.0, help="For loss of Bounding Box")
    parser.add_argument("--cls-weight", type=float, default=3.0, help="For loss of Classification")
    parser.add_argument("--mask-weight", type=float, default=7.0, help="For loss of Segmentation")
    parser.add_argument("--dfl-weight", type=float, default=1.5, help="Refine the position of the box edges")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--accumulation-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--conf-thresh", type=float, default=0.9, help="Confidence threshold for validation")
    parser.add_argument("--nms-thresh", type=float, default=0.9, help="NMS IoU threshold for validation")
    parser.add_argument("--mask-batch-size", type=int, default=16, help="Batch size for mask processing during validation")
    parser.add_argument("--epoch-val", type=int, default=5, help="A calculation of valdation per number of epoch")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode: check and visualize outputs before starting to train.")
    args = parser.parse_args()
    main(args)