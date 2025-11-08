import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from post_process_yolo.box_utils import make_anchor_points, decode_bboxes_with_dfl
from post_process_yolo.loss import YOLOv11Loss

def debug_predictions_vs_gt(model, data_loader, post_processor, device, num_samples=5):
    """
    Visualize predictions vs ground truth to check coordinate alignment.
    """
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx >= num_samples:
                break
                
            images = images.to(device)
            
            # Forward pass
            preds = model(images)
            input_shape = images.shape[-2:]
            results = post_processor(preds, input_shape)
            
            for i in range(min(2, len(images))):  # Show first 2 images per batch
                # Get image
                img = images[i].cpu().numpy()
                if img.shape[0] == 1:  # Grayscale
                    img = img[0]
                
                # Get original size info
                orig_size = targets[i]['orig_size'].cpu().numpy()
                size_after_resize = targets[i]['size'].cpu().numpy()
                
                orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
                w_resize, h_resize = int(size_after_resize[0]), int(size_after_resize[1])
                
                print(f"\n{'='*60}")
                print(f"Image {batch_idx}-{i}:")
                print(f"  Original size: {orig_w}x{orig_h}")
                print(f"  After resize: {w_resize}x{h_resize}")
                print(f"  After padding: {img.shape[1]}x{img.shape[0]}")
                print(f"  Num GT boxes: {len(targets[i]['boxes'])}")
                print(f"  Num predictions: {len(results[i]['boxes'])}")
                
                # Check if boxes are in reasonable range
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                pred_boxes = results[i]['boxes'].cpu().numpy()
                
                if len(gt_boxes) > 0:
                    print(f"\n  GT boxes range:")
                    print(f"    X: [{gt_boxes[:, 0].min():.1f}, {gt_boxes[:, 2].max():.1f}]")
                    print(f"    Y: [{gt_boxes[:, 1].min():.1f}, {gt_boxes[:, 3].max():.1f}]")
                    print(f"    Expected X range: [0, {w_resize}]")
                    print(f"    Expected Y range: [0, {h_resize}]")
                
                if len(pred_boxes) > 0:
                    print(f"\n  Pred boxes range:")
                    print(f"    X: [{pred_boxes[:, 0].min():.1f}, {pred_boxes[:, 2].max():.1f}]")
                    print(f"    Y: [{pred_boxes[:, 1].min():.1f}, {pred_boxes[:, 3].max():.1f}]")
                    print(f"    Scores: [{results[i]['scores'].min():.3f}, {results[i]['scores'].max():.3f}]")
                
                # Visualize
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Plot 1: Input image with GT boxes
                ax = axes[0]
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Ground Truth (n={len(gt_boxes)})')
                for box in gt_boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                             edgecolor='green', facecolor='none')
                    ax.add_patch(rect)
                
                # Plot 2: Input image with predictions
                ax = axes[1]
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Predictions (n={len(pred_boxes)})')
                for box, score in zip(pred_boxes, results[i]['scores'].cpu().numpy()):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                             edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8)
                
                # Plot 3: Overlay
                ax = axes[2]
                ax.imshow(img, cmap='gray')
                ax.set_title('Overlay (GT=green, Pred=red)')
                for box in gt_boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                             edgecolor='green', facecolor='none', alpha=0.7)
                    ax.add_patch(rect)
                for box in pred_boxes[:10]:  # Show top 10
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                             edgecolor='red', facecolor='none', alpha=0.7)
                    ax.add_patch(rect)
                
                plt.tight_layout()
                plt.savefig(f'debug_image_{batch_idx}_{i}.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved visualization to debug_image_{batch_idx}_{i}.png")


def check_assigner_outputs(model, data_loader, device, num_batches=3):
    """
    Check if Task-Aligned Assigner is working correctly.
    """
    model.train()  # Need gradients
    loss_fn = YOLOv11Loss(model, num_classes_obj=1, box_weight=7.0, cls_weight=1.0, mask_weight=7.0, dfl_weight=1.0)
    
    print("\n" + "="*60)
    print("Checking Task-Aligned Assigner")
    print("="*60)
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
            
        images = images.to(device)
        
        # Forward pass
        preds = model(images)
        
        # Get anchor points
        box_preds_dist, cls_preds, mask_coef_preds, proto_out = preds
        anchor_points, stride_tensor = make_anchor_points(box_preds_dist, [8.0, 16.0, 32.0])
        
        batch_size = box_preds_dist[0].shape[0]
        
        # Decode boxes
        box_preds_flat = torch.cat([
            p.permute(0, 2, 3, 1).reshape(batch_size, -1, 4 * 16) 
            for p in box_preds_dist
        ], dim=1)
        
        cls_preds_flat = torch.cat([
            p.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) 
            for p in cls_preds
        ], dim=1)
        
        anchor_points_batch = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
        stride_tensor_batch = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        decoded_boxes = decode_bboxes_with_dfl(anchor_points_batch, box_preds_flat, stride_tensor_batch)
        pred_scores = torch.sigmoid(cls_preds_flat)
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Total anchors: {anchor_points.shape[0]}")
        print(f"  Pred scores range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
        
        # Check assigner for each image
        gt_boxes = [t['boxes'].to(device) for t in targets]
        
        total_pos = 0
        for i in range(batch_size):
            gt_box_i = gt_boxes[i]
            
            if gt_box_i.shape[0] == 0:
                continue
            
            gt_labels = torch.zeros(gt_box_i.shape[0], device=device, dtype=torch.long)
            
            fg_mask, assigned_gt_idx = loss_fn.assigner(
                pred_scores[i],
                decoded_boxes[i],
                anchor_points,
                gt_labels,
                gt_box_i[:, :4]
            )
            
            num_pos = fg_mask.sum().item()
            total_pos += num_pos
            
            print(f"  Image {i}: GT={len(gt_box_i)}, Positive anchors={num_pos}")
            
            if num_pos == 0:
                print(f"    WARNING: No positive anchors assigned!")
                # Debug why
                from torchvision.ops import box_iou
                iou_matrix = box_iou(decoded_boxes[i], gt_box_i)
                max_iou = iou_matrix.max(dim=0)[0]
                print(f"    Max IoU per GT: {max_iou.detach().cpu().numpy()}")
                print(f"    Decoded boxes range: X=[{decoded_boxes[i][:, 0].min():.1f}, {decoded_boxes[i][:, 2].max():.1f}], Y=[{decoded_boxes[i][:, 1].min():.1f}, {decoded_boxes[i][:, 3].max():.1f}]")
                print(f"    GT boxes range: X=[{gt_box_i[:, 0].min():.1f}, {gt_box_i[:, 2].max():.1f}], Y=[{gt_box_i[:, 1].min():.1f}, {gt_box_i[:, 3].max():.1f}]")
        
        print(f"  Total positive anchors in batch: {total_pos}")
        print(f"  Average per image: {total_pos / batch_size:.1f}")


def verify_decoding(model, data_loader, device):
    """
    Verify that box decoding is working correctly.
    """
    print("\n" + "="*60)
    print("VERIFYING BOX DECODING")
    print("="*60)
    
    model.eval()
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            
            # Forward
            preds = model(images)
            box_preds_dist, cls_preds, _, _ = preds
            
            # Make anchors
            anchor_points, stride_tensor = make_anchor_points(box_preds_dist, [8.0, 16.0, 32.0])
            
            print(f"\n1. Anchor Points:")
            print(f"   Shape: {anchor_points.shape}")
            print(f"   Range: X=[{anchor_points[:, 0].min():.1f}, {anchor_points[:, 0].max():.1f}]")
            print(f"          Y=[{anchor_points[:, 1].min():.1f}, {anchor_points[:, 1].max():.1f}]")
            print(f"   First 5: {anchor_points[:5]}")
            
            # Flatten predictions
            batch_size = box_preds_dist[0].shape[0]
            box_preds_flat = torch.cat([
                p.permute(0, 2, 3, 1).reshape(batch_size, -1, 64) 
                for p in box_preds_dist
            ], dim=1)
            
            print(f"\n2. Box Predictions:")
            print(f"   Shape: {box_preds_flat.shape}")
            print(f"   Range (before softmax): [{box_preds_flat.min():.2f}, {box_preds_flat.max():.2f}]")
            
            # Decode
            anchor_points_batch = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
            stride_tensor_batch = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            decoded_boxes = decode_bboxes_with_dfl(
                anchor_points_batch,
                box_preds_flat,
                stride_tensor_batch,
                reg_max=16
            )
            
            print(f"\n3. Decoded Boxes:")
            print(f"   Shape: {decoded_boxes.shape}")
            print(f"   X range: [{decoded_boxes[0, :, 0].min():.1f}, {decoded_boxes[0, :, 2].max():.1f}]")
            print(f"   Y range: [{decoded_boxes[0, :, 1].min():.1f}, {decoded_boxes[0, :, 3].max():.1f}]")
            print(f"   First 3 boxes:\n{decoded_boxes[0, :3]}")
            
            # Compare with GT
            gt_boxes = targets[0]['boxes']
            print(f"\n4. Ground Truth Boxes:")
            print(f"   Shape: {gt_boxes.shape}")
            print(f"   X range: [{gt_boxes[:, 0].min():.1f}, {gt_boxes[:, 2].max():.1f}]")
            print(f"   Y range: [{gt_boxes[:, 1].min():.1f}, {gt_boxes[:, 3].max():.1f}]")
            print(f"   First 3 boxes:\n{gt_boxes[:3]}")
            
            # Calculate IoU
            from torchvision.ops import box_iou
            iou_matrix = box_iou(decoded_boxes[0], gt_boxes.to(device))
            max_iou_per_gt = iou_matrix.max(dim=0)[0]
            
            print(f"\n5. IoU Analysis:")
            print(f"   Max IoU per GT box: {max_iou_per_gt.cpu().numpy()}")
            print(f"   Number of GT with IoU > 0.1: {(max_iou_per_gt > 0.1).sum()}")
            print(f"   Number of GT with IoU > 0.5: {(max_iou_per_gt > 0.5).sum()}")
            
            break
    
    print("="*60 + "\n")