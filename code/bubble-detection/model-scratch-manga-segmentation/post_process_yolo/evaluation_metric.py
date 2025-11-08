import torch
from torchvision.ops import box_iou 

def mask_iou_torch(mask1, mask2, batch_size=16):
    """
    Calculate IoU between two sets of masks (PyTorch GPU version).
    Args:
        mask1: [N, H, W] binary tensor on GPU
        mask2: [M, H, W] binary tensor on GPU
    Returns:
        iou: [N, M] IoU matrix on GPU
    """
    N, H, W = mask1.shape
    M = mask2.shape[0]
    
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=mask1.device)
    
    # Flatten mask 2 only once time
    mask2_flat = mask2.view(M, -1).float()  # [M, H*W]
    area2 = mask2_flat.sum(axis=1)  # [M]

    iou = torch.zeros((N, M), device=mask1.device)
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        mask1_batch = mask1[i:end]

        mask1_batch_flat = mask1_batch.view(-1, H * W).float() # [batch_size, H*W]
    
        # Calculate intersection and union
        intersection_batch = torch.matmul(mask1_batch_flat, mask2_flat.T) # [batch_size, M]
        
        area1_batch = mask1_batch_flat.sum(axis=1) # [batch_size]
        union_batch = area1_batch[:, None] + area2[None, :] - intersection_batch
        iou_batch = intersection_batch / (union_batch + 1e-6)

        iou[i:end, :] = iou_batch
    
    return iou


class MetricsCalculator:
    """
    Calculate AP (Average Precision) on GPU.
    """
    def __init__(self, device, iou_thresholds=None, conf_threshold=0.001):
        self.device = device
        if iou_thresholds is None:
            self.iou_thresholds = torch.linspace(0.5, 0.95, 10, device=self.device)
        else:
            self.iou_thresholds = torch.tensor(iou_thresholds, device=self.device)
        
        self.conf_threshold = conf_threshold
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.stats = []
        self.total_gts = 0
    
    def add_batch(self, predictions, targets, iou_type='bbox'):
        """
        Add a batch of predictions and targets (all tensors on GPU).
        """
        for pred, target in zip(predictions, targets):
            
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            # Count total GTs
            self.total_gts += len(gt_boxes)
            
            # Filter by confidence
            conf_mask = pred_scores >= self.conf_threshold
            pred_boxes = pred_boxes[conf_mask]
            pred_scores = pred_scores[conf_mask]
            pred_labels = pred_labels[conf_mask]
            
            num_preds = len(pred_boxes)
            num_gts = len(gt_boxes)

            # Handle empty cases
            if num_preds == 0:
                continue
            
            if num_gts == 0:
                # All predictions are false positives
                correct = torch.zeros((num_preds, len(self.iou_thresholds)), 
                                     device=self.device, dtype=torch.bool)
                self.stats.append((correct, pred_scores, pred_labels))
                continue

            # Calculate IoU matrix on GPU
            if iou_type == 'bbox':
                iou_matrix = box_iou(pred_boxes, gt_boxes)
            else:  # segm
                pred_masks = pred['masks'][conf_mask]
                gt_masks = target['masks']
                iou_matrix = mask_iou_torch(pred_masks, gt_masks)
            
            # Match predictions to ground truths on GPU
            correct = self._match_predictions(iou_matrix, pred_labels, gt_labels)
            
            # Store GPU tensors
            self.stats.append((correct, pred_scores, pred_labels))
    
    def _match_predictions(self, iou_matrix, pred_labels, gt_labels):
        """ Match predictions to ground truths on GPU. """
        num_preds, num_gts = iou_matrix.shape
        num_thresholds = len(self.iou_thresholds)

        #  Initialize correct matrix
        correct = torch.zeros((num_preds, num_thresholds), 
                             device=self.device, dtype=torch.bool)
        
        # For each IoU threshold
        for t_idx, iou_thresh in enumerate(self.iou_thresholds):
            # Find matches above this threshold
            matches = iou_matrix > iou_thresh
            
            # Check class matching
            class_match = pred_labels[:, None] == gt_labels
            matches = matches & class_match
            
            # Greedy matching: assign each GT to best prediction
            matched_preds = set()
            
            # Sort GTs by max IoU (descending)
            if matches.any():
                max_ious_per_gt = iou_matrix.max(dim=0)[0]
                sorted_gt_indices = torch.argsort(max_ious_per_gt, descending=True)
                
                for gt_idx in sorted_gt_indices:
                    if matches[:, gt_idx].any():
                        # Find best matching prediction for this GT
                        pred_idx = iou_matrix[:, gt_idx].argmax().item()
                        
                        if pred_idx not in matched_preds and matches[pred_idx, gt_idx]:
                            correct[pred_idx, t_idx] = True
                            matched_preds.add(pred_idx)
        
        return correct
    
    def compute_ap(self):
        """Compute Average Precision on GPU."""
        if not self.stats:
            return {
                'ap': 0.0, 'ap50': 0.0, 'ap75': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
        
        # Concatenate all stats
        correct_list, conf_list, pred_cls_list = [], [], []
        for stat in self.stats:
            correct_list.append(stat[0])
            conf_list.append(stat[1])
            pred_cls_list.append(stat[2])
        
        correct = torch.cat(correct_list, 0)
        conf = torch.cat(conf_list, 0)
        
        # Sort by confidence (descending)
        sorted_indices = torch.argsort(conf, descending=True)
        correct = correct[sorted_indices]
        
        if self.total_gts == 0:
            return {
                'ap': 0.0, 'ap50': 0.0, 'ap75': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
        
        # Calculate AP for each IoU threshold
        aps = []
        for t_idx in range(len(self.iou_thresholds)):
            tp = correct[:, t_idx]
            fp = ~tp
            
            tp_cumsum = tp.cumsum(0).float()
            fp_cumsum = fp.cumsum(0).float()
            
            recall = tp_cumsum / (self.total_gts + 1e-6)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            # Calculate AP using 101-point interpolation
            ap = self._compute_ap_interp(recall, precision)
            aps.append(ap)
        
        aps = torch.stack(aps)
        
        # Get AP at specific IoU thresholds
        ap50_idx = torch.argmin(torch.abs(self.iou_thresholds - 0.5))
        ap75_idx = torch.argmin(torch.abs(self.iou_thresholds - 0.75))
        
        # Final metrics at IoU=0.5
        final_tp = correct[:, ap50_idx].sum().float()
        final_fp = len(correct) - final_tp
        final_precision = final_tp / (final_tp + final_fp + 1e-6)
        final_recall = final_tp / (self.total_gts + 1e-6)
        final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-6)
        
        return {
            'ap': aps.mean().item(),
            'ap50': aps[ap50_idx].item(),
            'ap75': aps[ap75_idx].item(),
            'precision': final_precision.item(),
            'recall': final_recall.item(),
            'f1': final_f1.item()
        }
    
    def _compute_ap_interp(self, recall, precision):
        """Compute AP using 101-point interpolation on GPU."""
        # Add sentinel values
        recall = torch.cat([
            torch.tensor([0.0], device=self.device), 
            recall, 
            torch.tensor([1.0], device=self.device)
        ])
        precision = torch.cat([
            torch.tensor([0.0], device=self.device), 
            precision, 
            torch.tensor([0.0], device=self.device)
        ])
        
        # Make precision monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = torch.maximum(precision[i], precision[i + 1])
        
        # Calculate AP using 101 recall levels
        recall_levels = torch.linspace(0, 1, 101, device=self.device)
        
        # Interpolate precision at each recall level
        precision_interp = []
        for r in recall_levels:
            # Find first recall >= r
            indices = torch.where(recall >= r)[0]
            if len(indices) > 0:
                precision_interp.append(precision[indices[0]])
            else:
                precision_interp.append(torch.tensor(0.0, device=self.device))
        
        ap = torch.stack(precision_interp).mean()
        return ap