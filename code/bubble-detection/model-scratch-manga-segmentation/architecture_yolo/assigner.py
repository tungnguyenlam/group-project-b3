import torch
import torch.nn as nn
from torchvision.ops import box_iou

class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner for object detection, inspired by YOLOv8.

    This assigner selects positive anchors based on a weighted score of
    classification confidence and IoU.
    """
    def __init__(self, topk=10, num_classes=1, alpha=1.0, beta=6.0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes):
        """
        Assigns ground-truth boxes to anchors.

        Args:
            pd_scores (Tensor): Predicted class scores for all anchors, shape [num_anchors, num_classes].
            pd_bboxes (Tensor): Decoded predicted bboxes for all anchors, shape [num_anchors, 4].
            anc_points (Tensor): Anchor points, shape [num_anchors, 2].
            gt_labels (Tensor): Ground truth labels, shape [num_gts].
            gt_bboxes (Tensor): Ground truth bboxes, shape [num_gts, 4].

        Returns:
            Tuple containing:
                - fg_mask (Tensor): A boolean mask indicating positive anchors, shape [num_anchors].
                - assigned_gt_idx (Tensor): The index of the GT box assigned to each positive anchor.
        """
        num_gts = gt_bboxes.shape[0]
        num_anchors = anc_points.shape[0]

        if num_gts == 0:
            return (
                torch.zeros(num_anchors, dtype=torch.bool, device=pd_scores.device),
                torch.zeros(num_anchors, dtype=torch.long, device=pd_scores.device)
            )

        # 1. Get candidate anchors that are inside the GT boxes
        # is_in_gts is a boolean mask of shape [num_anchors, num_gts]
        is_in_gts = self.get_candidates_in_gts(anc_points, gt_bboxes)
        
        # We only consider anchors that are candidates for at least one GT
        candidate_mask = is_in_gts.sum(dim=1) > 0
        candidate_idxs = torch.where(candidate_mask)[0]
        
        if len(candidate_idxs) == 0:
            return (
                torch.zeros(num_anchors, dtype=torch.bool, device=pd_scores.device),
                torch.zeros(num_anchors, dtype=torch.long, device=pd_scores.device)
            )
            
        candidate_pd_scores = pd_scores[candidate_idxs]
        candidate_pd_bboxes = pd_bboxes[candidate_idxs]

        # 2. Calculate alignment metric for candidate anchors
        # iou has shape [num_candidate_anchors, num_gts]
        iou = box_iou(candidate_pd_bboxes, gt_bboxes)
        
        # For single-class detection, use all scores
        if self.num_classes == 1:
            cls_scores = candidate_pd_scores[:, 0].unsqueeze(1).repeat(1, num_gts)
        else:
            # Multi-class: gather scores based on GT labels
            cls_scores = candidate_pd_scores.gather(
                1, gt_labels.unsqueeze(0).expand(len(candidate_idxs), -1)
            )
        
        # Only use scores for anchors inside GT boxes
        cls_scores = cls_scores * is_in_gts[candidate_idxs]

        # Alignment metric: score^alpha * iou^beta
        alignment_metric = cls_scores.pow(self.alpha) * iou.pow(self.beta)

        # 3. Select top-k anchors for each GT
        # topk_ious has shape [num_gts, topk]
        num_candidates = alignment_metric.shape[0]
        k = min(self.topk, num_candidates)
        topk_metrics, topk_indices = torch.topk(alignment_metric.T, k, dim=1, largest=True)
        
        # Create a mask for all selected top-k anchors
        topk_mask = torch.zeros_like(alignment_metric, dtype=torch.bool)
        topk_mask.scatter_(0, topk_indices.T, True)
        
        # Filter out assignments with zero metric (typically due to zero IoU or score)
        topk_mask = topk_mask & (alignment_metric > 0)

        # 4. Resolve overlaps: if an anchor is assigned to multiple GTs,
        # assign it to the one with the highest alignment metric.
        is_assigned = topk_mask.sum(dim=1) > 0
        
        # For each anchor that is assigned, find the GT it has the highest metric with
        max_metric_gt_idx = alignment_metric[is_assigned].argmax(dim=1)
        
        # Create the final foreground mask
        fg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=pd_scores.device)
        final_candidate_idxs = candidate_idxs[is_assigned]
        fg_mask[final_candidate_idxs] = True
        
        # Create the assigned GT index tensor
        assigned_gt_idx = torch.zeros(num_anchors, dtype=torch.long, device=pd_scores.device)
        assigned_gt_idx[final_candidate_idxs] = max_metric_gt_idx

        return fg_mask, assigned_gt_idx
    
    def get_candidates_in_gts(self, anchor_points, gt_bboxes):
        """Check if anchor points are inside any ground truth boxes."""
        gt_bboxes_expanded = gt_bboxes.unsqueeze(0) # [1, num_gts, 4]
        anchor_points_expanded = anchor_points.unsqueeze(1) # [num_anchors, 1, 2]
        
        x_anchors, y_anchors = anchor_points_expanded.chunk(2, dim=-1)
        x1_gt, y1_gt, x2_gt, y2_gt = gt_bboxes_expanded.chunk(4, dim=-1)
        
        is_in_gts = (x_anchors >= x1_gt) & (x_anchors <= x2_gt) & \
                    (y_anchors >= y1_gt) & (y_anchors <= y2_gt)
                    
        return is_in_gts.squeeze(-1) # [num_anchors, num_gts]