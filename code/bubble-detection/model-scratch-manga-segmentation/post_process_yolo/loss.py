import torch
import torch.nn as nn

from architecture_yolo.assigner import TaskAlignedAssigner
from post_process_yolo.box_utils import make_anchor_points, decode_bboxes_with_dfl


class YOLOv11Loss(nn.Module):
    """Complete Loss layer for YOLO-Seg with FIXED decoding"""
    def __init__(self, model, box_weight, cls_weight, mask_weight, dfl_weight ,num_classes_obj=1, reg_max=16):
        super().__init__()
        self.model = model
        self.num_classes_obj = num_classes_obj
        self.reg_max = reg_max
        self.strides = [8.0, 16.0, 32.0]
        
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.num_classes_obj, alpha=1.0, beta=6.0)
        
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_mask = nn.BCEWithLogitsLoss(reduction='mean') 
        self.loss_dfl = nn.CrossEntropyLoss(reduction='none')
        
        # Loss weights
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.mask_weight = mask_weight
        self.dfl_weight = dfl_weight
        
        self.anchor_points_cache = None
        self.stride_tensor_cache = None

    def forward(self, preds, targets):
        box_preds_dist, cls_preds, mask_coef_preds, proto_out = preds
        device = proto_out.device

        # Use anchor point generation
        if self.anchor_points_cache is None or self.anchor_points_cache.device != device:
            self.anchor_points_cache, self.stride_tensor_cache = make_anchor_points(
                box_preds_dist, self.strides
            )
        
        anchor_points = self.anchor_points_cache
        stride_tensor = self.stride_tensor_cache
        
        batch_size = box_preds_dist[0].shape[0]
        
        # Flatten predictions
        box_preds_flat = torch.cat([
            p.permute(0, 2, 3, 1).reshape(batch_size, -1, 4 * self.reg_max) 
            for p in box_preds_dist
        ], dim=1)
        
        cls_preds_flat = torch.cat([
            p.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes_obj) 
            for p in cls_preds
        ], dim=1)
        
        mask_coef_preds_flat = torch.cat([
            p.permute(0, 2, 3, 1).reshape(batch_size, -1, mask_coef_preds[0].shape[1]) 
            for p in mask_coef_preds
        ], dim=1)
        
        # Expand for batch
        anchor_points_batch = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
        stride_tensor_batch = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Use decoder
        decoded_boxes = decode_bboxes_with_dfl(
            anchor_points_batch, 
            box_preds_flat, 
            stride_tensor_batch,
            self.reg_max
        )
        
        pred_scores = torch.sigmoid(cls_preds_flat)

        gt_boxes = [t['boxes'].to(device) for t in targets]
        gt_masks = [t['masks'].to(device) for t in targets]
        
        loss_cls_sum, loss_box_sum, loss_dfl_sum, loss_mask_sum = 0.0, 0.0, 0.0, 0.0
        num_pos = 0.0

        for i in range(batch_size):
            gt_box_i = gt_boxes[i]
            gt_mask_i = gt_masks[i]
            
            if gt_box_i.shape[0] == 0:
                continue
            
            gt_labels = torch.zeros(gt_box_i.shape[0], device=device, dtype=torch.long)

            fg_mask, assigned_gt_idx = self.assigner(
                pred_scores[i],
                decoded_boxes[i],
                anchor_points,
                gt_labels,
                gt_box_i[:, :4]
            )
            
            current_pos = fg_mask.sum()
            if current_pos == 0:
                continue
            
            num_pos += current_pos

            pred_dist_matched = box_preds_flat[i][fg_mask]
            decoded_boxes_matched = decoded_boxes[i][fg_mask]
            cls_preds_matched = cls_preds_flat[i][fg_mask]
            mask_coefs_matched = mask_coef_preds_flat[i][fg_mask]
            anchor_points_matched = anchor_points[fg_mask]
            stride_tensor_matched = stride_tensor[fg_mask]
            
            gt_box_matched = gt_box_i[assigned_gt_idx[fg_mask], :4]
            gt_mask_matched = gt_mask_i[assigned_gt_idx[fg_mask]]

            # Box Loss (IoU)
            from torchvision.ops import box_iou
            iou = box_iou(decoded_boxes_matched, gt_box_matched).diag()
            loss_box_sum += (1.0 - iou).sum()
            
            # Class Loss
            cls_target_pos = iou.detach().clamp(0, 1).unsqueeze(1)
            loss_cls_sum += self.bce_cls(cls_preds_matched, cls_target_pos).sum()

            # DFL Loss 
            gt_ltrb = self.make_ltrb_fixed(anchor_points_matched, gt_box_matched, stride_tensor_matched)
            
            # Clamp to valid range [0, reg_max - 1e-6]
            target_ltrb_scaled = gt_ltrb.clamp(0, self.reg_max - 1e-6)
            
            # Get integer bounds
            target_left = target_ltrb_scaled.long()  # Floor
            target_right = (target_left + 1).clamp(max=self.reg_max - 1)  # Ceiling
            # Ensure we don't try to interpolate at the boundary
            at_boundary = (target_left >= self.reg_max - 1)
            target_right = torch.where(at_boundary, target_left, target_right)

            # Interpolation weights
            weight_right = target_ltrb_scaled - target_left.float()
            weight_left = 1.0 - weight_right

            weight_right = torch.where(at_boundary, torch.zeros_like(weight_right), weight_right)
            weight_left = torch.where(at_boundary, torch.ones_like(weight_left), weight_left)
            
            # Reshape predictions for DFL loss
            pred_dist_reshaped = pred_dist_matched.view(-1, self.reg_max)
            target_left_reshaped = target_left.view(-1)
            target_right_reshaped = target_right.view(-1)
            weight_left_reshaped = weight_left.view(-1)
            weight_right_reshaped = weight_right.view(-1)
            
            # Two-point distribution loss
            loss_dfl = (
                self.loss_dfl(pred_dist_reshaped, target_left_reshaped) * weight_left_reshaped +
                self.loss_dfl(pred_dist_reshaped, target_right_reshaped) * weight_right_reshaped
            )
            loss_dfl_sum += loss_dfl.sum()

            # Mask Loss
            proto_i = proto_out[i]
            num_prototypes, h_proto, w_proto = proto_i.shape
            pred_masks = (mask_coefs_matched @ proto_i.view(num_prototypes, -1)).view(-1, h_proto, w_proto)
            
            target_masks_aligned = nn.functional.interpolate(
                gt_mask_matched.unsqueeze(1).float(), 
                size=(h_proto, w_proto), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            
            num_masks_in_call = pred_masks.shape[0]
            if num_masks_in_call > 0:
                loss_mask_sum += self.bce_mask(pred_masks, target_masks_aligned) * num_masks_in_call

        if num_pos > 0:
            loss_cls = loss_cls_sum / num_pos
            loss_box = loss_box_sum / num_pos
            loss_dfl = loss_dfl_sum / (num_pos * 4)
            loss_mask = loss_mask_sum / num_pos
        else:
            zero_loss = box_preds_flat.sum() * 0.0 
            loss_cls, loss_box, loss_dfl, loss_mask = (
                zero_loss, zero_loss, zero_loss, zero_loss
            )

        total_loss = (
            self.box_weight * loss_box + 
            self.cls_weight * loss_cls + 
            self.mask_weight * loss_mask + 
            self.dfl_weight * loss_dfl
        )
        
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls.detach(),
            'loss_box': loss_box.detach(),
            'loss_dfl': loss_dfl.detach(),
            'loss_mask': loss_mask.detach()
        }
    
    def make_ltrb_fixed(self, anchor_points, gt_bboxes, stride_tensor):
        """
        Calculate left-top-right-bottom distances in STRIDE UNITS.
        
        Args:
            anchor_points: [N, 2] in PIXELS
            gt_bboxes: [N, 4] in PIXELS (xyxy format)
            stride_tensor: [N, 1] stride values
        
        Returns:
            ltrb: [N, 4] distances in stride units
        """
        # Calculate pixel distances
        d_left = anchor_points[:, 0:1] - gt_bboxes[:, 0:1]    # anchor_x - x1
        d_top = anchor_points[:, 1:2] - gt_bboxes[:, 1:2]     # anchor_y - y1
        d_right = gt_bboxes[:, 2:3] - anchor_points[:, 0:1]   # x2 - anchor_x
        d_bottom = gt_bboxes[:, 3:4] - anchor_points[:, 1:2]  # y2 - anchor_y
        
        ltrb_pixels = torch.cat([d_left, d_top, d_right, d_bottom], dim=1)
        
        # Convert to stride units
        ltrb_stride_units = ltrb_pixels / stride_tensor
        
        return ltrb_stride_units