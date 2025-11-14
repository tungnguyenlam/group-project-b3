import torch
import torch.nn as nn
from torchvision.ops import nms
from .box_utils import make_anchor_points, decode_bboxes_with_dfl

class PostProcessor:
    """
    Ultralytics-style post-processor: 
    NMS FIRST, then generate masks only for final detections.
    """
    def __init__(self, conf_thresh=0.25, nms_thresh=0.45, reg_max=16, max_det=300):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = [8.0, 16.0, 32.0]
        self.num_classes_obj = 1
        self.reg_max = reg_max
        self.max_det = max_det  # Maximum detections per image

    def __call__(self, preds, input_shape):
        """Main post-processing function."""
        box_preds_dist, cls_preds, mask_coef_preds, proto_out = preds
        
        # Concatenate predictions from all FPN layers
        preds_reshaped = []
        for box_p, cls_p, mask_p in zip(box_preds_dist, cls_preds, mask_coef_preds):
            B, _, H, W = box_p.shape
            
            box_p = box_p.permute(0, 2, 3, 1).reshape(B, -1, 4 * self.reg_max)
            cls_p = cls_p.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes_obj)
            mask_p = mask_p.permute(0, 2, 3, 1).reshape(B, -1, mask_coef_preds[0].shape[1])
            preds_reshaped.append(torch.cat([box_p, cls_p, mask_p], dim=-1))
            
        all_preds = torch.cat(preds_reshaped, dim=1)
        
        anchor_points, stride_tensor = make_anchor_points(box_preds_dist, self.strides)

        # Process each image
        batch_detections = []
        for i in range(all_preds.shape[0]):  
            detection = self._process_single_image(
                all_preds[i],
                anchor_points,
                stride_tensor,
                proto_out[i],
                input_shape
            )
            batch_detections.append(detection)

        return batch_detections

    def _process_single_image(self, preds_i, anchor_points, stride_tensor, proto, input_shape):
        """
        Process a single image with Ultralytics-style optimization:
        1. Decode boxes
        2. Filter by confidence
        3. Apply NMS
        4. Generate masks ONLY for final detections
        """
        # Decode predictions
        pred_dist = preds_i[..., :4 * self.reg_max]
        decoded_boxes = decode_bboxes_with_dfl(
            anchor_points.unsqueeze(0), 
            pred_dist.unsqueeze(0), 
            stride_tensor.unsqueeze(0)
        ).squeeze(0)

        start_index_cls = 4 * self.reg_max
        start_index_mask = start_index_cls + self.num_classes_obj
        scores = torch.sigmoid(preds_i[..., start_index_cls:start_index_mask]).squeeze(-1)
        mask_coefs = preds_i[..., start_index_mask:]
        
        # === STEP 1: Confidence Filtering ===
        keep_mask = scores > self.conf_thresh
        
        if not keep_mask.any():
            return self._empty_detection(proto.device)

        boxes_filtered = decoded_boxes[keep_mask]
        scores_filtered = scores[keep_mask]
        coefs_filtered = mask_coefs[keep_mask]
        
        # === STEP 2: NMS (BEFORE mask generation) ===
        nms_indices = nms(boxes_filtered, scores_filtered, self.nms_thresh)
        
        # Limit to max detections
        if len(nms_indices) > self.max_det:
            nms_indices = nms_indices[:self.max_det]
        
        final_boxes = boxes_filtered[nms_indices]
        final_scores = scores_filtered[nms_indices]
        final_coefs = coefs_filtered[nms_indices]
        final_labels = torch.ones_like(final_scores, dtype=torch.long)
        
        # === STEP 3: Generate masks ONLY for final detections ===
        if final_boxes.shape[0] > 0:
            final_masks = self._process_masks_ultralytics_style(
                final_coefs, 
                final_boxes, 
                proto, 
                input_shape
            )
        else:
            final_masks = torch.empty(0, *input_shape, device=proto.device)

        return {
            "boxes": final_boxes,
            "scores": final_scores,
            "labels": final_labels,
            "masks": final_masks
        }

    def _process_masks_ultralytics_style(self, mask_coefs, boxes, proto, input_shape):
        """
        Ultralytics-style mask processing:
        1. Generate masks at proto resolution (small)
        2. Upsample to input resolution
        3. Crop using box coordinates directly (no coordinate grids!)
        """
        h_img, w_img = input_shape
        h_proto, w_proto = proto.shape[1:]
        num_masks = mask_coefs.shape[0]
        
        # === Generate masks at proto resolution (SMALL) ===
        # Shape: [num_masks, h_proto, w_proto]
        masks_proto = torch.sigmoid(
            (mask_coefs @ proto.view(proto.shape[0], -1)).view(num_masks, h_proto, w_proto)
        )
        
        # === Upsample to input resolution ===
        # Use align_corners=False for better quality (like Ultralytics)
        masks_upsampled = nn.functional.interpolate(
            masks_proto.unsqueeze(1),  # Add channel dimension
            size=(h_img, w_img),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # Remove channel dimension
        
        # === Crop using direct indexing (NO coordinate grids) ===
        masks_cropped = self._crop_mask_fast(masks_upsampled, boxes, h_img, w_img)
        
        # === Binarize ===
        masks_binary = (masks_cropped > 0.5).float()
        
        return masks_binary

    @staticmethod
    def _crop_mask_fast(masks, boxes, h, w):
        """
        Fast mask cropping without creating coordinate grids.
        Based on Ultralytics implementation.
        
        Args:
            masks: [N, H, W] tensor
            boxes: [N, 4] tensor in xyxy format
            h, w: image dimensions
        
        Returns:
            cropped_masks: [N, H, W] tensor
        """
        # Clamp boxes to image boundaries
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        
        n = masks.shape[0]
        
        # Create output tensor
        masks_cropped = torch.zeros_like(masks)
        
        # Process each mask individually (most memory efficient)
        for i in range(n):
            x1, y1, x2, y2 = boxes[i].int()
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Direct slice assignment (in-place, no extra memory)
            masks_cropped[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
        
        return masks_cropped

    def _empty_detection(self, device):
        """Create an empty detection dictionary."""
        return {
            "boxes": torch.empty(0, 4, device=device),
            "scores": torch.empty(0, device=device),
            "labels": torch.empty(0, dtype=torch.long, device=device),
            "masks": torch.empty(0, device=device)
        }

# ============================================================================
# ALTERNATIVE: Even more optimized version with batch processing
# ============================================================================

class PostProcessorV2(PostProcessor):
    """
    Ultra-optimized version that processes masks in mini-batches
    to balance memory usage and speed.
    """
    def __init__(self, conf_thresh=0.25, nms_thresh=0.45, reg_max=16, 
                 max_det=300, mask_batch_size=32):
        super().__init__(conf_thresh, nms_thresh, reg_max, max_det)
        self.mask_batch_size = mask_batch_size

    def _process_masks_ultralytics_style(self, mask_coefs, boxes, proto, input_shape):
        """
        Process masks in mini-batches for optimal memory/speed tradeoff.
        """
        h_img, w_img = input_shape
        h_proto, w_proto = proto.shape[1:]
        num_masks = mask_coefs.shape[0]
        device = mask_coefs.device
        
        # Allocate output
        all_masks = torch.zeros(num_masks, h_img, w_img, device=device)
        
        # Process in batches
        for start_idx in range(0, num_masks, self.mask_batch_size):
            end_idx = min(start_idx + self.mask_batch_size, num_masks)
            
            # Generate masks for this batch
            batch_coefs = mask_coefs[start_idx:end_idx]
            batch_boxes = boxes[start_idx:end_idx]
            
            # Generate at proto resolution
            masks_proto = torch.sigmoid(
                (batch_coefs @ proto.view(proto.shape[0], -1))
                .view(-1, h_proto, w_proto)
            )
            
            # Upsample
            masks_up = nn.functional.interpolate(
                masks_proto.unsqueeze(1),
                size=(h_img, w_img),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            # Crop
            masks_cropped = self._crop_mask_fast(masks_up, batch_boxes, h_img, w_img)
            
            # Store
            all_masks[start_idx:end_idx] = masks_cropped
        
        all_masks.clamp_(0, 1)
        all_masks.round_()
        # Binarize
        return all_masks.float()