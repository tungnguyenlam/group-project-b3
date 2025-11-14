import torch

def make_anchor_points(feats, strides, grid_cell_offset=0.5):
    """
    Generate a grid of anchor points from the feature maps. 
    These are the center points of each cell in the feature grid.
    """
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        # Create x, y coordinate grid
        sx = torch.arange(end=w, device=feats[i].device, dtype=torch.float32) + grid_cell_offset
        sy = torch.arange(end=h, device=feats[i].device, dtype=torch.float32) + grid_cell_offset
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')
        
        # Stack and multiply by stride to get coordinates on original image
        anchor_points.append(torch.stack((grid_x, grid_y), -1).view(-1, 2) * stride)
        stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device, dtype=torch.float32))
        
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def decode_bboxes_with_dfl(anchor_points, pred_dist, stride_tensor, reg_max=16):
    """
    Decode bounding boxes using DFL output for a batch.
    
    Args:
        anchor_points: [batch_size, num_anchors, 2] - anchor center coordinates in PIXELS
        pred_dist: [batch_size, num_anchors, 4 * reg_max] - DFL predictions
        stride_tensor: [batch_size, num_anchors, 1] - stride for each anchor
        reg_max: maximum distance value (default 16)
    
    Returns:
        decoded_boxes: [batch_size, num_anchors, 4] in xyxy format
    """
    b, a, _ = pred_dist.shape

    # Reshape to [batch, num_anchors, 4, reg_max] and apply softmax
    pred_dist = pred_dist.reshape(b, a, 4, reg_max)
    pred_dist = pred_dist.softmax(-1)  # Softmax over reg_max dimension
    
    # Create distance projection vector [0, 1, 2, ..., reg_max-1]
    project = torch.arange(reg_max, device=pred_dist.device, dtype=torch.float32)
    
    # DFL: Weighted sum to get expected distance
    # Result shape: [batch, num_anchors, 4]
    ltrb_dist = (pred_dist * project.view(1, 1, 1, reg_max)).sum(-1)
    
    # CRITICAL: Multiply by stride to convert from stride units to pixels
    # ltrb_dist is in "stride units", we need pixels
    ltrb_dist_pixels = ltrb_dist * stride_tensor  # [b, a, 4] * [b, a, 1] broadcasts to [b, a, 4]
    
    # Split into left-top and right-bottom distances
    lt_dist = ltrb_dist_pixels[..., :2]  # [b, a, 2] - distances to left and top
    rb_dist = ltrb_dist_pixels[..., 2:]  # [b, a, 2] - distances to right and bottom
    
    # Calculate final box coordinates
    # x1 = anchor_x - left_distance
    # y1 = anchor_y - top_distance  
    # x2 = anchor_x + right_distance
    # y2 = anchor_y + bottom_distance
    x1y1 = anchor_points - lt_dist
    x2y2 = anchor_points + rb_dist
    
    decoded_boxes = torch.cat([x1y1, x2y2], -1)  # [b, a, 4] in xyxy format
    
    return decoded_boxes
