def load_gt_polygons(label_path, img_height, img_width):
    """Loads ground-truth polygons from a YOLO .txt file."""
    polygons = []
    if not os.path.exists(label_path):
        return polygons
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Denormalize polygon
            poly = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
            poly[:, 0] *= img_width
            poly[:, 1] *= img_height
            polygons.append(poly.astype(np.int32))
    return polygons

def polygons_to_mask(polygons, height, width):
    """Converts a list of polygons to a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, pts=polygons, color=255)
    return mask

def calculate_area_iou(gt_polygons, pred_polygon, height, width):
    """Calculates Area IoU between a predicted polygon and all GT polygons."""
    pred_mask = polygons_to_mask([pred_polygon], height, width)
    gt_mask = polygons_to_mask(gt_polygons, height, width)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    return intersection / union if union > 0 else 0

def calculate_boundary_iou(gt_poly, pred_poly, height, width, thickness=2):
    """Calculates Boundary IoU for a single pair of polygons."""
    gt_boundary_mask = np.zeros((height, width), dtype=np.uint8)
    pred_boundary_mask = np.zeros((height, width), dtype=np.uint8)
    
    cv2.polylines(gt_boundary_mask, [gt_poly], isClosed=True, color=255, thickness=thickness)
    cv2.polylines(pred_boundary_mask, [pred_poly], isClosed=True, color=255, thickness=thickness)
    
    intersection = np.logical_and(gt_boundary_mask, pred_boundary_mask).sum()
    union = np.logical_or(gt_boundary_mask, pred_boundary_mask).sum()
    
    return intersection / union if union > 0 else 0