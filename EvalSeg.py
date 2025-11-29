"""
expect the result of the segmentation task to be 

binary mask: true/false (numpy array)

bbox: xyxy (numpy array)

"""
import torch
import numpy as np
from typing import List, Dict, Union, Optional

class EvalSeg:
    """
    Đánh giá segmentation cho manga bubble detection.
    Mỗi image có: nhiều bboxes + 1 combined mask
    """
    def __init__(self, 
                 gt_masks: List[torch.Tensor], 
                 gt_bboxes: List[List[List[float]]], 
                 pred_masks: List[torch.Tensor], 
                 pred_bboxes: List[List[List[float]]]):
        """
        Args:
            gt_masks: List of ground truth masks, mỗi item shape [H, W] (combined mask)
            gt_bboxes: List of ground truth bboxes, mỗi item là [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            pred_masks: List of predicted masks, mỗi item shape [H, W] (combined mask)
            pred_bboxes: List of predicted bboxes, mỗi item là [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
        """
        self.gt_masks = gt_masks
        self.gt_bboxes = gt_bboxes
        self.pred_masks = pred_masks
        self.pred_bboxes = pred_bboxes
        
    @staticmethod
    def iou_bbox(box1: Union[np.ndarray, List], box2: Union[np.ndarray, List]) -> float:
        """Tính IoU giữa 2 bounding boxes [x1, y1, x2, y2]"""
        if isinstance(box1, torch.Tensor):
            box1 = box1.cpu().numpy()
        if isinstance(box2, torch.Tensor):
            box2 = box2.cpu().numpy()
        
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        inter = max(0, xB - xA) * max(0, yB - yA)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - inter
        
        iou = inter / union if union > 0 else 0
        return iou
    
    @staticmethod
    def iou_mask(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Tính IoU cho masks (hỗ trợ batch)"""
        mask1 = mask1.bool()
        mask2 = mask2.bool()
        
        inter = (mask1 & mask2).sum(dim=(-2, -1))
        union = (mask1 | mask2).sum(dim=(-2, -1))
        iou = torch.where(union > 0, 
                         inter.float() / union.float(), 
                         torch.zeros_like(inter, dtype=torch.float))
        return iou
    
    @staticmethod
    def dice_mask(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Tính Dice coefficient cho masks"""
        mask1 = mask1.bool()
        mask2 = mask2.bool()
        
        inter = (mask1 & mask2).sum(dim=(-2, -1))
        total = mask1.sum(dim=(-2, -1)) + mask2.sum(dim=(-2, -1))
        dice = torch.where(total > 0, 
                          2 * inter.float() / total.float(), 
                          torch.zeros_like(inter, dtype=torch.float))
        return dice
    
    def eval_bbox(self, iou_thresholds: Optional[List[float]] = None) -> Dict:
        """
        Đánh giá bounding boxes.
        Dùng Hungarian matching để match pred boxes với gt boxes.
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75]
        
        all_ious = []
        total_gt = 0
        total_pred = 0
        
        # Duyệt qua từng image
        for pred_boxes, gt_boxes in zip(self.pred_bboxes, self.gt_bboxes):
            num_gt = len(gt_boxes)
            num_pred = len(pred_boxes)
            
            total_gt += num_gt
            total_pred += num_pred
            
            if num_pred == 0 or num_gt == 0:
                continue
            
            # Tính IoU matrix giữa tất cả các cặp
            iou_matrix = np.zeros((num_pred, num_gt))
            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    iou_matrix[i, j] = self.iou_bbox(pred_box, gt_box)
            
            # Greedy matching: mỗi pred box match với gt box có IoU cao nhất
            matched_ious = []
            for i in range(num_pred):
                if iou_matrix.shape[1] > 0:
                    max_iou = iou_matrix[i].max()
                    matched_ious.append(max_iou)
            
            all_ious.extend(matched_ious)
        
        if len(all_ious) == 0:
            return {
                "mean_iou": 0.0,
                "mAP": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "total_gt": total_gt,
                "total_pred": total_pred
            }
        
        mean_iou = np.mean(all_ious)
        
        # Tính metrics cho từng threshold
        aps = []
        results_per_threshold = {}
        
        for t in iou_thresholds:
            TP = sum([iou >= t for iou in all_ious])
            FP = total_pred - TP
            FN = total_gt - TP
            
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            aps.append(precision)
            
            results_per_threshold[f'AP@{t}'] = precision
            results_per_threshold[f'precision@{t}'] = precision
            results_per_threshold[f'recall@{t}'] = recall
            results_per_threshold[f'f1@{t}'] = f1
        
        return {
            "mean_iou": mean_iou,
            "mAP": np.mean(aps),
            "precision": precision,  # Của threshold cuối cùng
            "recall": recall,
            "f1": f1,
            "total_gt": total_gt,
            "total_pred": total_pred,
            **results_per_threshold
        }
    
    def eval_mask(self) -> Dict:
        """
        Đánh giá combined segmentation masks.
        So sánh từng cặp pred_mask vs gt_mask, rồi lấy trung bình.
        """
        if len(self.pred_masks) == 0 or len(self.gt_masks) == 0:
            return {
                "mean_iou": 0.0,
                "mean_dice": 0.0,
                "pixel_precision": 0.0,
                "pixel_recall": 0.0,
                "pixel_f1": 0.0
            }
        
        # Stack masks thành batch
        pred_masks = torch.stack([m.bool() for m in self.pred_masks])
        gt_masks = torch.stack([m.bool() for m in self.gt_masks])
        
        # Tính IoU và Dice cho từng image
        ious = self.iou_mask(pred_masks, gt_masks)
        dices = self.dice_mask(pred_masks, gt_masks)
        
        # Tính pixel-level metrics
        TP = (pred_masks & gt_masks).sum(dim=(-2, -1)).float()
        FP = (pred_masks & (~gt_masks)).sum(dim=(-2, -1)).float()
        FN = ((~pred_masks) & gt_masks).sum(dim=(-2, -1)).float()
        
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        pixel_f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            "mean_iou": ious.mean().item(),
            "mean_dice": dices.mean().item(),
            "pixel_precision": precision.mean().item(),
            "pixel_recall": recall.mean().item(),
            "pixel_f1": pixel_f1.mean().item()
        }
    
    def eval_all(self, iou_thresholds: Optional[List[float]] = None) -> Dict:
        """Đánh giá cả bbox và mask, trả về dict tổng hợp"""
        bbox_metrics = self.eval_bbox(iou_thresholds)
        mask_metrics = self.eval_mask()
        
        return {
            "bbox": bbox_metrics,
            "mask": mask_metrics
        }



def evaluate_model(model, dataloader, device='cuda'):
    
    model.eval()
    
    all_gt_masks = []
    all_gt_bboxes = []
    all_pred_masks = []
    all_pred_bboxes = []
    
    with torch.no_grad():
        for imgs, gt_masks, gt_bboxes in dataloader:
            imgs = imgs.to(device)
            
            # Forward pass - adapt theo model của bạn
            # Giả sử model trả về: pred_masks (list), pred_bboxes (list)
            pred_masks, pred_bboxes = model(imgs)
            
            # Thu thập kết quả
            batch_size = len(imgs)
            for i in range(batch_size):
                all_gt_masks.append(gt_masks[i])
                all_gt_bboxes.append(gt_bboxes[i])
                all_pred_masks.append(pred_masks[i].cpu())
                all_pred_bboxes.append(pred_bboxes[i])
    
    # Tạo evaluator
    evaluator = EvalSeg(
        gt_masks=all_gt_masks,
        gt_bboxes=all_gt_bboxes,
        pred_masks=all_pred_masks,
        pred_bboxes=all_pred_bboxes
    )
    
    # Đánh giá
    results = evaluator.eval_all(iou_thresholds=[0.5, 0.75, 0.9])
    
    return results


def print_results(results: Dict):
    
    print("\n BBOX METRICS:")
    bbox = results['bbox']
    print(f"  Mean IoU:        {bbox['mean_iou']:.4f}")
    print(f"  mAP:             {bbox['mAP']:.4f}")
    print(f"  Precision:       {bbox['precision']:.4f}")
    print(f"  Recall:          {bbox['recall']:.4f}")
    print(f"  F1 Score:        {bbox['f1']:.4f}")
    print(f"  Total GT:        {bbox['total_gt']}")
    print(f"  Total Pred:      {bbox['total_pred']}")
    
    print("\n MASK METRICS:")
    mask = results['mask']
    print(f"  Mean IoU:        {mask['mean_iou']:.4f}")
    print(f"  Mean Dice:       {mask['mean_dice']:.4f}")
    print(f"  Pixel Precision: {mask['pixel_precision']:.4f}")
    print(f"  Pixel Recall:    {mask['pixel_recall']:.4f}")
    print(f"  Pixel F1:        {mask['pixel_f1']:.4f}")
