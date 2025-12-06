"""
expect the result of the segmentation task to be 

binary mask: true/false (numpy array)

bbox: xyxy (numpy array)

"""
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from scipy.optimize import linear_sum_assignment

def yolo_style_ap(recalls, precisions):
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    return np.trapz(precisions, recalls)

    

class EvalSeg:
    """
    Đánh giá segmentation cho manga bubble detection.
    Mỗi image có: nhiều bboxes + 1 combined mask
    """
    def __init__(self, 
                 gt_masks: List[torch.Tensor], 
                 gt_bboxes: List[List[List[float]]], 
                 pred_masks: List[torch.Tensor], 
                 pred_bboxes: List[List[List[float]]],
                pred_probs: List[List[float]]):
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
        self.pred_probs= pred_probs
        
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

    

    def eval_bbox(self, iou_thresholds=None):
        """
        YOLO-style bbox evaluation:
        - Hungarian-style matching per image
        - Predictions sorted by confidence
        - Computes AP for multiple IoU thresholds
        - Computes mean IoU across TP matches
        """
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, 10)
    
        # ---- Flatten GT + Pred ----
        all_gt = []
        all_pred = []
    
        for img_idx, (gt_boxes, pred_boxes, probs) in enumerate(
                zip(self.gt_bboxes, self.pred_bboxes, self.pred_probs)):
    
            # Ground truth
            for box in gt_boxes:
                all_gt.append({"image": img_idx, "box": np.array(box)})
    
            # Predictions
            for box, conf in zip(pred_boxes, probs):
                all_pred.append({
                    "image": img_idx,
                    "box": np.array(box),
                    "conf": float(conf)
                })
    
        # Sort predictions theo confidence giảm dần
        all_pred.sort(key=lambda x: -x["conf"])
    
        aps = []
        results_per_threshold = {}
        mean_ious_list = []
    
        for thr in iou_thresholds:
            TP = []
            FP = []
    
            used_gt = {}  # image_idx → set(gt_idx already matched)
            matched_ious = []
    
            for pred in all_pred:
                img = pred["image"]
                pred_box = pred["box"]
    
                # Lấy GT của ảnh này
                gt_of_img = [g for g in all_gt if g["image"] == img]
                if len(gt_of_img) == 0:
                    FP.append(1)
                    TP.append(0)
                    continue
    
                # IoU với mỗi GT
                ious = np.array([self.iou_bbox(pred_box, g["box"]) for g in gt_of_img])
                max_iou = ious.max()
                max_gt_idx = ious.argmax()
    
                if max_iou >= thr:
                    if img not in used_gt:
                        used_gt[img] = set()
                    if max_gt_idx not in used_gt[img]:
                        TP.append(1)
                        FP.append(0)
                        used_gt[img].add(max_gt_idx)
                        matched_ious.append(max_iou)  # lưu IoU của TP
                    else:
                        FP.append(1)
                        TP.append(0)
                else:
                    FP.append(1)
                    TP.append(0)
    
            # Tính mean IoU cho threshold này
            mean_ious_list.append(np.mean(matched_ious) if matched_ious else 0.0)
    
            TP = np.array(TP)
            FP = np.array(FP)
    
            cum_TP = np.cumsum(TP)
            cum_FP = np.cumsum(FP)
    
            total_gt = len(all_gt)
    
            recall = cum_TP / (total_gt + 1e-6)
            precision = cum_TP / (cum_TP + cum_FP + 1e-6)
    
            ap = yolo_style_ap(recall, precision)
            aps.append(ap)
    
            results_per_threshold[f"AP@{thr:.2f}"] = ap
            results_per_threshold[f"P@{thr:.2f}"] = precision[-1]
            results_per_threshold[f"R@{thr:.2f}"] = recall[-1]
            results_per_threshold[f"mIoU@{thr:.2f}"] = mean_ious_list[-1]
    
        return {
            "mAP50": results_per_threshold["AP@0.50"],
            "mAP50-95": np.mean(aps),
            "precision": results_per_threshold["P@0.50"],
            "recall": results_per_threshold["R@0.50"],
            "mean_iou": np.mean(mean_ious_list)  # trung bình IoU qua tất cả thresholds

        }

                    
    def eval_mask(self, iou_thresholds=None):
        """
        YOLO/COCO-style mask AP + mean IoU
        Assumes:
            - self.gt_masks: list of list of GT masks per image, each mask [H,W]
            - self.pred_masks: list of list of predicted masks per image, each mask [H,W]
            - self.pred_probs: list of list of confidence scores per predicted mask
        """
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, 10)
    
        # ---- 1. Flatten GT + Pred ----
        all_gt = []
        all_pred = []
    
        for img_idx, (gt_list, pred_list, probs) in enumerate(
                zip(self.gt_masks, self.pred_masks, self.pred_probs)):
    
            # GT masks
            for gmask in gt_list:
                all_gt.append({"image": img_idx, "mask": gmask.bool()})
    
            # Predictions
            for pmask, conf in zip(pred_list, probs):
                all_pred.append({
                    "image": img_idx,
                    "mask": pmask.bool(),
                    "conf": float(conf)
                })
    
        # Sort predictions theo confidence giảm dần
        all_pred.sort(key=lambda x: -x["conf"])
    
        aps = []
        results_per_threshold = {}
        mean_ious_list = []
    
        for thr in iou_thresholds:
            TP, FP = [], []
            used_gt = {}  # image_idx -> set of matched gt indices
            matched_ious = []
    
            for pred in all_pred:
                img = pred["image"]
                pmask = pred["mask"]
    
                gt_of_img = [g for g in all_gt if g["image"] == img]
    
                if len(gt_of_img) == 0:
                    FP.append(1)
                    TP.append(0)
                    continue
    
                # Compute IoU list
                ious = np.array([self.iou_mask(pmask, g["mask"]) for g in gt_of_img])
                best = ious.max()
                best_gt_idx = ious.argmax()
    
                if best >= thr:
                    if img not in used_gt:
                        used_gt[img] = set()
    
                    if best_gt_idx not in used_gt[img]:
                        TP.append(1)
                        FP.append(0)
                        used_gt[img].add(best_gt_idx)
                        matched_ious.append(best)
                    else:
                        FP.append(1)
                        TP.append(0)
                else:
                    FP.append(1)
                    TP.append(0)
    
            mean_ious_list.append(np.mean(matched_ious) if matched_ious else 0.0)
    
            TP = np.array(TP)
            FP = np.array(FP)
            cum_TP = np.cumsum(TP)
            cum_FP = np.cumsum(FP)
            total_gt = len(all_gt)
    
            recall = cum_TP / (total_gt + 1e-6)
            precision = cum_TP / (cum_TP + cum_FP + 1e-6)
    
            ap = yolo_style_ap(recall, precision)
            aps.append(ap)
    
            results_per_threshold[f"AP@{thr:.2f}"] = ap
            results_per_threshold[f"P@{thr:.2f}"] = precision[-1]
            results_per_threshold[f"R@{thr:.2f}"] = recall[-1]
            results_per_threshold[f"mIoU@{thr:.2f}"] = mean_ious_list[-1]
    
        return {
            "mAP50": results_per_threshold.get("AP@0.50", 0.0),
            "mAP50-95": np.mean(aps),
            "precision": results_per_threshold.get("P@0.50", 0.0),
            "recall": results_per_threshold.get("R@0.50", 0.0),
            "mean_iou": np.mean(mean_ious_list)
        }



        
    
    def eval_all(self, iou_thresholds: Optional[List[float]] = None) -> Dict:
        """Đánh giá cả bbox và mask, trả về dict tổng hợp"""
        bbox_metrics = self.eval_bbox(iou_thresholds)
        mask_metrics = self.eval_mask()
        
        return {
            "bbox": bbox_metrics,
            "mask": mask_metrics
        }


def print_results(results: Dict):
    # --- BBOX METRICS ---
    bbox = results['bbox']
    print("\n BBOX METRICS:")
    print(f"  Mean IoU:        {bbox.get('mean_iou', 0.0):.4f}")
    print(f"  mAP50:           {bbox.get('mAP50', 0.0):.4f}")
    print(f"  mAP50-95:        {bbox.get('mAP50-95', 0.0):.4f}")
    print(f"  Precision:       {bbox.get('precision', 0.0):.4f}")
    print(f"  Recall:          {bbox.get('recall', 0.0):.4f}")
    print(f"  F1 Score:        {bbox.get('f1', 0.0):.4f}")
    
    # --- MASK METRICS ---
    mask = results['mask']
    print("\n MASK METRICS:")
    print(f"  Mean IoU:        {mask.get('mean_iou', 0.0):.4f}")
    print(f"  mAP50:           {mask.get('mAP50', 0.0):.4f}")
    print(f"  mAP50-95:        {mask.get('mAP50-95', 0.0):.4f}")
    print(f"  Precision:       {mask.get('precision', 0.0):.4f}")
    print(f"  Recall:          {mask.get('recall', 0.0):.4f}")
