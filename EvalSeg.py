"""
expect the result of the segmentation task to be 

binary mask: true/false (numpy array)

bbox: xyxy (numpy array)

"""
import numpy as np
import torch


class EvalSeg:
    def __init__(self):
        self.gt_masks= None
        self.gt_bboxes= None
        self.masks= None
        self.bboxes= None

    def load_info(self, gt_masks, gt_bboxes, masks, bboxes):
        self.gt_masks= gt_masks
        self.gt_bboxes= gt_bboxes      
        self.masks= masks
        self.bboxes= bboxes

    @staticmethod
    def iou_bbox(box1, box2):
        if isinstance(box1, torch.Tensor):
            box1 = box1.cpu().numpy()
        if isinstance(box2, torch.Tensor):
            box2 = box2.cpu().numpy()
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
    def iou_mask(mask1, mask2):
        """
        mask1, mask2: torch.bool tensors, shape [H,W] hoặc [N,H,W]
        Trả về IoU: float nếu single mask, tensor nếu batch
        """
        inter = (mask1 & mask2).sum(dim=(-2, -1))  # sum over H,W
        union = (mask1 | mask2).sum(dim=(-2, -1))
        iou = torch.where(union > 0, inter.float() / union.float(), torch.zeros_like(inter, dtype=torch.float))
        return iou

    @staticmethod
    def dice_mask(mask1, mask2):
        """
        mask1, mask2: torch.bool tensors, shape [H,W] hoặc [N,H,W]
        Trả về Dice coefficient
        """
        inter = (mask1 & mask2).sum(dim=(-2, -1))
        total = mask1.sum(dim=(-2, -1)) + mask2.sum(dim=(-2, -1))
        dice = torch.where(total > 0, 2 * inter.float() / total.float(), torch.zeros_like(inter, dtype=torch.float))
        return dice


    def eval_bbox(self, iou_thresholds=None):
        """
        pred_boxes, gt_boxes: list of bbox per image, format [[x1,y1,x2,y2],...]
        returns: dict of mAP, mean IoU, Precision/Recall/F1
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5]

        ious = []
        for pb, gb in zip(self.bboxes, self.gt_bboxes):
            ious.append(self.iou_bbox(pb, gb))
        mean_iou = np.mean(ious)

        aps = []
        for t in iou_thresholds:
            TP = sum([iou >= t for iou in ious])
            FP = sum([iou < t for iou in ious])
            FN = 0  # assuming 1 gt per pred, adjust if multiple gt
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            f1 = 2*precision*recall / (precision+recall+1e-6)
            aps.append(precision)  # simple AP per threshold

        return {
            "mean_iou": mean_iou,
            "mAP": np.mean(aps),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def eval_mask(self):
        """
        Trả về dict: mean_iou, mean_dice, mean_pixel_f1
        """
        # Tính IoU & Dice
        ious = self.iou_mask(self.masks, self.gt_masks)      # tensor [N]
        dices = self.dice_mask(self.masks, self.gt_masks)    # tensor [N]

        # Tính Pixel-level F1
        TP = (self.masks & self.gt_masks).sum(dim=(-2, -1)).float()
        FP = (self.masks & (~self.gt_masks)).sum(dim=(-2, -1)).float()
        FN = ((~self.masks) & self.gt_masks).sum(dim=(-2, -1)).float()

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        pixel_f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "mean_iou": ious.mean().item(),
            "mean_dice": dices.mean().item(),
            "mean_pixel_f1": pixel_f1.mean().item()
        }

    