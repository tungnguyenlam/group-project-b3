from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import random
import numpy as np


class ClassEvaluator():
    def __init__(self, train_loader, model, device= None):
        self.train_loader= train_loader.get_loader()
        self.model= model
        self._results = None
        if device is None: 
            self.device= 'cpu'
        else:
            self.device= device


    def visualize_image(self, img_tensor, masks, boxes, alpha=0.5):
        """
        Visualize a single image with all predicted masks overlaid and bounding boxes.
    
        Args:
            img_tensor: Tensor [C,H,W], giá trị 0-255 hoặc 0-1
            masks: list of [H,W] bool masks
            boxes: list of [x1, y1, x2, y2]
            alpha: float, độ trong suốt của mask overlay
        """
        # --- Chuyển ảnh sang HWC và scale 0-1 nếu cần ---
        img = img_tensor.permute(1,2,0).cpu().numpy()
        if img.max() > 1.0:
            img = img / 255.0
    
        # --- Tạo ảnh overlay từ tất cả mask ---
        overlay = np.zeros_like(img)
    
        for mask in masks:
            color = np.random.rand(3)  # màu random
            # Broadcast mask lên 3 kênh
            overlay += np.where(mask[..., None], color, 0)
    
        # --- Clip overlay để không bị >1 ---
        overlay = np.clip(overlay, 0, 1)
    
        # --- Hiển thị ảnh gốc + overlay ---
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.imshow(overlay, alpha=alpha)  # overlay tất cả mask
           
        # --- Vẽ bbox ---
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
    
        plt.axis('off')
        plt.show()

        

    def evaluate(self):
        from EvalSeg import EvalSeg
        
        loader= self.train_loader
        model= self.model
        device= self.device
        
        all_gt_masks = []
        all_gt_boxes = []
        pred_masks = []
        pred_boxes = []
        pred_probs= []

        pbar = tqdm(total=len(loader), desc="Evaluating all images")


        for batch_idx, (imgs, batch_gt_masks, batch_gt_bboxes) in enumerate(loader):
            
            for img_idx in range(len(batch_gt_masks)):
                all_gt_masks.append(batch_gt_masks[img_idx])
                all_gt_boxes.append(batch_gt_bboxes[img_idx])  # Đây đã là list của lists [[x1,y1,x2,y2], ...]
            
            results = model.predict(imgs, device= device)

            for p in results:
                # Boxes cho 1 image - LƯU TOÀN BỘ LIST
                img_pred_boxes = []
                img_probs= []
                
                for pb in p.boxes.xyxy.cpu().numpy():
                    img_pred_boxes.append(pb.tolist())  # Thêm từng box vào list
                
                # Mask cho 1 image - TẠO COMBINED MASK
                if p.masks is not None:
                    binary_masks = p.masks.data > 0.5  # [N, H, W]
                    img_pred_masks = [binary_masks[i] for i in range(binary_masks.shape[0])]
                else:
                    H, W = imgs.shape[2], imgs.shape[3]
                    img_pred_masks = [torch.zeros((H, W), dtype=torch.bool)]
                
                # Thêm vào list chính
                pred_masks.append(img_pred_masks)        
                pred_boxes.append(img_pred_boxes) # 1 LIST of boxes per image

                probs= p.boxes.conf.cpu().tolist()
                pred_probs.append(probs)
                
            global_idx_start = batch_idx * len(imgs)
            rand_local_idx = random.randint(0, len(imgs)-1)
            rand_global_idx = global_idx_start + rand_local_idx
            
            self.visualize_image(
                imgs[rand_local_idx],
                pred_masks[rand_global_idx],
                pred_boxes[rand_global_idx]
            )
            pbar.update(len(imgs))


        tot_images= len(all_gt_masks)
        tot_gt_masks= sum(len(gt_masks) for gt_masks in all_gt_masks)
        tot_pred_masks= sum(len(masks) for masks in pred_masks)
        tot_gt_bboxes= sum(len(boxes) for boxes in all_gt_boxes)
        tot_pred_bboxes= sum(len(boxes) for boxes in pred_boxes)
        

        eval_seg = EvalSeg(
            gt_masks=all_gt_masks,      # 93 items, mỗi item là tensor [H, W]
            gt_bboxes=all_gt_boxes,     # 93 items, mỗi item là [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            pred_masks=pred_masks,      # 93 items, mỗi item là tensor [H, W]
            pred_bboxes=pred_boxes,     # 93 items, mỗi item là [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            pred_probs= pred_probs
        )

        bbox_metrics= eval_seg.eval_bbox()
        mask_metrics= eval_seg.eval_mask()

        self._results={
            'bbox_metrics': bbox_metrics,
            'mask_metrics': mask_metrics,
            'tot_images': tot_images,
            'tot_gt_masks': tot_gt_masks,
            'tot_gt_bboxes': tot_gt_bboxes,
            'tot_pred_bboxes': tot_pred_bboxes,
            'tot_pred_masks': tot_pred_masks,
        }

        return self._results

    def print_results(self):
        results = self._results
        print(f"BBox metrics: {results['bbox_metrics']}\n")
        print(f"Mask metrics: {results['mask_metrics']}\n")
        print(f"Total images: {results['tot_images']}\n")
        print(f"Total ground truth masks: {results['tot_gt_masks']}\n")
        print(f"Total ground truth bboxes: {results['tot_gt_bboxes']}\n")
        print(f"Total predicted bboxes: {results['tot_pred_bboxes']}\n")
        print(f"Total predicted masks: {results['tot_pred_masks']}")

