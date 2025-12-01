from tqdm import tqdm
import torch

class ClassEvaluator():
    def __init__(self, train_loader, model):
        self.train_loader= train_loader.get_loader()
        self.model= model
        self._results = None


    def evaluate(self):
        from EvalSeg import EvalSeg
        
        loader= self.train_loader
        model= self.model
        
        all_gt_masks = []
        all_gt_boxes = []
        pred_masks = []
        pred_boxes = []
        pred_probs= []

        iterator = tqdm(loader, desc="Evaluating")

        for batch_idx, (imgs, batch_gt_masks, batch_gt_bboxes) in enumerate(iterator):
            
            for img_idx in range(len(batch_gt_masks)):
                all_gt_masks.append(batch_gt_masks[img_idx])
                all_gt_boxes.append(batch_gt_bboxes[img_idx])  # Đây đã là list của lists [[x1,y1,x2,y2], ...]
            
            results = model.predict(imgs)

            for p in results:
                # Boxes cho 1 image - LƯU TOÀN BỘ LIST
                img_pred_boxes = []
                img_probs= []
                
                for pb in p.boxes.xyxy.cpu().numpy():
                    img_pred_boxes.append(pb.tolist())  # Thêm từng box vào list
                
                # Mask cho 1 image - TẠO COMBINED MASK
                if p.masks is not None:
                    binary_masks = p.masks.data > 0.5  # [N, H, W]
                    combined_mask = binary_masks.any(dim=0)  # [H, W]
                    img_pred_mask = combined_mask.cpu()
                else:
                    H, W = imgs.shape[2], imgs.shape[3]
                    img_pred_mask = torch.zeros((H, W), dtype=torch.bool)
                
                # Thêm vào list chính
                pred_masks.append(img_pred_mask)           # 1 mask per image
                pred_boxes.append(img_pred_boxes) # 1 LIST of boxes per image

                probs= p.boxes.conf.cpu().tolist()
                pred_probs.append(probs)                

        tot_images= len(all_gt_masks)
        tot_pred_masks= len(pred_masks)
        tot_gt_bboxes= sum(len(boxes) for boxes in all_gt_boxes)
        tot_pred_bboxes= sum(len(boxes) for boxes in pred_boxes)
        

        eval_seg = EvalSeg(
            gt_masks=all_gt_masks,      # 93 items, mỗi item là tensor [H, W]
            gt_bboxes=all_gt_boxes,     # 93 items, mỗi item là [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            pred_masks=pred_masks,      # 93 items, mỗi item là tensor [H, W]
            pred_bboxes=pred_boxes,     # 93 items, mỗi item là [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
        )

        bbox_metrics= eval_seg.eval_bbox()
        mask_metrics= eval_seg.eval_mask()

        self._results={
            'bbox_metrics': bbox_metrics,
            'mask_metrics': mask_metrics,
            'tot_images': tot_images,
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
        print(f"Total ground truth bboxes: {results['tot_gt_bboxes']}\n")
        print(f"Total predicted bboxes: {results['tot_pred_bboxes']}\n")
        print(f"Total predicted masks: {results['tot_pred_masks']}")

