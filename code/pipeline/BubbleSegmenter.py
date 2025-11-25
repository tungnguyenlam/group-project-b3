# code/pipeline/BubbleSegmenter.py
import cv2
import numpy as np
import sys
from ultralytics import YOLO

class BubbleSegmenter:
    def __init__(self, yolo_model_path):
        self.yolo_model = YOLO(yolo_model_path)
        # Hyperparameters for splitting
        self.MIN_DEFECT_DEPTH = 13
        self.MAX_ANGLE_DEG = 170
        self.MIN_DIST_BETWEEN_DEFECTS = 20
        sys.setrecursionlimit(2000)

    def detect_and_segment(self, image_path):
        """
        Main entry point: Detects raw masks and refines them via splitting.
        Returns a list of dictionaries with 'mask', 'bbox', 'contour'.
        """
        # 1. Run YOLO
        results = self.yolo_model.predict(source=image_path, verbose=False)
        result = results[0]
        
        image_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        raw_masks = result.masks.xy if result.masks else []

        refined_bubbles = []

        # 2. Process and Split Masks
        for mask_points in raw_masks:
            h, w = image_rgb.shape[:2]
            initial_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(initial_mask, [np.array(mask_points, dtype=np.int32)], 255)

            # Recursive split
            split_masks = self._split_connected_bubbles(initial_mask)

            # 3. Format results
            for single_mask in split_masks:
                contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                
                contour = max(contours, key=cv2.contourArea)
                bbox = cv2.boundingRect(contour)
                
                refined_bubbles.append({
                    'mask': single_mask,
                    'bbox': bbox,
                    'contour': contour
                })

        # Sort: Top-to-bottom, then Right-to-left (standard Manga reading order)
        refined_bubbles.sort(key=lambda b: (b['bbox'][1], -b['bbox'][0]))
        
        return image_rgb, refined_bubbles

    def _attempt_split_once(self, bubble_mask):
        """Core math logic for finding convexity defects and cutting."""
        contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return [bubble_mask]
        
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 500: return [bubble_mask]

        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is None or len(hull_indices) < 3: return [bubble_mask]
            defects = cv2.convexityDefects(contour, hull_indices)
        except: return [bubble_mask]

        if defects is None: return [bubble_mask]

        candidates = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
            
            if depth > self.MIN_DEFECT_DEPTH:
                far = tuple(contour[f][0])
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                
                v1 = np.array(start) - np.array(far)
                v2 = np.array(end) - np.array(far)
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                
                if n1 == 0 or n2 == 0: continue
                
                cosine = np.dot(v1, v2) / (n1 * n2)
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                
                if angle < self.MAX_ANGLE_DEG:
                    candidates.append({'point': far, 'depth': depth})

        # Clustering
        candidates.sort(key=lambda x: x['depth'], reverse=True)
        unique_candidates = []
        for cand in candidates:
            is_distinct = True
            for exist in unique_candidates:
                dist = np.linalg.norm(np.array(cand['point']) - np.array(exist['point']))
                if dist < self.MIN_DIST_BETWEEN_DEFECTS:
                    is_distinct = False
                    break
            if is_distinct: unique_candidates.append(cand)

        # Execute Cut
        split_mask = bubble_mask.copy()
        cut_happened = False
        
        if len(unique_candidates) >= 2:
            # Shortest distance logic
            best_pair = None
            min_cut_dist = float('inf')
            consider_list = unique_candidates[:4]
            
            for i in range(len(consider_list)):
                for j in range(i + 1, len(consider_list)):
                    p1 = consider_list[i]['point']
                    p2 = consider_list[j]['point']
                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    if dist < min_cut_dist:
                        min_cut_dist = dist
                        best_pair = (p1, p2)
            
            if best_pair:
                cv2.line(split_mask, best_pair[0], best_pair[1], 0, 3)
                cut_happened = True
                
        elif len(unique_candidates) == 1:
            # Cut to centroid
            p1 = unique_candidates[0]['point']
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                dx, dy = cx - p1[0], cy - p1[1]
                target = (int(cx + dx*0.5), int(cy + dy*0.5))
                cv2.line(split_mask, p1, target, 0, 3)
                cut_happened = True

        if cut_happened:
            new_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_masks = []
            for c in new_contours:
                if cv2.contourArea(c) > 100: 
                    m = np.zeros_like(bubble_mask)
                    cv2.drawContours(m, [c], -1, 255, -1)
                    new_masks.append(m)
            if len(new_masks) > 1:
                return new_masks

        return [bubble_mask]

    def _split_connected_bubbles(self, bubble_mask):
        """Recursive wrapper."""
        initial_results = self._attempt_split_once(bubble_mask)
        if len(initial_results) == 1:
            return initial_results
        
        final_bubbles = []
        for sub_mask in initial_results:
            sub_results = self._split_connected_bubbles(sub_mask)
            final_bubbles.extend(sub_results)
        return final_bubbles