import os
import json
import torch
import numpy as np
from PIL import Image
from .base_dataset import BaseVisionDataset
from collections import defaultdict
from pycocotools import mask as mask_util
from sklearn.model_selection import train_test_split 

class MangaDataset(BaseVisionDataset):
    def __init__(self, json_dir, image_root, split='train', transforms=None, debug=False):
        super().__init__(transforms)
        self.image_root = image_root
        self.split = split
        self.debug = debug  # Toggle debug prints
        
        self.records = []
        self._build_records(json_dir)

        print(f"Successfully built {len(self.records)} records for the '{self.split}' split.")

    def _build_records(self, json_dir):
        """
        Builds a list of complete, verified records (image + its annotations).
        """
        all_json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        manga_titles = sorted([os.path.splitext(f)[0] for f in all_json_files])
        
        train_titles, val_titles = train_test_split(manga_titles, test_size=0.2, random_state=42)
        
        titles_for_this_split = train_titles if self.split == 'train' else val_titles
        print(f"Building records for {len(titles_for_this_split)} manga series in '{self.split}' split...")

        for manga_title in titles_for_this_split:
            json_path = os.path.join(json_dir, f"{manga_title}.json")
            
            with open(json_path, 'r') as f:
                data = json.load(f)

            annotations_map = defaultdict(list)
            for ann_info in data.get('annotations', []):
                if ann_info.get('category_id') == 5:
                    annotations_map[ann_info['image_id']].append(ann_info)
            
            for img_info in data.get('images', []):
                image_id_in_json = img_info['id']
                annotations = annotations_map.get(image_id_in_json)

                if not annotations:
                    continue

                plain_filename = os.path.basename(img_info['file_name'])
                correct_relative_path = os.path.join(manga_title, plain_filename)
                full_image_path = os.path.join(self.image_root, correct_relative_path)

                if os.path.exists(full_image_path):
                    record = {
                        'image_path': full_image_path,
                        'file_name_for_debug': correct_relative_path,
                        'height': img_info['height'],
                        'width': img_info['width'],
                        'annotations': annotations,
                        'image_id': image_id_in_json
                    }
                    self.records.append(record)

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]

        image = Image.open(record['image_path']).convert("L")
        
        boxes, labels, masks = [], [], []
        
        for ann in record['annotations']:
            boxes.append(ann['bbox'])
            labels.append(1)  # Single class: text bubble
            
            h, w = record['height'], record['width']
            segmentation = ann['segmentation']
    
            try:
                # Handle different segmentation formats
                if isinstance(segmentation, dict) and 'counts' in segmentation:
                    # Already in RLE dict format
                    rles = [segmentation]
                    
                elif isinstance(segmentation, str):
                    # RLE string format
                    rle_obj = {
                        'counts': segmentation.encode('utf-8') if isinstance(segmentation, str) else segmentation,
                        'size': [h, w]
                    }
                    rles = [rle_obj]

                elif isinstance(segmentation, list):
                    # Polygon format
                    if not segmentation or not segmentation[0]:
                        continue
                    
                    # Ensure it's a list of polygons
                    if not isinstance(segmentation[0], list):
                        segmentation = [segmentation]
                    
                    rles = mask_util.frPyObjects(segmentation, h, w)

                else:
                    if self.debug:
                        print(f"Unknown segmentation format for {record['file_name_for_debug']}")
                    continue
                
                # Merge and decode
                rle = mask_util.merge(rles)
                mask = mask_util.decode(rle)
                
                # Only add non-empty masks
                if mask.sum() > 0:
                    masks.append(mask)
                    
            except Exception as e:
                if self.debug:
                    print(f"Error processing mask in {record['file_name_for_debug']}: {e}")
                continue

        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Convert from XYWH to XYXY format
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if masks:
            masks = np.array(masks, dtype=np.uint8)
            masks = torch.from_numpy(masks)
        else:
            masks = torch.empty((0, record['height'], record['width']), dtype=torch.uint8)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks, 
            "image_id": torch.tensor([record['image_id']]),
            "file_name": record['file_name_for_debug'],
            "orig_size": torch.tensor([record['width'], record['height']])
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target