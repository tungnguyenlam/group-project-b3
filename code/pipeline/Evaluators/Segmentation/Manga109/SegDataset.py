import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils



def normalize_series_name(name):
    name = name.replace("'", "")
    return ''.join(c for c in name if c.isalnum() or c in ["_", "-"])
    
def gather_json(series_list, mask_dir, keep_cat_id=5):
    combined_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": f"category_{keep_cat_id}"}]
    }
    
    ann_id_counter = 1
    new_img_id = 1
    
    # Remove duplicates
    series_list = list(dict.fromkeys(series_list))
    print(f" Processing {len(series_list)} unique series...\n")
    
    for idx, s in enumerate(series_list, 1):
        print(f"[{idx}/{len(series_list)}] Processing series: {s}")
        
        json_file = os.path.join(mask_dir, f"{normalize_series_name(s)}.json")
        
        if not os.path.exists(json_file):
            print(f" Warning: {json_file} not found\n")
            continue
        
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f" Error reading {json_file}: {e}\n")
            continue
        
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        
        print(f"   Total images in JSON: {len(images)}")
        print(f"   Total annotations in JSON: {len(annotations)}")
        
        # STEP 1: Tìm images nào có annotation category keep_cat_id
        img_has_target_ann = set()
        for ann in annotations:
            if ann["category_id"] == keep_cat_id:
                img_has_target_ann.add(ann["image_id"])
        
        print(f"  Images with category {keep_cat_id}: {len(img_has_target_ann)}")
        
        # STEP 2: CHỈ thêm images có trong img_has_target_ann
        img_id_map = {}
        added_imgs = 0
        
        for img in images:
            old_id = img["id"]
            
            # CRITICAL: Skip nếu image không có annotation category 5
            if old_id not in img_has_target_ann:
                continue
            
            img_name = os.path.basename(img["file_name"])
            file_name = f"{s}/{img_name}"
            
            img_id_map[old_id] = new_img_id
            
            combined_data["images"].append({
                "id": new_img_id,
                "file_name": file_name,
                "width": img.get("width", 0),
                "height": img.get("height", 0)
            })
            new_img_id += 1
            added_imgs += 1
        
        print(f"  Added {added_imgs} images (filtered)")
        
        # STEP 3: Add annotations
        cat5_count = 0
        skipped_count = 0
        
        for ann in annotations:
            if ann["category_id"] != keep_cat_id:
                continue
            
            old_img_id = ann["image_id"]
            new_id = img_id_map.get(old_img_id)
            
            if new_id is None:
                skipped_count += 1
                continue
            
            combined_data["annotations"].append({
                "id": ann_id_counter,
                "image_id": new_id,
                "category_id": 1,
                "segmentation": ann.get("segmentation", []),
                "bbox": ann.get("bbox", []),
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0)
            })
            cat5_count += 1
            ann_id_counter += 1
        
        print(f"  Added {cat5_count} annotations")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} annotations (orphaned)")
        print()
    
    print(f"   Total images: {len(combined_data['images'])}")
    print(f"   Total annotations: {len(combined_data['annotations'])}")
    print(f"   Images per series (avg): {len(combined_data['images']) / len(series_list):.1f}")
    
    if combined_data['images']:
        print("\n Sample file_names (last 5):")
        for fname in [img['file_name'] for img in combined_data['images'][-5:]]:
            print(f"   - {fname}")
    
    return combined_data

class MangaBubbleDataset(Dataset):
    def __init__(self, json_file, img_dir, img_size, transform=None):
        self.img_dir = img_dir
        self.img_size = img_size  # (height, width)
        self.transform = transform

        # Load COCO JSON
        with open(json_file, "r") as f:
            ann = json.load(f)

        self.images = ann["images"]
        self.annotations = ann["annotations"]

        # Mapping: image_id -> list of annotations
        self.annos_by_id = {}
        for a in self.annotations:
            img_id = a["image_id"]
            if img_id not in self.annos_by_id:
                self.annos_by_id[img_id] = []
            self.annos_by_id[img_id].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load image
        img = Image.open(img_path).convert("RGB")
        orig_width, orig_height = img_info["width"], img_info["height"]
        img = img.resize(self.img_size)
        scale_x = self.img_size[0] / orig_width
        scale_y = self.img_size[1] / orig_height

        # Get annotations for this image
        annos = self.annos_by_id.get(img_id, [])

        # --- Scale bboxes ---
        scaled_bboxes = []
        for ann in annos:
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = map(float, bbox)
                scaled_bboxes.append([x*scale_x, y*scale_y,
                                      (x + w)*scale_x, (y + h)*scale_y])

        
        object_mask= []
        for ann in annos:
            seg = ann.get("segmentation", [])
            if seg:
                rle = mask_utils.frPyObjects(seg, orig_height, orig_width)
                mask = mask_utils.decode(rle)
                if mask.ndim == 3:
                    mask_2d = mask.squeeze()
                else:
                    mask_2d = mask
                mask_resized = cv2.resize(mask_2d.astype(np.uint8),
                                  (self.img_size[1], self.img_size[0]),
                                  interpolation=cv2.INTER_NEAREST)
                mask_tensor= torch.from_numpy(mask_resized.astype(bool))

            object_mask.append(mask_tensor)

        # Apply transform to image
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, object_mask, scaled_bboxes
                
                
                
            
            
                

        

        

    
    
    