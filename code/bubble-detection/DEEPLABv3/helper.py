import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os

class MangaBubbleDataset(Dataset):
    def __init__(self, json_file, img_dir, img_size=(256, 256), transform=None):
        """
        Args:
            json_file: Đường dẫn đến train.json hoặc val.json
            img_dir: Thư mục gốc chứa images (VD: /kaggle/input/manga109-images/images)
            img_size: Size để resize (width, height)
            transform: Transform cho image (nếu có)
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        
        # Load COCO JSON
        with open(json_file, "r") as f:
            ann = json.load(f)
        
        self.images = ann["images"]
        self.annotations = ann["annotations"]
        
        # Mapping: image_id → image_info
        self.image_infos = {img["id"]: img for img in self.images}
        
        # Mapping: image_id → list of annotations
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
        img_name = img_info["file_name"]  # VD: "Nekodama/001.jpg"
        
        # Load image với đường dẫn đầy đủ
        img_path = os.path.join(self.img_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Lấy kích thước gốc
        orig_width = img_info["width"]
        orig_height = img_info["height"]
        
        # Resize image
        img = img.resize(self.img_size)
        
        # Tạo empty mask
        mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)
        
        # Tính scale factor
        scale_x = self.img_size[0] / orig_width
        scale_y = self.img_size[1] / orig_height
        
        # Vẽ polygon mask từ annotations
        if img_id in self.annos_by_id:
            for ann in self.annos_by_id[img_id]:
                segs = ann.get("segmentation", [])
                if not segs:
                    continue
                
                for seg in segs:
                    # seg là list [x1,y1,x2,y2,...,xn,yn]
                    poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    
                    # Scale polygon theo resize
                    poly[:, 0] *= scale_x
                    poly[:, 1] *= scale_y
                    
                    # Vẽ polygon lên mask
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        
        # Apply transform cho image
        if self.transform:
            img = self.transform(img)
        else:
            # Nếu không có transform, convert sang tensor
            img = transforms.ToTensor()(img)
        
        # Convert mask → tensor [1, H, W]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return img, mask


def normalize_series_name(name):
    name = name.replace("'s", "")
    return ''.join(c for c in name if c.isalnum() or c in ["_", "-"])

def gather_json(series_list, keep_cat_id=5, mask_dir="./masks"):
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