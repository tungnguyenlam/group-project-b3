from ultralytics import YOLO
import os
import gc
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import numpy as np

def plot_patch(ax, x, y, width, height):
    rect = Rectangle((x, y), width, height, 
                     linewidth = 1, edgecolor = "blue", fill = False, alpha = 0.7)
    ax.add_patch(rect)
    return ax


def plot_image(ax, image, boxes, plot_bbox=True, alpha=0.4):
    # --- convert image to numpy ---
    if isinstance(image, torch.Tensor):
        img = image.permute(1, 2, 0).cpu().numpy()
    else:
        img = image

    if img.max() > 1.0:
        img = img / 255.0
        
    ax.imshow(img)

    # --- plot bounding boxes ---
    if plot_bbox and boxes is not None and len(boxes) > 0:
        for box in boxes:  # box = [x1,y1,x2,y2]
            ax = plot_patch(ax, box[0], box[1], box[2] - box[0], box[3] - box[1])

    """
    # --- plot segmentation masks ---
    if masks is not None and len(masks) > 0:
        overlay = np.zeros_like(img)
        h, w = img.shape[:2]

        for mask in masks:
            # Resize từng mask về đúng kích thước ảnh
            mask_resized = cv2.resize(
                mask.astype(np.uint8), 
                (w, h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            color = np.random.rand(3)
            overlay += np.where(mask_resized[..., None], color, 0)

        overlay = np.clip(overlay, 0, 1)
        ax.imshow(overlay, alpha=alpha)
    """
    
    return ax


class YoloSeg:
    def __init__(self, model_pt_path: str, device, verbose):
        if not os.path.exists(model_pt_path):
            raise FileNotFoundError("Model path is not exist")
        else:
            self.model_pt_path = model_pt_path
            self.model = None
            self.device= device
            self.verbose = verbose 


    def load_model(self):
        try:
            self.model = YOLO(self.model_pt_path)
            print("Model load complete")
        except:
            print("Model path is not valid")

    def predict(self, imgs, device, verbose, print_bbox: bool= False, plot: bool= False, plot_bbox= False):
        
        verbose= self.verbose
        device= self.device
        results = self.model.predict(imgs, device= device, verbose= verbose)
        result= results[0]
        
        list_bboxes= []
        
        boxes= result.boxes.xyxy.cpu().numpy().astype(float) 
        for box in boxes: 
            list_bboxes.append(box.tolist())  
            
        #masks= result.masks.data.cpu().numpy().astype(bool) 
        masks_tensor= result.masks.data
        probs= result.boxes.conf.cpu()
        
        if print_bbox:
            print(list_bboxes)

        if plot:
        # result.orig_img is BGR numpy, convert to RGB
            orig_img = result.orig_img
            img_rgb = orig_img[:, :, ::-1]  

            fig, ax = plt.subplots(1,1)
            ax = plot_image(ax, img_rgb, list_bboxes, plot_bbox=plot_bbox)
            plt.show()
        
        return result.orig_img, list_bboxes, masks_tensor, probs

    """
    def predict_debug(self, image_path: str, print_bbox: bool = False, plot: bool = False, plot_bbox = True):
            if self.model is None: # Use 'is None' for comparison
                raise ValueError("Model has not been loaded successfully")
            elif not os.path.exists(image_path):   
                raise FileNotFoundError(f"Image path is not valid: {image_path}")
            else:
                result_seg = self.model(image_path)
                print(result_seg)

                # Check if results are empty
                if not result_seg or len(result_seg) == 0:
                    print("Warning: No results found in image.")
                    img = cv2.imread(image_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img_rgb, [] # Return image and empty list

                result = result_seg[0]
                
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Handle cases with no boxes
                if result.boxes is None or len(result.boxes) == 0:
                    print("No text bubbles found")
                    boxes = []
                else:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    masks = result.masks.xy
                    print(f"Found {len(boxes)} text bubbles")

                    if print_bbox:
                        print(boxes)

                    if plot:
                        fig, ax = plt.subplots(1,1)
                        ax = plot_image(ax, img_rgb, boxes, plot_bbox = True)
                        ax.plot()
        
                return img_rgb, boxes, masks
    """
    def unload_model(self):
        if self.model == None:
            print("The model is not loaded yet")
        else: 
            del self.model
            self.model = None
            gc.collect()