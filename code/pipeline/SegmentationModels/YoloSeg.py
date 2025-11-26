from ultralytics import YOLO
import os
import gc
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_patch(ax, x, y, width, height):
    rect = Rectangle((x, y), width, height, 
                     linewidth = 1, edgecolor = "blue", fill = False, alpha = 0.7)
    ax.add_patch(rect)
    return ax


def plot_image(ax, image, boxes, plot_bbox = True):
    ax.imshow(image)
    if plot_bbox:
        for box in boxes:
            ax = plot_patch(ax, box[0], box[1], box[2]-box[0], box[3]-box[1])
    return ax


class YoloSeg:
    def __init__(self, model_pt_path: str):
        if not os.path.exists(model_pt_path):
            raise FileNotFoundError("Model path is not exist")
        else:
            self.model_pt_path = model_pt_path
            self.model = None

    def load_model(self):
        try:
            self.model = YOLO(self.model_pt_path)
            print("Model load complete")
        except:
            print("Model path is not valid")

    def predict(self, image_path: str, print_bbox: bool = False, plot: bool = False, plot_bbox = True):
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

    def unload_model(self):
        if self.model == None:
            print("The model is not loaded yet")
        else: 
            del self.model
            self.model = None
            gc.collect()