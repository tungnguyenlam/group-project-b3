import os

ENDWITHS = 'Pipelines'

NOTEBOOK_DIR = os.getcwd()

if not NOTEBOOK_DIR.endswith(ENDWITHS):
    raise ValueError(f"Not in correct dir, expect end with {ENDWITHS}, but got {NOTEBOOK_DIR} instead")

BASE_DIR = os.path.join(NOTEBOOK_DIR, '..', '..','..')

import sys
import os

# Add the code directory to path
sys.path.insert(0, os.path.join(BASE_DIR, 'code'))

# Now import from pipeline directly (not code.pipeline)
from tqdm.auto import tqdm
from pipeline.SegmentationModels.YoloSeg import YoloSeg, plot_patch, plot_image
from pipeline.OCRModels.MangaOCRModel import MangaOCRModel
from pipeline.TranslationModels.LLMTranslator import LLMTranslator
from ultralytics import YOLO
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import numpy as np
from math import ceil, floor

YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
EX_IMG_PATH = os.path.join(BASE_DIR, "data/Manga109_released_2023_12_07/images/AisazuNihaIrarenai/007.jpg")

yolo_model = YoloSeg(YOLO_MODEL_PATH)
yolo_model.load_model()
image_rgb, bboxes, masks = yolo_model.predict(image_path=EX_IMG_PATH, print_bbox = True, plot = True)
yolo_model.unload_model()

image_rgb = np.array(image_rgb)

manga_ocr_model = MangaOCRModel()
manga_ocr_model.load_model()

text_ocr_list = manga_ocr_model.predict(image_rgb, bboxes)
manga_ocr_model.unload_model()

for text in text_ocr_list:
    print(text)


import time

model_trans = LLMTranslator()
model_trans.load_model(model_name = "Qwen/Qwen2.5-1.5B")

start = time.time()

text_translated_list = model_trans.predict(text_ocr_list)

print("Time taken: ", time.time() - start)

for i, (ocr, trans) in enumerate(zip(text_ocr_list, text_translated_list)):
    print(f"Bbox {i}")
    print(ocr)
    print(trans,"\n")

ratio = image_rgb.shape[1] / image_rgb.shape[0]
width = 40
height = width / ratio

fig, ax = plt.subplots(1, 1, figsize=(width, height))
ax = plot_image(ax, image = image_rgb, boxes=bboxes, plot_bbox=True)

for box, ocr, trans in zip(bboxes, text_ocr_list, text_translated_list):
    # ax.text(x = box[0], y = box[3] + 15, s=ocr, fontsize=50, color="brown")
    ax.text(x = box[0] - 10, y = box[1], s=trans, fontsize=20, color="purple")



