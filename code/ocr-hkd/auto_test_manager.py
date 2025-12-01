import os
import re
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    VisionEncoderDecoderModel, 
    AutoTokenizer, 
    AutoImageProcessor,
    AutoModel          
)
import evaluate
from .auto_train_manager import SegformerFeatureCorrector

# ================= CONFIGURATION =================
BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'models')
JSON_PATH = os.path.join(BASE_DIR, 'data', 'manga109_ocr_dataset.json')
IMAGE_ROOT = os.path.join(BASE_DIR, 'data', 'Manga109_released_2023_12_07', 'images')

# Pipeline Config (Default False)
USE_MASK = False 

# ================= UTILS: VERSION MANAGER =================
def get_latest_model():
    """Find the model folder with the highest version (manga_ocr_vX)"""
    if not os.path.exists(MODELS_DIR):
        print(f"Directory {MODELS_DIR} does not exist.")
        return None, None

    versions = []
    for d in os.listdir(MODELS_DIR):
        match = re.match(r"manga_ocr_v(\d+)", d)
        if match:
            versions.append(int(match.group(1)))
    
    latest_ver = max(versions)
    model_path = os.path.join(MODELS_DIR, f"manga_ocr_v{latest_ver}")
    return model_path, latest_ver

# ================= HELPER FUNCTIONS =================
def normalize_jp_text(text):
    """Normalize: Remove whitespace for exact comparison"""
    return text.replace(" ", "").replace("\u3000", "").strip()

def apply_mask_processing(img, bbox, mask_polys):
    """Remove background based on Mask"""
    xmin, ymin, xmax, ymax = map(int, bbox)
    crop = img[ymin:ymax, xmin:xmax].copy()
    if not mask_polys: return crop
    
    h, w = crop.shape[:2]
    local_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in mask_polys:
        pts = np.array(poly).reshape(-1, 2) - np.array([xmin, ymin])
        cv2.fillPoly(local_mask, [pts.astype(np.int32)], 255)
        
    white_bg = np.ones_like(crop) * 255
    mask_3ch = cv2.merge([local_mask, local_mask, local_mask])
    return np.where(mask_3ch == 255, crop, white_bg)

# ================= MAIN TEST LOGIC =================
def run_auto_test():
    # 1. Find Latest Model
    model_path, version_num = get_latest_model()
    if not model_path:
        return

    csv_output_name = f"test_results_v{version_num}.csv"
    print("="*50)
    print(f"🔍 FOUND LATEST MODEL: v{version_num}")
    print(f"📂 Path: {model_path}")
    print(f"💾 Results will be saved to: {csv_output_name}")
    print("="*50)

    # 2. Load Model & Components
    print("Loading model components...")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_path).cuda()
        model.encoder = SegformerFeatureCorrector(model.encoder)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)

            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Load Data
    print(f"Loading test data from {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    test_data = [item for item in full_data if item['split'] == 'test']
    print(f"Test samples: {len(test_data)}")

    # 4. Inference
    cer_metric = evaluate.load("cer")
    clean_preds, clean_refs = [], []
    results = []

    model.eval()
    print(f"🚀 Running Inference on v{version_num}...")
    
    for i, item in enumerate(tqdm(test_data)):
        # Load Image
        img_path = os.path.join(IMAGE_ROOT, item['img_path'])
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop
        if USE_MASK and item.get('mask'):
            crop = apply_mask_processing(img, item['bbox'], item['mask'])
        else:
            xmin, ymin, xmax, ymax = map(int, item['bbox'])
            # Clamp coords
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img.shape[1], xmax), min(img.shape[0], ymax)
            crop = img[ymin:ymax, xmin:xmax]

        # Generate
        pixel_values = processor(crop, return_tensors="pt").pixel_values.cuda()
        
        with torch.no_grad():
            # --- EXPLICIT GENERATION CONFIG ---
            # Pass the token ID directly here to ensure the model knows the start/end point
            generated_ids = model.generate(
                pixel_values, 
                max_length=300, 
                num_beams=4,
                decoder_start_token_id=tokenizer.cls_token_id,
                eos_token_id=tokenizer.sep_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        pred_raw = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        truth_raw = item['text']
        
        # Normalize
        pred_clean = normalize_jp_text(pred_raw)
        truth_clean = normalize_jp_text(truth_raw)

        clean_preds.append(pred_clean)
        clean_refs.append(truth_clean)
        is_exact_match = (pred_clean == truth_clean)
        
        # Individual CER
        if len(truth_clean) > 0:
            sample_cer = cer_metric.compute(predictions=[pred_clean], references=[truth_clean])
        else:
            sample_cer = 1.0 if len(pred_clean) > 0 else 0.0
        
        results.append({
            "img_path": item['img_path'],
            "ground_truth_clean": truth_clean,
            "prediction_clean": pred_clean,
            "cer": sample_cer,
            "exact_match": is_exact_match,
            "raw_pred": pred_raw
        })

    # 5. Final Metrics
    print("\nCalculating metrics...")
    total_cer = cer_metric.compute(predictions=clean_preds, references=clean_refs)
    exact_matches = sum([1 for res in results if res['exact_match']])
    em_accuracy = exact_matches / len(results) if len(results) > 0 else 0

    print(f"\n======== REPORT: MODEL v{version_num} ========")
    print(f"Normalized CER:       {total_cer:.4f} (Lower is better)")
    print(f"Exact Match Accuracy: {em_accuracy*100:.2f}%")
    print("========================================")

    # 6. Save
    df = pd.DataFrame(results)
    df.to_csv(csv_output_name, index=False, encoding='utf-8-sig')
    print(f"Detailed results saved to: {csv_output_name}")
    
    # Print Top 10 Errors for quick check
    print("\n--- Top 10 Failures ---")
    df_err = df.sort_values(by="cer", ascending=False).head(10)
    for _, row in df_err.iterrows():
        print(f"GT:   {row['ground_truth_clean']}")
        print(f"Pred: {row['prediction_clean']}")
        print("-" * 20)

if __name__ == "__main__":
    run_auto_test()