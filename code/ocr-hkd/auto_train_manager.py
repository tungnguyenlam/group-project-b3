import os
import re
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import evaluate
import random
import albumentations as A
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    default_data_collator,
    AutoTokenizer,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
    AutoModel,              
    AutoModelForCausalLM    
)

# ================= CONFIGURATION =================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
JSON_PATH = os.path.join(DATA_DIR, 'manga109_ocr_dataset.json')
IMAGE_ROOT = os.path.join(DATA_DIR, 'Manga109_released_2023_12_07', 'images')

# Model Base IDs (Only used for v1)
ENCODER_ID = "nvidia/mit-b2"
DECODER_ID = "tohoku-nlp/bert-base-japanese-char-v3"
IMAGE_SIZE = 512

# Experiment Config
USE_MASK_AUGMENTATION = False
BATCH_SIZE = 16

# ================= FIX ERROR DIMENSION FOR MIT-B2====================
class SegformerFeatureCorrector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Explicitly copy these attributes
        self.config = encoder.config 
        self.main_input_name = getattr(encoder, "main_input_name", "pixel_values")
        
    def forward(self, pixel_values, **kwargs):
        outputs = self.encoder(pixel_values, **kwargs)
        last_hidden_state = outputs.last_hidden_state 
        # Convert from [Batch, Channels, H, W] to [Batch, SeqLen, HiddenSize]
        outputs.last_hidden_state = last_hidden_state.flatten(2).transpose(1, 2)
        return outputs
    
    def get_output_embeddings(self):
        """SegFormer encoders don't have output embeddings. Return None."""
        return None
    
    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        encoder = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(encoder)

# ================= UTILS: VERSION MANAGER =================
def get_next_version():
    """
    Scans the models folder for the latest version.
    Returns: (current_version_path, next_version_name, is_fresh_start)
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    # Find all folders with the format "manga_ocr_v{number}"
    existing_versions = []
    for d in os.listdir(MODELS_DIR):
        match = re.match(r"manga_ocr_v(\d+)", d)
        if match:
            existing_versions.append(int(match.group(1)))
            
    if not existing_versions:
        # No version yet -> Train Fresh -> Output v1
        return None, "manga_ocr_v1", True
    
    # Old version available -> Take highest version as input -> Output v+1
    max_ver = max(existing_versions)
    input_path = os.path.join(MODELS_DIR, f"manga_ocr_v{max_ver}")
    output_name = f"manga_ocr_v{max_ver + 1}"
    
    return input_path, output_name, False

# ================= UTILS: MODEL INITIALIZER =================
def init_fresh_model():
    """Initialize Base Model (MiT-b2 + BERT) with Config Fixes"""
    print(f"[Init] Loading Base Models: {ENCODER_ID} + {DECODER_ID}")
    print(f"[Init] Target Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # 1. Load Processor & Tokenizer
    image_processor = AutoImageProcessor.from_pretrained(ENCODER_ID)
    # Enforce 512x512 size
    image_processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    if hasattr(image_processor, 'crop_size'):
        image_processor.crop_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
        
    tokenizer = AutoTokenizer.from_pretrained(DECODER_ID, trust_remote_code=True)
    
    # 2. Load Encoder & Fix Config (The hidden_size fix)
    raw_encoder = AutoModel.from_pretrained(ENCODER_ID)
    # SegFormer uses 'hidden_sizes' (list), but VisionEncoderDecoder expects 'hidden_size'
    raw_encoder.config.hidden_size = raw_encoder.config.hidden_sizes[-1]
    # Update image size in config
    raw_encoder.config.image_size = IMAGE_SIZE

    # Apply to convert [batch_size, channels, height, width] -> [batch_size, seq_len, hidden_size]
    encoder = SegformerFeatureCorrector(raw_encoder) 

    # 3. Load Decoder
    decoder = AutoModelForCausalLM.from_pretrained(DECODER_ID, is_decoder=True, add_cross_attention=True)

    # 4. Combine into VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    # Update global config
    model.config.encoder.image_size = IMAGE_SIZE
    
    # 5. Truncate Decoder Layers (12 -> 4)
    print("[Init] Truncating Decoder to 4 layers...")
    model.decoder.bert.encoder.layer = model.decoder.bert.encoder.layer[:4]
    model.config.decoder.num_hidden_layers = 4
    model.decoder.config.num_hidden_layers = 4
    
    return model, tokenizer, image_processor

# ================= DATASET CLASS =================
class MangaOCRDataset(Dataset):
    def __init__(self, json_path, img_root, processor, tokenizer, split='train', use_mask=False, transform=None):
        self.img_root = img_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.transform = transform
        self.use_mask = use_mask
        
        with open(json_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        self.data = [item for item in full_data if item['split'] == split]
        print(f"Dataset '{split}' loaded: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def apply_mask_processing(self, img, bbox, mask_polys):
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

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_root, item['img_path'])
        image = cv2.imread(img_path)
        if image is None: return self.__getitem__((idx + 1) % len(self.data))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_mask and item.get('mask'):
            crop = self.apply_mask_processing(image, item['bbox'], item['mask'])
        else:
            xmin, ymin, xmax, ymax = map(int, item['bbox'])
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(image.shape[1], xmax), min(image.shape[0], ymax)
            crop = image[ymin:ymax, xmin:xmax]

        if self.transform:
            crop = self.transform(image=crop)['image']

        pixel_values = self.processor(crop, return_tensors="pt").pixel_values.squeeze()
        labels = self.tokenizer(
            item['text'], padding="max_length", max_length=300, truncation=True
        ).input_ids
        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in labels]
        
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

# ================= METRICS =================
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    # Decode
    # Note: tokenizer object needs to be accessible globally or passed
    pred_str = tokenizer_global.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer_global.pad_token_id
    label_str = tokenizer_global.batch_decode(labels_ids, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    # Print sample
    print(f"\n[Sample] Pred: {pred_str[0]}")
    print(f"[Sample] True: {label_str[0]}")
    
    return {"cer": cer}

# ================= MAIN AUTOMATION LOGIC =================
if __name__ == "__main__":
    # 1. Check Versions
    input_path, output_name, is_fresh_start = get_next_version()
    output_path = os.path.join(MODELS_DIR, output_name)
    
    print("="*50)
    if is_fresh_start:
        print(f"🚀 MODE: FRESH START (Creating {output_name})")
        # Config for Fresh Train
        LEARNING_RATE = 1e-5
        EPOCHS = 5
        WARMUP = 0.05
    else:
        print(f"🔄 MODE: CONTINUE TRAINING")
        print(f"📥 Input:  {input_path}")
        print(f"📤 Output: {output_path}")
        # Config for Continue Train (Lower, more technical)
        LEARNING_RATE = 1e-6
        EPOCHS = 3
        WARMUP = 0.1
    print("="*50)

    # 2. Load Model & Tokenizer
    if is_fresh_start:
        model, tokenizer, processor = init_fresh_model()
    else:
        print(f"Loading previous model from {input_path}...")
        tokenizer = AutoTokenizer.from_pretrained(input_path)
        processor = AutoImageProcessor.from_pretrained(input_path)
        # ignore_mismatched_sizes=True to be safe
        model = VisionEncoderDecoderModel.from_pretrained(input_path, ignore_mismatched_sizes=True)

    # 3. Apply Critical Model Configs (Apply for both cases to be sure)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 300
    model.config.num_beams = 4
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    
    # Make tokenizer global for metrics function
    global tokenizer_global
    tokenizer_global = tokenizer

    # 4. Augmentation Pipeline
    train_transform = A.Compose([
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.Rotate(limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Blur(blur_limit=3, p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        A.ImageCompression(quality_range=(50, 90), p=0.2),
        A.RandomScale(scale_limit=0.1, p=0.3),
    ])

    # 5. Datasets
    print("Loading Datasets...")
    train_dataset = MangaOCRDataset(JSON_PATH, IMAGE_ROOT, processor, tokenizer, split='train', use_mask=USE_MASK_AUGMENTATION, transform=train_transform)
    val_dataset = MangaOCRDataset(JSON_PATH, IMAGE_ROOT, processor, tokenizer, split='test', use_mask=USE_MASK_AUGMENTATION)
    random.seed(42) # Fixed seed
    val_dataset.data = random.sample(val_dataset.data, 1000)

    # 6. Training Arguments (Dynamic based on mode)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2, # Effective Batch Size = 32
        per_device_eval_batch_size=BATCH_SIZE,
        
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="cosine",    # Always use Cosine for best convergence
        warmup_ratio=WARMUP,
        
        max_grad_norm=1.0,
        fp16=True,
        
        eval_strategy="steps",
        eval_steps=1000,

        logging_strategy="steps",
        logging_steps=100,

        save_steps=2000,
        save_total_limit=2,
        
        predict_with_generate=True,
        load_best_model_at_end=True,         # Load the best checkpoint (lowest CER) when done
        metric_for_best_model="cer",
        generation_max_length=300,
        dataloader_num_workers=8,
        report_to="none"
    )

    # 7. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    # 8. Run
    print("\nStarting Training...")
    trainer.train()

    # 9. Save
    print(f"Saving final model to {output_path}...")
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("\n✅ DONE! You can run this script again to generate the NEXT version.")