import torch
from transformers import (
    VisionEncoderDecoderModel, 
    AutoTokenizer, 
    AutoImageProcessor,
    AutoModel,             
    AutoModelForCausalLM
)

def initialize_model():
    print("=== STARTING MODEL INITIALIZATION (MiT-b2 + BERT) ===")
    
    # 1. Define Model IDs
    # Encoder: SegFormer (MiT-b2) - Excellent for high-res details & dynamic shapes
    # Decoder: BERT-Japanese-Char-V3 - Great for Japanese Kanji
    encoder_id = "nvidia/mit-b2"
    decoder_id = "tohoku-nlp/bert-base-japanese-char-v3"
    
    # Define Target Image Size for Training
    # 512x512 is much better for Manga than 224x224.
    # MiT supports this resolution natively without issues.
    IMAGE_SIZE = 512
    
    # 2. Load Processor & Tokenizer
    print(f"\n[Step 1] Loading Processor and Tokenizer...")
    print(f" - Encoder: {encoder_id}")
    print(f" - Decoder: {decoder_id}")
    print(f" - Target Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # Load Processor (SegFormerImageProcessor)
    image_processor = AutoImageProcessor.from_pretrained(encoder_id)
    
    # Enforce 512x512 size for training stability
    image_processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    # Some processor versions use 'crop_size', setting it just in case
    if hasattr(image_processor, 'crop_size'):
        image_processor.crop_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)
    
    # 3. Load the Full Model (Pre-trained weights)
    print("\n[Step 2] Loading Pre-trained Weights (Full Model)...")
    encoder = AutoModel.from_pretrained(encoder_id)
    encoder.config.hidden_size = encoder.config.hidden_sizes[-1]
    decoder = AutoModelForCausalLM.from_pretrained(decoder_id, is_decoder=True, add_cross_attention=True)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    # IMPORTANT: Update model config to match our desired image size
    # This ensures the model knows we are using 512x512
    model.config.encoder.image_size = IMAGE_SIZE
    
    initial_params = model.num_parameters() / 1_000_000
    print(f" -> Initial Size (12-layer Decoder): {initial_params:.2f} M params")
    
    # 4. OPTIMIZATION: Truncate Decoder Layers
    # We reduce the BERT decoder from 12 layers to 4 layers.
    # This keeps the model lightweight and fast.
    target_decoder_layers = 4
    print(f"\n[Step 3] Optimizing Model: Truncating Decoder to {target_decoder_layers} layers...")
    
    # Slice the layer list to keep only the first N layers
    model.decoder.bert.encoder.layer = model.decoder.bert.encoder.layer[:target_decoder_layers]
    
    # Update config to match the new architecture
    model.config.decoder.num_hidden_layers = target_decoder_layers
    model.decoder.config.num_hidden_layers = target_decoder_layers
    
    # 5. Configure Model Settings
    print("\n[Step 4] Configuring Special Tokens & Generation...")
    
    # Linking Tokenizer IDs to Model Config
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Generation Config (Optimized for OCR)
    model.config.max_length = 300           # Max length of a bubble text
    model.config.num_beams = 4              # Beam search width for better accuracy
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    
    # 6. Final Verification
    final_params = model.num_parameters() / 1_000_000
    print("\n" + "="*40)
    print("MODEL READY FOR TRAINING")
    print("="*40)
    print(f"Encoder:            {encoder_id} (SegFormer)")
    print(f"Decoder:            Truncated BERT ({target_decoder_layers} layers)")
    print(f"Total Parameters:   {final_params:.2f} M")
    print(f"Input Image Size:   {image_processor.size}")
    
    return model, tokenizer, image_processor

if __name__ == "__main__":
    model, tokenizer, processor = initialize_model()