import os
import gdown

BASE_DIR = os.getcwd()
if not BASE_DIR.endswith('bubble-segmentation-final-deep-learning'):
    raise ValueError('To run this you should be in the bubble-segmentation-final-deep-learning directory')

MODEL_DIR = os.path.join(BASE_DIR, 'models', 'bubble-detection') 

model_dir_dict = {
    'yolo_scratch': os.path.join(MODEL_DIR, 'model-scratch-manga-segmentation', 'yolo_outputs'),
    'detectron2': os.path.join(MODEL_DIR, 'detectron2', 'output_balloon_segmentation_v3'),
    'deeplabv3': os.path.join(MODEL_DIR, 'DEEPLABv3'),
    'unet': os.path.join(MODEL_DIR, 'UNET')
}

path_file_dict = {
    'yolo_scratch': "https://drive.google.com/file/d/1pCSfEszGmHxmkxUwqNOz2A9bOoSSzpi4/view?usp=drive_link",
    'detectron2': "https://drive.google.com/file/d/1JXJ99PFviYwB4PENSkDUcSuzIMy65qUi/view?usp=sharing",
    'deeplabv3': "https://drive.google.com/file/d/1DqqjKqS8Sgc08TWfjkApU3yumHJq37fP/view?usp=sharing",
    'unet': "https://drive.google.com/file/d/107dMAAXMIk3WNWLxENAZpiy7lrovEoXZ/view?usp=sharing"
}

output_filename_dict = {
    'yolo_scratch': 'yolo_best.pth',
    'detectron2': 'model_final.pth',
    'deeplabv3': 'best_deeplabv3_model.pth',
    'unet': 'best_unet_model.pth'
}

for key in model_dir_dict.keys():
    dst_dir = model_dir_dict[key]
    url = path_file_dict[key]
    output_file = os.path.join(dst_dir, output_filename_dict[key])
    
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"Downloading {key}...")
    saved_path = gdown.download(
        url=url,
        output=output_file,
        fuzzy=True,
        quiet=False
        # resume=False  # Force overwrite/re-download
    )
    print(f"  Saved to: {saved_path}\n")

