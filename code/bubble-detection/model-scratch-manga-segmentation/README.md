# Manga Segmentation Project

This project is a personal study on implementing and training custom segmentation models,  YOLOv11 (try to mimic), for identifying text bubbles in manga pages.
# This project I do for studying and fun =))

## Repository structure

```
.
├── architecture_yolo
│   ├── assigner.py
│   ├── backbone.py
│   ├── blocks.py
│   ├── __init__.py
│   ├── yolo11_blocks.py
│   ├── yolo_head.py
│   └── yolo_seg.py
├── data_pipeline
│   ├── base_dataset.py
│   ├── data_loader_factory.py
│   ├── __init__.py
│   ├── manga_dataset.py
│   └── transforms.py
├── Manga109_released_2023_12_07
│   └── Manga109_released_2023_12_07
│       └── images
├── MangaSegmentation
│   ├── jsons
│   ├── jsons_processed
├── outputs_yolo
├── post_process_yolo
│   ├── box_utils.py
│   ├── debug.py
│   ├── loss.py
│   ├── __init__.py
│   ├── processor.py
│   └── evaluation_metric.py
├── README.md
├── rle-to-polygon.py
├── run_yolo
│   ├── info_yolo.py
│   ├── test_yolo.py
│   └── train_yolo.py
└── visualize_data.py
```

## Custom YOLOv11 Nano Segment

This section details how to train and evaluate the custom YOLOv11 segmentation model.

### 1. Check Model Information

Before running, you can display a summary of the model architecture, including the number of parameters and layers.

```bash
python -m run_yolo.info_yolo
```

### 2. Training the Model

This script will train the model, periodically run evaluations, save confusion matrices, create a `result.csv` file with training metrics, and save the best-performing model weights (`yolo_best.pth`) to the output directory.

#### Simple Training Command

To start training with all the default parameters defined in the script:

```bash
python -m run_yolo.train_yolo
```

#### Custom Training Command

You can customize the training run by specifying arguments. Here is an example:

```bash
python -m run_yolo.train_yolo \
    --num-epochs 50 \
    --batch-size 8 \
    --accumulation-steps 4 \
    --lr 0.0005 \
    --output-dir ./outputs_yolo_custom_run \
    --epoch-val 2 \
    --debug
```

#### Training Arguments

| Argument | Description | Default Value |
| :--- | :--- | :--- |
| **Data Paths** | | |
| `--json-dir` | Path to the directory with processed JSON annotations. | `MangaSegmentation/jsons_processed` |
| `--image-root`| Path to the root directory of the images. | `Manga109_.../images` |
| `--output-dir`| Directory to save checkpoints, logs, and results. | `./outputs_yolo` |
| **Model & Training** | | |
| `--num-classes` | Number of classes, including the background. | `2` |
| `--num-epochs` | Total number of epochs to train for. | `30` |
| `--batch-size` | Batch size for training. **Adjust based on VRAM.** | `16` |
| `--eval-batch-size` | Batch size for evaluation to conserve VRAM. | `4` |
| `--accumulation-steps` | Number of steps to accumulate gradients before updating weights. Effective batch size = `batch-size * accumulation-steps`. | `2` |
| `--num-workers` | Number of CPU workers for data loading. | `8` |
| `--lr` | The maximum learning rate for the OneCycleLR scheduler. | `0.001` |
| **Loss Weights** | | |
| `--box-weight` | Weight for the bounding box regression loss. | `7.0` |
| `--cls-weight` | Weight for the classification loss. | `3.0` |
| `--mask-weight`| Weight for the segmentation mask loss. | `7.0` |
| `--dfl-weight` | Weight for the Distribution Focal Loss (refines box edges). | `1.5` |
| **Validation** | | |
| `--conf-thresh`| Confidence threshold for predictions during validation. | `0.9` |
| `--nms-thresh` | Non-Maximum Suppression (NMS) IoU threshold for validation. | `0.9` |
| `--mask-batch-size` | Batch size for mask processing during validation to manage VRAM. | `16` |
| `--epoch-val` | Run validation every `N` epochs. | `5` |
| **Debugging** | | |
| `--debug` | Enables debug mode to visualize model outputs and check assigner logic before training starts. | `False` |

---

### 3. Testing the Model

This script evaluates a trained model on the validation dataset using the standard COCO evaluation protocol.

#### Simple Testing Command

To test a model, you must provide the path to the trained weights (`.pth` file).

```bash
python -m run_yolo.test_yolo --model-path ./outputs_yolo/yolo_best.pth
```

#### Custom Testing Command

You can specify different thresholds and save the results to a JSON file.

```bash
python -m run_yolo.test_yolo \
    --model-path ./outputs_yolo/yolo_best.pth \
    --conf-thresh 0.85 \
    --nms-thresh 0.8 \
    --batch-size 8 \
    --save-results
```

#### Testing Arguments

| Argument | Description | Default Value |
| :--- | :--- | :--- |
| **Required** | | |
| `--model-path` | **(Required)** Path to the trained `.pth` model file for evaluation. | `None` |
| **Data Paths** | | |
| `--json-dir` | Path to the directory with processed JSON annotations. | `MangaSegmentation/jsons_processed`|
| `--image-root`| Path to the root directory of the images. | `Manga109_.../images` |
| **Evaluation** | | |
| `--batch-size` | Batch size for evaluation. **Adjust based on VRAM.** | `4` |
| `--num-workers` | Number of CPU workers for the data loader. | `8` |
| `--conf-thresh` | Confidence threshold for post-processing. | `0.98` |
| `--nms-thresh` | NMS IoU threshold for post-processing. | `0.9` |
| **Performance** | | |
| `--mask-batch-size` | Batch size for mask processing during inference. **Crucial for VRAM management.** | `4` |
| `--resize-batch-size` | Processes N masks at a time for resizing operations to save memory. | `16` |
| **Output** | | |
| `--save-results`| If set, saves the final COCO evaluation metrics to a JSON file. | `False` |

> **Note**: Carefully adjust `batch-size`, `eval-batch-size`, and `mask-batch-size` based on the available VRAM of your GPU to avoid out-of-memory errors.

---

