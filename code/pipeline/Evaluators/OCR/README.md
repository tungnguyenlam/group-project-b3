# MangaOCR Evaluator CLI

CLI tool để đánh giá hiệu suất OCR trên manga sử dụng Manga109 dataset.

## Cài đặt

```bash
cd /mnt/e/B3/group_prj/group-project-b3
```

## Cách sử dụng

### 1. Đánh giá cơ bản (so sánh text bbox vs bubble bbox)

```bash
python code/pipeline/Evaluators/OCR/evaluate_manga_ocr.py \
    --manga-name AisazuNihaIrarenai \
    --max-images 10
```

### 2. Chỉ parse XML sang JSON (không evaluate)

```bash
python code/pipeline/Evaluators/OCR/evaluate_manga_ocr.py \
    --manga-name AisazuNihaIrarenai \
    --parse-only
```

### 3. Đánh giá chỉ với text bbox

```bash
python code/pipeline/Evaluators/OCR/evaluate_manga_ocr.py \
    --manga-name AisazuNihaIrarenai \
    --bbox-type text \
    --max-images 10
```

### 4. Đánh giá chỉ với bubble bbox

```bash
python code/pipeline/Evaluators/OCR/evaluate_manga_ocr.py \
    --manga-name AisazuNihaIrarenai \
    --bbox-type bubble \
    --max-images 10
```

### 5. Đánh giá toàn bộ volume với verbose output

```bash
python code/pipeline/Evaluators/OCR/evaluate_manga_ocr.py \
    --manga-name AisazuNihaIrarenai \
    --verbose
```

### 6. Chỉ định device cụ thể

```bash
# Sử dụng CUDA
python code/pipeline/Evaluators/OCR/evaluate_manga_ocr.py \
    --manga-name AisazuNihaIrarenai \
    --device cuda

# Sử dụng CPU
python code/pipeline/Evaluators/OCR/evaluate_manga_ocr.py \
    --manga-name AisazuNihaIrarenai \
    --device cpu
```

## Tham số

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--manga-name` | Tên volume truyện (bắt buộc) | - |
| `--max-images` | Số lượng trang tối đa để test | All |
| `--bbox-type` | Loại bbox: `text`, `bubble`, `compare` | `compare` |
| `--verbose` | Hiển thị chi tiết predictions | False |
| `--parse-only` | Chỉ parse XML, không evaluate | False |
| `--device` | Device: `cpu`, `cuda`, `mps`, `auto` | `auto` |
| `--batch-size` | Batch size cho evaluation | 1 |
| `--xml-dir` | Thư mục chứa XML annotations | `data/Manga109.../annotations` |
| `--images-dir` | Thư mục chứa hình ảnh manga | `data/Manga109.../images` |
| `--output-dir` | Thư mục output cho JSON | `data/MangaOCR/jsons_processed` |

## Output

Script tạo ra:

1. **JSON file**: `data/MangaOCR/jsons_processed/{manga_name}.json`
   - Parsed annotations trong COCO format

2. **Results file**: `data/MangaOCR/jsons_processed/{manga_name}_*_results.txt`
   - Text bbox results: `{manga_name}_text_results.txt`
   - Bubble bbox results: `{manga_name}_bubble_results.txt`
   - Comparison results: `{manga_name}_comparison_results.txt`

## Ví dụ Output

```
======================================================================
MangaOCR Evaluation - AisazuNihaIrarenai
======================================================================
Device: cuda
XML: data/Manga109_released_2023_12_07/annotations/AisazuNihaIrarenai.xml
Images: data/Manga109_released_2023_12_07/images/AisazuNihaIrarenai
Max images: 10
======================================================================

[1/2] Parsing XML annotations...
Parsing AisazuNihaIrarenai...
  Total pages: 10
  Total text annotations: 245
  Saved to data/MangaOCR/jsons_processed/AisazuNihaIrarenai.json

[2/2] Evaluating OCR...

======================================================================
EVALUATING WITH TEXT BBOX
======================================================================
Starting OCR evaluation with text bboxes...
100%|████████████████████████████████████████| 10/10 [00:45<00:00,  4.52s/it]

============================================================
OCR EVALUATION METRICS (TEXT BBOX)
╒═══════════════════════════════╤═════════╕
│ Metric                        │ Value   │
╞═══════════════════════════════╪═════════╡
│ BBox Type                     │ TEXT    │
├───────────────────────────────┼─────────┤
│ Character Error Rate (CER)    │ 0.1234  │
├───────────────────────────────┼─────────┤
│ Word Error Rate (WER)         │ 0.2345  │
├───────────────────────────────┼─────────┤
│ Number of Text Samples        │ 245     │
╘═══════════════════════════════╧═════════╛
============================================================

[... Similar output for BUBBLE BBOX ...]

============================================================
COMPARISON RESULTS
╒═════════╤════════════╤═══════════════╤══════════════╕
│         │ Text BBox  │ Bubble BBox   │ Difference   │
╞═════════╪════════════╪═══════════════╪══════════════╡
│ CER     │ 0.1234     │ 0.1456        │ 0.0222       │
├─────────┼────────────┼───────────────┼──────────────┤
│ WER     │ 0.2345     │ 0.2789        │ 0.0444       │
├─────────┼────────────┼───────────────┼──────────────┤
│ Samples │ 245        │ 245           │              │
╘═════════╧════════════╧═══════════════╧══════════════╛
============================================================

✓ Results saved to: data/MangaOCR/jsons_processed/AisazuNihaIrarenai_comparison_results.txt
✓ Evaluation complete!
```

## Workflow

1. **Parse XML → JSON**: Chuyển đổi Manga109 XML annotations sang COCO format
2. **Load Data**: Đọc JSON và lọc annotations có text
3. **Create Dataset**: Tạo dataset với loại bbox được chọn
4. **Evaluate**: Chạy OCR và tính CER/WER metrics
5. **Save Results**: Lưu kết quả vào file text

## Tips

- Dùng `--max-images 10` để test nhanh
- Dùng `--verbose` để debug predictions
- Dùng `--parse-only` nếu chỉ cần convert XML sang JSON
- So sánh `text` vs `bubble` bbox để thấy impact của bbox quality lên OCR performance
