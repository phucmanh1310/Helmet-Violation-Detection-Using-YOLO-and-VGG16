# 🚀 KẾ HOẠCH CHI TIẾT CẢI THIỆN MODEL - NHÓM 4 NGƯỜI

> **Mục tiêu**: Hiểu sâu về dự án Helmet Violation Detection và cải thiện model để báo cáo cho giáo viên
>
> **Thời gian**: 4 tuần (có thể điều chỉnh)
>
> **Thành viên**: 4 người (Person A, B, C, D)

---

## 📋 THÔNG TIN DỰ ÁN

### Repository gốc

- **Link**: https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16
- **Fork của bạn**: https://github.com/phucmanh1310/Helmet-Violation-Detection-Using-YOLO-and-VGG16

### Kiến trúc hệ thống hiện tại

```
Input (Image/Video)
    ↓
┌─────────────────────────────────────┐
│  Model 1: Motorcycle Detection      │
│  - YOLOv8 (Motov10l.pt)            │
│  - Detect: Motorcycles in scene     │
└─────────────────────────────────────┘
    ↓ (Crop motorcycles)
┌─────────────────────────────────────┐
│  Model 2: Helmet & LP Detection     │
│  - YOLOv8 (HelmetLP.pt)            │
│  - Detect: helmet/no-helmet/LP      │
└─────────────────────────────────────┘
    ↓ (Crop license plates)
┌─────────────────────────────────────┐
│  Model 3: OCR License Plate         │
│  - EasyOCR                          │
│  - Read: License plate text         │
└─────────────────────────────────────┘
    ↓
Output (Violations + LP numbers)
```

---

## 📊 PHÂN TÍCH VẤN ĐỀ HIỆN TẠI (Quan trọng để hiểu!)

### ❌ Vấn đề 1: Chỉ nhận diện được góc nhìn từ camera giao thông

**Nguyên nhân:**

- Dataset training chỉ có ảnh từ camera giao thông (góc nhìn từ trên cao 45-60°)
- Thiếu đa dạng góc độ: từ phía trước, sau, bên, góc thấp
- Model học theo bias của data → generalization kém

**Ví dụ thực tế:**

- ✅ Nhận diện OK: Camera giao thông nhìn từ trên cao
- ❌ Nhận diện kém: Ảnh chụp từ điện thoại góc ngang
- ❌ Nhận diện kém: Ảnh từ dashcam xe hơi

### ❌ Vấn đề 2: Độ chính xác thấp, sai sót nhiều

**Nguyên nhân cụ thể:**

1. **Dataset quá nhỏ**:

   - Có thể chỉ 500-1000 ảnh
   - Cần tối thiểu 3000-5000 ảnh cho kết quả tốt

2. **Labeling không chính xác**:

   - Bounding box không khít object
   - Nhầm class (helmet vs no-helmet)
   - Miss label một số objects

3. **Model architecture không phù hợp**:

   - Có thể đang dùng YOLOv8n (nano) - quá nhỏ
   - Cần YOLOv8m hoặc YOLOv8l

4. **Training chưa đủ**:

   - Epochs có thể chỉ 50-100
   - Chưa convergence
   - Early stopping quá sớm

5. **Class imbalance**:
   - Ví dụ: 80% có mũ, 20% không mũ
   - Model bias về class nhiều hơn

### ❌ Vấn đề 3: Các lý do khác

**Điều kiện môi trường:**

- Ánh sáng: ban ngày sáng OK, tối/mưa/ngược sáng kém
- Khoảng cách: gần OK, xa >20m kém
- Occlusion: người bị che khuất một phần

**Đặc thù Việt Nam:**

- Nhiều người trên xe (2-3 người)
- Đội mũ không cài quai
- Loại mũ đa dạng (fullface, 3/4, nửa đầu)

---

## 🎯 LỘ TRÌNH HỌC TẬP & THỰC HÀNH 4 TUẦN

---

## TUẦN 1: TÌM HIỂU LÝ THUYẾT & PHÂN TÍCH DỰ ÁN

### 📅 Ngày 1-2: Hiểu kiến trúc & thuật toán (CẢ NHÓM)

#### Buổi sáng: Object Detection với YOLO

**Person A - Tìm hiểu YOLO Architecture:**

```
□ Đọc paper: "YOLOv8: A Comprehensive Guide"
□ Link: https://docs.ultralytics.com/
□ Ghi chú:
  - YOLO là gì? (You Only Look Once)
  - Khác gì với R-CNN, Fast R-CNN?
  - YOLOv8 có gì mới so với YOLOv5, YOLOv7?
  - Các phiên bản: nano, small, medium, large, xlarge

□ Tạo file: docs/YOLO_Architecture.md
□ Tóm tắt:
  - Cách YOLO detect objects (grid system)
  - Anchor boxes
  - Loss function (box loss, cls loss, dfl loss)
  - NMS (Non-Maximum Suppression)
```

**Person B - Tìm hiểu Transfer Learning & Training:**

```
□ Đọc: "Transfer Learning with YOLO"
□ Link: https://docs.ultralytics.com/modes/train/
□ Ghi chú:
  - Transfer Learning là gì?
  - Pretrained weights (COCO dataset)
  - Fine-tuning vs Training from scratch
  - Hyperparameters: lr, epochs, batch size
  - Data augmentation techniques

□ Tạo file: docs/Transfer_Learning.md
□ Thực hành nhỏ:
  - Chạy thử train YOLO với dummy data
  - Code: python train_demo.py
```

**Person C - Tìm hiểu OCR & EasyOCR:**

```
□ Đọc: "OCR for License Plate Recognition"
□ Link: https://github.com/JaidedAI/EasyOCR
□ Ghi chú:
  - OCR là gì? (Optical Character Recognition)
  - EasyOCR vs Tesseract vs PaddleOCR
  - Preprocessing cho OCR (thresholding, denoising)
  - Biển số Việt Nam format: XX-YYYYY

□ Tạo file: docs/OCR_Analysis.md
□ Thực hành:
  - Test EasyOCR với ảnh biển số mẫu
  - Code: python test_ocr.py
```

**Person D - Phân tích code hiện tại:**

```
□ Review toàn bộ source code:
  - Source/main_app.py
  - Source/_Motobike.py
  - Source/_LP_Helmet.py
  - Source/ui_app.py

□ Tạo file: docs/Code_Analysis.md
□ Vẽ sơ đồ luồng xử lý (flowchart)
□ Liệt kê functions chính và mục đích
□ Tìm bottlenecks và vấn đề trong code
```

#### Buổi chiều: Họp nhóm & chia sẻ kiến thức

**13:00 - 15:00: Presentation nội bộ**

```
- Mỗi người trình bày phần mình tìm hiểu (15 phút/người)
- Hỏi đáp, thảo luận
- Ghi chú vào file: docs/DAY1-2_Meeting_Notes.md
```

**15:00 - 17:00: Chạy thử hệ thống hiện tại**

```
□ Setup môi trường (cả nhóm)
□ Chạy main_app.py với ảnh test
□ Chạy ui_app.py (giao diện web)
□ Test với:
  - 10 ảnh góc cao (camera giao thông)
  - 10 ảnh góc ngang
  - 10 ảnh góc khác

□ Ghi nhận kết quả vào: results/current_performance.xlsx
  Columns: Image_Name | Angle | Detected_Moto | Detected_Helmet | Correct? | Notes
```

---

### 📅 Ngày 3-4: Đánh giá dataset & tìm nguồn data mới

#### Person A: Phân tích dataset hiện tại

**Công việc cụ thể:**

```python
# File: analysis/analyze_current_dataset.py

import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

# 1. Đếm số lượng ảnh
train_dir = 'data/LP-Helmet.v2i.yolov8/train/images'
val_dir = 'data/LP-Helmet.v2i.yolov8/valid/images'
test_dir = 'data/LP-Helmet.v2i.yolov8/test/images'

print(f"Train: {len(os.listdir(train_dir))} images")
print(f"Val: {len(os.listdir(val_dir))} images")
print(f"Test: {len(os.listdir(test_dir))} images")

# 2. Phân tích class distribution
def count_classes(label_dir):
    class_counts = Counter()
    for label_file in os.listdir(label_dir):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
    return class_counts

train_classes = count_classes('data/LP-Helmet.v2i.yolov8/train/labels')
print("\nClass distribution:")
print(f"Class 0 (helmet): {train_classes[0]}")
print(f"Class 1 (no-helmet): {train_classes[1]}")
print(f"Class 2 (LP): {train_classes[2]}")

# 3. Vẽ biểu đồ
plt.bar(train_classes.keys(), train_classes.values())
plt.xlabel('Class ID')
plt.ylabel('Count')
plt.title('Class Distribution in Training Set')
plt.savefig('analysis/class_distribution.png')

# 4. Phân tích image quality
# - Resolution
# - Brightness
# - Blur level

# Output: analysis/dataset_analysis_report.md
```

**Checklist:**

```
□ Chạy script phân tích
□ Tạo báo cáo: analysis/dataset_analysis_report.md
  - Tổng số ảnh: ___
  - Train/Val/Test split: ___/___/___
  - Class distribution: helmet ___, no-helmet ___, LP ___
  - Imbalance ratio: ___
  - Average image resolution: ___
  - Issues found: ___

□ Kết luận: Dataset có đủ lớn không? Cân bằng không?
```

#### Person B: Tìm datasets online (Roboflow)

**Hướng dẫn chi tiết:**

```
Bước 1: Vào Roboflow Universe
  https://universe.roboflow.com/

Bước 2: Tìm kiếm datasets
  Từ khóa:
  - "helmet detection"
  - "motorcycle helmet"
  - "safety helmet"
  - "hard hat"

Bước 3: Chọn datasets
  Tiêu chí:
  ✅ Số lượng: >1000 images
  ✅ Quality: có preview, nhìn giống data VN
  ✅ Format: YOLOv8
  ✅ License: Public/Free

Bước 4: Download top 5 datasets
  Lưu vào: data/external/roboflow/
  Format: YOLOv8

Bước 5: Ghi chép
  File: data/external/roboflow/datasets_info.md

  | Dataset Name | Images | Classes | Link | Notes |
  |--------------|--------|---------|------|-------|
  | Dataset1     | 2000   | 3       | ...  | Good  |
  | ...          | ...    | ...     | ...  | ...   |
```

**Checklist:**

```
□ Download ít nhất 3 datasets (tổng >3000 ảnh)
□ Extract và kiểm tra format
□ Tạo file datasets_info.md
□ Upload lên Google Drive nhóm (nếu quá lớn)
```

#### Person C: Tìm datasets trên Kaggle

**Hướng dẫn chi tiết:**

````
Bước 1: Tạo tài khoản Kaggle (nếu chưa có)
  https://www.kaggle.com/

Bước 2: Tìm datasets
  Link trực tiếp:
  1. https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
  2. https://www.kaggle.com/datasets/andrewmvd/helmet-detection
  3. Tìm thêm: search "motorcycle helmet detection"

Bước 3: Download bằng Kaggle API
  ```bash
  pip install kaggle
  kaggle datasets download -d andrewmvd/hard-hat-detection
  unzip hard-hat-detection.zip -d data/external/kaggle/
````

Bước 4: Convert format (nếu cần)

- Nếu dataset format COCO → Convert sang YOLO
- Dùng script: scripts/convert_coco_to_yolo.py

```

**Checklist:**
```

□ Download 2-3 datasets từ Kaggle
□ Convert sang YOLOv8 format (nếu cần)
□ Tạo file: data/external/kaggle/datasets_info.md
□ Tổng hợp số lượng ảnh đã download

```

#### Person D: Thu thập data từ video YouTube

**Hướng dẫn chi tiết:**
```

Bước 1: Tìm video traffic cam Việt Nam
YouTube search:

- "Vietnam traffic cam"
- "Camera giao thông Việt Nam"
- "Traffic in Ho Chi Minh City"
- "Hanoi traffic"

Chọn video:
✅ Độ phân giải cao (1080p+)
✅ Góc nhìn đa dạng
✅ Nhiều xe máy
✅ Điều kiện khác nhau (sáng, tối, mưa)

Bước 2: Download video
Tool: youtube-dl hoặc 4K Video Downloader

```bash
pip install yt-dlp
yt-dlp -f best "https://youtube.com/watch?v=VIDEO_ID"
```

Bước 3: Extract frames

```python
# File: scripts/extract_frames.py
import cv2

def extract_frames(video_path, output_dir, frame_rate=5):
    """Extract 1 frame every 5 seconds"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * frame_rate

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            cv2.imwrite(f'{output_dir}/frame_{saved:05d}.jpg', frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Extracted {saved} frames")

# Usage
extract_frames('videos/traffic1.mp4', 'data/extracted_frames/video1/', frame_rate=5)
```

Bước 4: Lọc ảnh có xe máy

- Dùng model hiện tại để lọc
- Chỉ giữ ảnh có >1 xe máy
- Lưu vào: data/collected/youtube_frames/

```

**Checklist:**
```

□ Download 3-5 videos (mỗi video 5-10 phút)
□ Extract frames (tổng ~500-1000 frames)
□ Lọc frames có xe máy (dùng Motov10l.pt)
□ Lưu vào folder organized
□ Tạo file: data/collected/youtube_sources.md (ghi link videos)

```

#### Buổi chiều: Họp nhóm & tổng hợp

**15:00 - 17:00: Tổng hợp data đã thu thập**
```

□ Merge all datasets vào: data/merged_v1/
□ Tính tổng số ảnh: \_\_\_
□ Phân loại theo nguồn:

- Current dataset: \_\_\_ images
- Roboflow: \_\_\_ images
- Kaggle: \_\_\_ images
- YouTube: \_\_\_ images

□ Tạo spreadsheet: data/Data_Inventory.xlsx
Columns: Source | Images | Labeled? | Quality | Notes

```

---

### 📅 Ngày 5-6-7: Labeling data mới

#### Setup công cụ labeling (CẢ NHÓM)

**Option 1: Roboflow (Khuyến nghị)**
```

Bước 1: Tạo project

- Vào https://roboflow.com/
- Tạo account
- New Project → Object Detection
- Classes: helmet, no-helmet, LP

Bước 2: Upload ảnh cần label

- Upload batch 200-500 ảnh
- Organize thành folders

Bước 3: Label

- Draw bounding boxes
- Phím tắt:
  - 1: class helmet
  - 2: class no-helmet
  - 3: class LP
  - Space: next image

Bước 4: Export

- Format: YOLOv8
- Download về data/labeled/

````

**Option 2: LabelImg (Offline)**
```bash
pip install labelImg
labelImg data/unlabeled/ data/labeled/
````

#### Phân công labeling

**Person A: Label YouTube frames (500 ảnh)**

```
□ Folder: data/collected/youtube_frames/
□ Output: data/labeled/youtube/
□ Target: 500 ảnh trong 2 ngày
□ Mỗi ngày: 250 ảnh (~2 giờ, tốc độ 2 phút/ảnh)
□ Guidelines:
  - Bounding box phải khít object
  - Không skip ảnh khó
  - Note ảnh unclear vào file: unclear_images.txt
```

**Person B: Label Roboflow data (500 ảnh)**

```
□ Chọn 500 ảnh chưa label từ Roboflow downloads
□ Output: data/labeled/roboflow/
□ Target: 500 ảnh trong 2 ngày
□ Focus: ảnh góc khác, điều kiện khác
```

**Person C: Label Kaggle data (500 ảnh)**

```
□ Review và fix labels của Kaggle datasets
□ Một số dataset có label sai → cần sửa
□ Output: data/labeled/kaggle_fixed/
□ Target: Review 500 ảnh
```

**Person D: QA và merge datasets**

````
□ Quality Assurance:
  - Kiểm tra 10% labels của mỗi người
  - Tìm lỗi common (box quá lớn, sai class, miss object)

□ Merge datasets:
  ```python
  # File: scripts/merge_datasets.py
  # Merge all labeled data thành 1 dataset
  # Split: 70% train, 20% val, 10% test
  # Output: data/final_dataset_v1/
````

□ Tạo data.yaml:

```yaml
path: ../data/final_dataset_v1
train: train/images
val: valid/images
test: test/images

nc: 3
names: ["helmet", "no-helmet", "LP"]
```

````

#### Quality Control Guidelines (CẢ NHÓM phải đọc)

```markdown
# Labeling Guidelines

## Bounding Box Rules
✅ DO:
- Khít object, không để lộ quá nhiều background
- Bao gồm toàn bộ object (kể cả phần bị che)
- Consistent giữa các ảnh

❌ DON'T:
- Box quá lớn (nhiều background)
- Box quá nhỏ (cắt mất phần object)
- Skip objects nhỏ/xa

## Class Definition
- **helmet**: Người ĐANG ĐỘI mũ bảo hiểm (dù có cài quai hay không)
- **no-helmet**: Người KHÔNG ĐỘI mũ bảo hiểm (đầu trần hoặc đội mũ lưỡi trai)
- **LP**: Biển số xe rõ ràng, đọc được

## Edge Cases
- Mũ bị che khuất >50%: vẫn label nếu chắc chắn là mũ
- Nhiều người trên xe: label TẤT CẢ
- Người ở xa/nhỏ: vẫn label nếu nhìn thấy
- Mũ trong tay (không đội): KHÔNG label
````

---

## TUẦN 2: TRAINING VÀ THỰC NGHIỆM

### 📅 Ngày 8-9: Training baseline models

#### Person A: Train YOLOv8n (baseline)

### A. Thu thập thêm dữ liệu đa dạng

#### 📸 Đa dạng góc nhìn:

```
✅ Góc camera giao thông (từ trên cao 45°-60°)
✅ Góc ngang (0°-20°) - từ vỉa hè
✅ Góc thấp (từ dưới lên)
✅ Góc nghiêng trái/phải
✅ Góc phía sau
✅ Góc phía trước
```

#### 🌈 Đa dạng điều kiện:

```
✅ Ban ngày sáng
✅ Hoàng hôn/bình minh
✅ Ban đêm (có đèn)
✅ Trời mưa
✅ Nắng gắt (overexposed)
✅ Ngược sáng
```

#### 🏍️ Đa dạng kịch bản:

```
✅ 1 người trên xe
✅ 2 người trên xe
✅ 3 người trên xe (phổ biến ở VN)
✅ Người lớn + trẻ em
✅ Xe đứng yên
✅ Xe đang chạy (motion blur)
```

#### 🪖 Đa dạng loại mũ:

```
✅ Mũ bảo hiểm fullface
✅ Mũ bảo hiểm 3/4
✅ Mũ bảo hiểm nửa đầu
✅ Mũ lưỡi trai (không phải mũ BH)
✅ Không đội gì
✅ Đội mũ không cài quai
```

### B. Nguồn thu thập data

#### 🌐 Online Sources:

```python
# 1. Roboflow Universe
- Tìm kiếm: "helmet detection", "motorcycle helmet"
- Download datasets có sẵn và merge

# 2. Kaggle Datasets
- Safety Helmet Detection
- Motorcycle Helmet Detection Dataset

# 3. YouTube Videos
- Quay màn hình traffic cam
- Dashcam videos
- GoPro motorcycle videos

# 4. Google Images
- Sử dụng script download hàng loạt
```

#### 📹 Tự thu thập:

```
1. Quay video tại:
   - Giao lộ đông đúc
   - Trước cổng trường
   - Bãi giữ xe
   - Đường phố

2. Cắt frame từ video
3. Tự label bằng Roboflow/LabelImg
```

### C. Tăng cường dữ liệu (Data Augmentation)

```python
# File: augmentation_config.yaml
# Sử dụng trong YOLO training

augmentation:
  # Geometric transforms
  degrees: 15.0           # Xoay ảnh ±15°
  translate: 0.2          # Dịch chuyển 20%
  scale: 0.9              # Scale 90%-110%
  shear: 5.0              # Nghiêng
  perspective: 0.0003     # Phối cảnh
  flipud: 0.0             # Không lật dọc
  fliplr: 0.5             # Lật ngang 50%

  # Color transforms
  hsv_h: 0.015            # Thay đổi Hue
  hsv_s: 0.7              # Thay đổi Saturation
  hsv_v: 0.4              # Thay đổi Value/Brightness

  # Noise & effects
  mosaic: 1.0             # Mosaic augmentation
  mixup: 0.3              # Mixup augmentation
  copy_paste: 0.3         # Copy-paste
  blur: 0.01              # Motion blur
  noise: 0.02             # Gaussian noise
```

### D. Cân bằng dữ liệu (Balance Dataset)

```python
# Script: balance_dataset.py
import os
from collections import Counter
import shutil
import random

def balance_classes(label_dir):
    """Cân bằng số lượng ảnh có mũ vs không mũ"""

    class_counts = Counter()

    # Đếm số lượng mỗi class
    for label_file in os.listdir(label_dir):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

    print("Class distribution:", class_counts)

    # Tìm class có ít nhất
    min_count = min(class_counts.values())

    # Undersample hoặc oversample
    # ... implement logic here

    return class_counts

# Chạy
balance_classes('data/LP-Helmet.v2i.yolov8/train/labels')
```

**Target ratio:**

```
helmet : no_helmet : LP = 1 : 1 : 0.8
```

---

## 2️⃣ CẢI THIỆN KIẾN TRÚC MODEL

### A. Thử các phiên bản YOLO khác nhau

```python
# So sánh performance

# YOLOv8 (hiện tại)
model = YOLO('yolov8n.pt')  # nano - nhanh
model = YOLO('yolov8s.pt')  # small
model = YOLO('yolov8m.pt')  # medium - cân bằng
model = YOLO('yolov8l.pt')  # large - chính xác hơn
model = YOLO('yolov8x.pt')  # xlarge - chính xác nhất

# YOLOv9 (mới hơn)
model = YOLO('yolov9c.pt')
model = YOLO('yolov9e.pt')

# YOLOv10 (nhanh nhất)
model = YOLO('yolov10n.pt')
model = YOLO('yolov10m.pt')

# YOLO11 (mới nhất 2024)
model = YOLO('yolo11n.pt')  # ⭐ Khuyến nghị thử
model = YOLO('yolo11m.pt')
```

**Khuyến nghị:**

- **YOLOv8m** hoặc **YOLOv8l** cho độ chính xác tốt
- **YOLO11m** nếu muốn công nghệ mới nhất

### B. Transfer Learning từ pretrained weights tốt hơn

```python
# Thay vì train từ đầu, sử dụng weights đã train trên COCO
model = YOLO('yolov8m.pt')  # Pretrained trên COCO

# Hoặc tìm weights đã train trên helmet detection
# Roboflow Universe có nhiều pretrained models
```

---

## 3️⃣ TỐI ƯU HYPERPARAMETERS

### A. File config training tối ưu

```python
# File: train_improved.py
from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # Dùng medium thay vì nano

results = model.train(
    # Dataset
    data='data/LP-Helmet.v2i.yolov8/data.yaml',

    # Training settings
    epochs=300,              # ⬆️ Tăng từ 100 lên 300
    patience=50,             # Early stopping patience
    batch=16,                # Tùy GPU (16 hoặc 32)
    imgsz=640,               # Hoặc 1280 nếu GPU mạnh

    # Optimization
    optimizer='AdamW',       # ⭐ Tốt hơn SGD
    lr0=0.001,              # Learning rate ban đầu
    lrf=0.01,               # Learning rate cuối
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,

    # Data augmentation
    degrees=15.0,
    translate=0.2,
    scale=0.9,
    shear=5.0,
    perspective=0.0003,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.3,
    copy_paste=0.3,

    # Advanced
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    # Others
    device=0,                # GPU 0
    workers=8,               # CPU workers
    project='runs/detect',
    name='helmet_v8m_improved',
    exist_ok=True,
    pretrained=True,
    verbose=True,
    save=True,
    save_period=10,          # Save mỗi 10 epochs

    # Multi-scale training
    rect=False,              # Rectangular training
    cos_lr=True,             # Cosine learning rate
    close_mosaic=10,         # Tắt mosaic 10 epochs cuối

    # Loss weights
    box=7.5,                 # Box loss weight
    cls=0.5,                 # Class loss weight
    dfl=1.5,                 # DFL loss weight
)
```

### B. Thử AutoML (tự động tìm hyperparameters)

```python
# YOLO có sẵn hyperparameter tuning
model.tune(
    data='data/LP-Helmet.v2i.yolov8/data.yaml',
    epochs=30,
    iterations=300,          # Số lần thử
    optimizer='AdamW',
    plots=True,
    save=True,
    val=True
)
```

---

## 4️⃣ CẢI THIỆN QUÁ TRÌNH TRAINING

### A. Multi-stage training

```python
# Stage 1: Train với data đơn giản trước
# - Ảnh rõ ràng, góc chuẩn
# - 100 epochs

model = YOLO('yolov8m.pt')
model.train(data='data/easy_cases/data.yaml', epochs=100)

# Stage 2: Fine-tune với data khó hơn
# - Thêm góc độ khó, ánh sáng khó
# - 100 epochs

model = YOLO('runs/detect/stage1/weights/best.pt')
model.train(data='data/hard_cases/data.yaml', epochs=100, lr0=0.0001)

# Stage 3: Fine-tune với full dataset
# - Merge tất cả
# - 100 epochs

model = YOLO('runs/detect/stage2/weights/best.pt')
model.train(data='data/full/data.yaml', epochs=100, lr0=0.00001)
```

### B. Cross-validation

```python
# Train nhiều lần với splits khác nhau
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    # Tạo train/val split
    # Train model
    # Save best weights

    model = YOLO('yolov8m.pt')
    model.train(
        data=f'data/fold{fold}/data.yaml',
        epochs=200,
        name=f'fold{fold}'
    )

# Ensemble 5 models để dự đoán
```

---

## 5️⃣ POST-PROCESSING CẢI TIẾN

### A. Tăng độ chính xác với confidence threshold tuning

```python
# File: Source/ui_app_improved.py

# Thử nghiệm với các ngưỡng khác nhau
MOTO_CONF = 0.5          # ⬆️ Tăng từ 0.4 → 0.5
HELMET_LP_CONF = 0.6     # ⬆️ Tăng từ 0.4 → 0.6

# Hoặc động thái
def adaptive_confidence(image_quality):
    """Điều chỉnh confidence theo chất lượng ảnh"""
    if image_quality == 'good':
        return 0.5
    elif image_quality == 'medium':
        return 0.4
    else:
        return 0.3
```

### B. Non-Maximum Suppression (NMS) tuning

```python
# Trong predict
results = model.predict(
    image,
    conf=0.5,
    iou=0.5,      # ⬇️ Giảm để giữ nhiều boxes hơn
    max_det=100,   # Max detections
    agnostic_nms=True  # Class-agnostic NMS
)
```

### C. Multi-model Ensemble

```python
# Kết hợp nhiều models
def ensemble_predict(image, models, weights=None):
    """Ensemble prediction từ nhiều models"""
    all_predictions = []

    for model in models:
        pred = model.predict(image)
        all_predictions.append(pred)

    # Weighted voting hoặc averaging
    final_pred = weighted_boxes_fusion(all_predictions, weights)

    return final_pred

# Sử dụng
models = [
    YOLO('models/yolov8m_v1.pt'),
    YOLO('models/yolov8l_v2.pt'),
    YOLO('models/yolo11m_v3.pt'),
]

results = ensemble_predict(image, models, weights=[0.3, 0.4, 0.3])
```

### D. Tracking để giảm False Positives

```python
# Sử dụng tracking cho video
from ultralytics import YOLO

model = YOLO('models/HelmetLP.pt')

# Track thay vì predict
results = model.track(
    video_path,
    conf=0.5,
    iou=0.5,
    tracker='bytetrack.yaml',  # Hoặc 'botsort.yaml'
    persist=True
)

# Chỉ báo cáo vi phạm nếu:
# - Xuất hiện liên tục trong >= 5 frames
# - Confidence trung bình >= 0.6
```

---

## 6️⃣ CẢI THIỆN LOGIC PHÁT HIỆN

### A. Context-aware detection

```python
def smart_violation_detection(helmet_results, moto_crop):
    """
    Logic thông minh hơn để phát hiện vi phạm
    """

    # 1. Đếm số người trên xe (bằng detect heads/persons)
    num_persons = detect_persons_on_bike(moto_crop)

    # 2. Đếm số mũ bảo hiểm
    num_helmets = count_helmets(helmet_results)

    # 3. Logic:
    if num_persons > num_helmets:
        violation = True
        violation_count = num_persons - num_helmets
    else:
        violation = False
        violation_count = 0

    return {
        'violation': violation,
        'num_persons': num_persons,
        'num_helmets': num_helmets,
        'violation_count': violation_count
    }
```

### B. Temporal consistency (cho video)

```python
class ViolationTracker:
    """Track vi phạm qua nhiều frames"""

    def __init__(self, window_size=10, threshold=0.7):
        self.window_size = window_size
        self.threshold = threshold
        self.history = {}

    def update(self, license_plate, is_violation):
        """Cập nhật lịch sử"""
        if license_plate not in self.history:
            self.history[license_plate] = []

        self.history[license_plate].append(is_violation)

        # Giữ chỉ window_size frames gần nhất
        if len(self.history[license_plate]) > self.window_size:
            self.history[license_plate].pop(0)

    def is_violation(self, license_plate):
        """Kiểm tra có phải vi phạm không"""
        if license_plate not in self.history:
            return False

        # Nếu >70% frames là vi phạm → báo cáo
        violation_ratio = sum(self.history[license_plate]) / len(self.history[license_plate])

        return violation_ratio >= self.threshold

# Sử dụng
tracker = ViolationTracker(window_size=10, threshold=0.7)

for frame in video:
    # Detect
    results = detect_helmet(frame)

    for lp, violation in results:
        tracker.update(lp, violation)

        if tracker.is_violation(lp):
            # Báo cáo vi phạm
            report_violation(lp)
```

---

## 7️⃣ ĐÁNH GIÁ VÀ MONITORING

### A. Metrics chi tiết

```python
# Sau khi train, phân tích chi tiết
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# Validate
metrics = model.val(
    data='data/LP-Helmet.v2i.yolov8/data.yaml',
    split='test',
    save_json=True,
    save_hybrid=True,
    conf=0.001,  # Thấp để tính tất cả
    iou=0.6,
    max_det=300
)

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")

# Per-class metrics
for i, cls in enumerate(model.names.values()):
    print(f"{cls}: P={metrics.box.ap_class_index[i]:.3f}")
```

### B. Confusion Matrix Analysis

```python
# Vẽ confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Dự đoán trên test set
results = model.predict('data/test/images', save=False)

# Tính confusion matrix
y_true = []  # Ground truth
y_pred = []  # Predictions

for result in results:
    # Extract labels
    # ...
    pass

cm = confusion_matrix(y_true, y_pred)

# Vẽ
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Helmet Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')

# Phân tích:
# - Helmet bị nhầm thành No Helmet? → Cần thêm data mũ đặc biệt
# - No Helmet bị nhầm thành Helmet? → Cần thêm data không mũ
```

### C. Error Analysis

```python
# Tìm ảnh bị sai để phân tích
def analyze_errors(model, test_dir, output_dir='error_analysis'):
    """Tìm và lưu các ảnh dự đoán sai"""

    os.makedirs(output_dir, exist_ok=True)

    false_positives = []
    false_negatives = []

    for img_path in glob.glob(f'{test_dir}/*.jpg'):
        # Predict
        result = model.predict(img_path)[0]

        # Load ground truth
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        gt_labels = load_labels(label_path)

        # So sánh
        fp, fn = compare_predictions(result, gt_labels)

        if fp:
            false_positives.append(img_path)
            shutil.copy(img_path, f'{output_dir}/false_positives/')

        if fn:
            false_negatives.append(img_path)
            shutil.copy(img_path, f'{output_dir}/false_negatives/')

    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")

    return false_positives, false_negatives

# Chạy
fp, fn = analyze_errors(model, 'data/test/images')

# Xem các ảnh sai để hiểu pattern
# → Thêm data tương tự vào training set
```

---

## 8️⃣ CẢI THIỆN OCR BIỂN SỐ

### A. Sử dụng OCR tốt hơn

```python
# Thay vì EasyOCR, thử:

# 1. PaddleOCR (tốt hơn cho tiếng Việt)
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def read_license_plate(lp_crop):
    result = ocr.ocr(lp_crop, cls=True)
    text = ''.join([line[1][0] for line in result[0]])
    return text.replace(' ', '').upper()

# 2. TrOCR (Transformer-based OCR)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# 3. Train custom OCR model cho biển số VN
```

### B. Preprocessing tốt hơn cho biển số

```python
def preprocess_license_plate(lp_crop):
    """Tiền xử lý biển số trước khi OCR"""

    # 1. Resize
    h, w = lp_crop.shape[:2]
    if h < 50:
        scale = 50 / h
        lp_crop = cv2.resize(lp_crop, None, fx=scale, fy=scale)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray)

    # 4. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 5. Deskew (xoay thẳng)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # 6. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, kernel)

    return morph
```

---

## 9️⃣ LỘ TRÌNH CẢI THIỆN ƯU TIÊN

### 📅 Tuần 1-2: Thu thập và chuẩn bị data

- [ ] Thu thập 2000-5000 ảnh mới từ nhiều góc độ
- [ ] Label cẩn thận bằng Roboflow
- [ ] Augmentation để tăng lên 10000-15000 ảnh
- [ ] Balance classes

### 📅 Tuần 3-4: Training cải thiện

- [ ] Train YOLOv8m hoặc YOLOv8l
- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Early stopping theo validation loss

### 📅 Tuần 5: Evaluation và Error Analysis

- [ ] Đánh giá trên test set
- [ ] Phân tích confusion matrix
- [ ] Tìm pattern lỗi
- [ ] Thu thập thêm data cho cases lỗi

### 📅 Tuần 6: Fine-tuning và Deployment

- [ ] Fine-tune trên hard cases
- [ ] Ensemble models
- [ ] Optimize inference speed
- [ ] Deploy và test thực tế

---

## 🎯 KẾT QUẢ KỲ VỌNG

### Trước cải thiện:

```
mAP50: ~0.65
Precision: ~0.60
Recall: ~0.55
False Positive Rate: Cao
```

### Sau cải thiện (mục tiêu):

```
mAP50: >0.85
Precision: >0.80
Recall: >0.80
False Positive Rate: <10%
```

---

## 📚 TÀI LIỆU THAM KHẢO

### Papers:

1. **YOLOv8**: [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
2. **Data Augmentation**: "A survey on Image Data Augmentation for Deep Learning"
3. **Helmet Detection**: "Helmet Detection Using Deep Learning" (IEEE)

### Datasets:

1. [Roboflow Helmet Detection](https://universe.roboflow.com/search?q=helmet)
2. [Kaggle Safety Helmet](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
3. [SEFD Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)

### Tools:

1. **Roboflow**: Labeling, augmentation, dataset management
2. **CVAT**: Video annotation tool
3. **WandB**: Experiment tracking
4. **TensorBoard**: Training visualization

---

## 💡 MẸO BỔ SUNG

### 1. Sử dụng Active Learning

```python
# Chọn ảnh khó để label thêm
def select_hard_examples(model, unlabeled_images, n=100):
    """Chọn n ảnh mà model không tự tin nhất"""

    uncertainties = []

    for img in unlabeled_images:
        result = model.predict(img)
        # Tính uncertainty (thấp = không tự tin)
        uncertainty = 1 - max(result[0].boxes.conf)
        uncertainties.append((img, uncertainty))

    # Sort và lấy top n
    uncertainties.sort(key=lambda x: x[1], reverse=True)

    return [img for img, _ in uncertainties[:n]]

# Chạy
hard_images = select_hard_examples(model, unlabeled_pool, n=500)
# → Label 500 ảnh này và thêm vào training set
```

### 2. Sử dụng Synthetic Data

```python
# Tạo data giả bằng compositing
import albumentations as A

def create_synthetic_data(background, helmet_images, num_samples=1000):
    """Ghép người đội mũ vào ảnh background"""

    transform = A.Compose([
        A.RandomScale(scale_limit=0.3),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ])

    # Logic ghép ảnh
    # ...

    return synthetic_images

# Tạo 1000 ảnh synthetic
synthetic = create_synthetic_data(traffic_bg, helmet_crops, 1000)
```

### 3. Regular Retraining

```python
# Cứ mỗi 2 tuần, retrain với data mới
# Script tự động:

import schedule
import time

def retrain_job():
    print("Starting retraining...")

    # 1. Collect new data from production
    new_data = collect_production_data()

    # 2. Auto-label với model hiện tại
    auto_labels = current_model.predict(new_data)

    # 3. Human review những cases khó
    reviewed = human_review(auto_labels)

    # 4. Merge vào training set
    merge_dataset(reviewed)

    # 5. Retrain
    new_model = YOLO('current_best.pt')
    new_model.train(data='data/merged/data.yaml', epochs=50)

    # 6. Evaluate
    if new_model.val().box.map > current_model.val().box.map:
        # Deploy new model
        deploy(new_model)

# Schedule mỗi 2 tuần
schedule.every(2).weeks.do(retrain_job)

while True:
    schedule.run_pending()
    time.sleep(1)
```

---

## ✅ CHECKLIST HÀNH ĐỘNG

### Ngay lập tức:

- [ ] Download 2-3 helmet detection datasets từ Roboflow/Kaggle
- [ ] Merge với dataset hiện tại
- [ ] Retrain với YOLOv8m, epochs=200

### Ngắn hạn (1-2 tuần):

- [ ] Thu thập thêm 1000 ảnh từ nhiều góc độ
- [ ] Label cẩn thận
- [ ] Implement augmentation mạnh hơn
- [ ] Retrain và đánh giá

### Trung hạn (1 tháng):

- [ ] Implement ensemble models
- [ ] Active learning pipeline
- [ ] OCR improvement
- [ ] Tracking cho video

### Dài hạn (2-3 tháng):

- [ ] Continuous learning system
- [ ] A/B testing framework
- [ ] Production monitoring
- [ ] Auto-retraining pipeline

---

**🎯 Bắt đầu từ việc thu thập data đa dạng - đây là yếu tố quan trọng nhất!**

Good luck! 🚀
