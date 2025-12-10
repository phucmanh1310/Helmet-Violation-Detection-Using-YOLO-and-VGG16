# 🛵 Helmet Violation Detection Using YOLO - 2-Stage Detection System

> **Hệ thống phát hiện vi phạm mũ bảo hiểm và nhận diện biển số xe sử dụng kiến trúc 2-Stage YOLO**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.196-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#️-kiến-trúc-hệ-thống)
- [Tính năng](#-tính-năng)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Training Models](#️-training-models)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Tài liệu](#-tài-liệu)
- [Kết quả](#-kết-quả)
- [Đội ngũ](#-đội-ngũ-phát-triển)

---

## 🎯 Giới thiệu

Dự án **Helmet Violation Detection** sử dụng **kiến trúc 2-Stage Detection** với YOLOv8 để phát hiện vi phạm không đội mũ bảo hiểm khi tham gia giao thông. Hệ thống có khả năng:

- ✅ Phát hiện người đi xe máy (motorcyclist) trong ảnh/video giao thông
- ✅ Phát hiện vi phạm không đội mũ bảo hiểm với độ chính xác cao
- ✅ Nhận diện và đọc biển số xe tự động (License Plate OCR)
- ✅ Thống kê, báo cáo vi phạm chi tiết
- ✅ Giao diện web thân thiện (Gradio UI)

### 🔥 Điểm nổi bật

- **2-Stage Detection**: Tách riêng detection xe và detection vi phạm → Tăng accuracy
- **ROI-based Processing**: Chỉ xử lý vùng quan tâm → Giảm false positives
- **High Performance**: mAP > 0.80, Real-time 15-25 FPS
- **Easy Deployment**: Web UI + CLI + Python API
- **Production Ready**: PyTorch 2.6 compatible, Windows/Linux support

---

## 🏗️ Kiến trúc hệ thống

### Pipeline tổng quan

```
📸 Input Image/Video
       ↓
┌──────────────────────────────────────┐
│  STAGE 1: Motorcyclist Detection    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Model: Motov10l.pt (YOLOv8)         │
│  Classes: motorcyclist (1 class)     │
│  Input: Full scene 640x640           │
│  Output: Bounding boxes của xe máy  │
└──────────────────────────────────────┘
       ↓
  🔲 Crop ROI từ motorcyclist boxes
       ↓
┌──────────────────────────────────────┐
│  STAGE 2: Helmet/LP Detection       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Model: HelmetLP.pt (YOLOv8)         │
│  Classes: helmet, nohelmet, LP       │
│  Input: ROI crops 768x768            │
│  Output: Bounding boxes chi tiết    │
└──────────────────────────────────────┘
       ↓
  📝 OCR License Plate (EasyOCR)
       ↓
┌──────────────────────────────────────┐
│  Violation Analysis & Reporting     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  - Phân tích vi phạm                 │
│  - Extract biển số                   │
│  - Tạo báo cáo                       │
└──────────────────────────────────────┘
       ↓
  📊 Output: Annotated Media + Report Table
```

### Lý do sử dụng 2-Stage

| Aspect              | 1-Stage (Direct) | 2-Stage (Ours)             |
| ------------------- | ---------------- | -------------------------- |
| **Accuracy**        | 70-75%           | **85-90%** ✅              |
| **False Positives** | Cao (~15%)       | Thấp (<10%) ✅             |
| **Small Object**    | Khó detect       | Tốt hơn (higher res) ✅    |
| **Speed**           | Nhanh hơn        | Chấp nhận được (15-25 FPS) |

---

## ✨ Tính năng

### 🌐 Web Interface (Gradio)

- Upload ảnh/video qua browser
- Real-time detection với progress bar
- Visualize kết quả với màu sắc:
  - 🔴 **Đỏ**: Vi phạm (không đội mũ)
  - 🟢 **Xanh**: An toàn (đội mũ)
  - 🔵 **Xanh dương**: Biển số xe

### 📊 Báo cáo chi tiết

| Thông tin      | Mô tả                   |
| -------------- | ----------------------- |
| **STT**        | Số thứ tự vi phạm       |
| **Biển số**    | OCR tự động từ ảnh      |
| **Thời gian**  | Timestamp phát hiện     |
| **ID vi phạm** | Unique identifier       |
| **Tùy chỉnh**  | Họ tên, Email (mở rộng) |

### 🎥 Xử lý Video

- Hỗ trợ: MP4, AVI, MOV, MKV
- Frame skipping (configurable)
- Export video annotated
- CSV/JSON export

---

## 🚀 Cài đặt

### Yêu cầu

```yaml
Python: >= 3.8 (khuyến nghị 3.13)
CUDA: >= 11.8 (GPU) hoặc CPU
GPU: Nên sử dụng GPU tăng hiệu suất huấn luyện
RAM: >= 8GB
Storage: >= 10GB
```

### Bước 1: Clone repo

```bash
git clone https://github.com/phucmanh1310/Helmet-Violation-Detection-Using-YOLO-and-VGG16.git
cd Helmet-Violation-Detection-Using-YOLO-and-VGG16
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Setup Dataset

**Option A: Download từ Google Drive (Nhanh nhất)**

Dataset (~6GB) được lưu tại: [Google Drive](https://drive.google.com/drive/folders/1gcd40p0yV5krJvlOQVOs7RVOUhq54NBT)

```bash
# 1. Download folder "DataSet" từ Drive trên
# 2. Extract vào thư mục project
# 3. Rename thành "data" nếu cần
# Structure sau khi extract:
#   data/
#   ├── _merged_all/          (merged dataset từ 3 sources)
#   ├── _stage1_motorcyclist/ (dataset for Model 1)
#   ├── _stage2_helmet_lp_crops/ (dataset for Model 2 - crops)
#   └── _stage2_helmet_lp_fullscene/ (dataset for Model 2 - full scene)
```

**Option B: Auto-generate từ Roboflow (Nếu có account)**

```bash
# Require: Roboflow account + API key
# Chỉnh config/roboflow_config.json
py -3.13 scripts/merge_and_prepare_datasets.py
py -3.13 scripts/filter_labels_by_classes.py ...
py -3.13 scripts/make_roi_crops_from_class.py ...
```

**Option C: Skip dataset (Inference only)**

Nếu chỉ muốn test inference:

```bash
# Dùng ảnh test trong img/test/
# Dataset không bắt buộc
```

### Bước 4: Verify models

Kiểm tra models đã có trong `models/`:

- ✅ `Motov10l.pt` - Model 1 (Motorcyclist Detection)
- ✅ `HelmetLP.pt` - Model 2 (Helmet/LP Detection)

Nếu chưa có, train theo [HUONG_DAN_TRAIN_2_MODELS.md](HUONG_DAN_TRAIN_2_MODELS.md)

---

## 💻 Sử dụng

### 🌐 Web UI (Khuyến nghị)

```bash
python quick_start_ui.py
```

Mở browser: **http://127.0.0.1:7860**

### 🖥️ Command Line

```bash
cd Source
python main_app.py --image path/to/image.jpg
python main_app.py --video path/to/video.mp4
```

### 🐍 Python API

```python
from Source._Motobike import detect_motorcyclists
from Source._LP_Helmet import detect_helmet_and_lp
import cv2

img = cv2.imread('test.jpg')

# Stage 1
moto_boxes = detect_motorcyclists(img, conf=0.4)

# Stage 2
for box in moto_boxes:
    roi = img[box[1]:box[3], box[0]:box[2]]
    results = detect_helmet_and_lp(roi, conf=0.3)

    # Analyze
    has_violation = any(r['class'] == 'nohelmet' for r in results)
    if has_violation:
        print("🚨 Vi phạm phát hiện!")
```

### Đọc thêm hướng dẫn khởi động

Khởi động không cần UI: **[HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md)**
Khởi động sử dụng UI: **[Huong_dan_UI.md](Huong_dan_UI.md)**

---

## 🏋️ Training Models

Chi tiết xem: **[HUONG_DAN_TRAIN_2_MODELS.md](HUONG_DAN_TRAIN_2_MODELS.md)**

### Quick Training

```bash
# Model 1: Motorcyclist (100 epochs, ~2-3 giờ)
py -3.13 scripts/train_model1_motorcyclist.py

# Model 2: Helmet/LP ROI Crops (150 epochs, ~4-6 giờ)
py -3.13 scripts/train_model2_crops.py
```

### Resume Training

```bash
py -3.13 scripts/resume_model1_training.py
```

### Datasets

| Dataset                        | Images  | Classes | Purpose                        |
| ------------------------------ | ------- | ------- | ------------------------------ |
| `_stage1_motorcyclist/`        | 11,996  | 1       | Model 1 training               |
| `_stage2_helmet_lp_crops/`     | ~25,000 | 3       | Model 2 training (khuyến nghị) |
| `_stage2_helmet_lp_fullscene/` | 17,340  | 3       | Model 2 baseline               |

---

## 📁 Cấu trúc thư mục

```
📦 Helmet-Violation-Detection/
│
├── 📂 Source/                  # Source code
│   ├── ui_app.py              # Gradio Web UI
│   ├── main_app.py            # CLI app
│   ├── _Motobike.py           # Model 1 module
│   ├── _LP_Helmet.py          # Model 2 module
│   └── _myFunc.py             # Utilities
│
├── 📂 models/                  # Trained models
│   ├── Motov10l.pt            # Model 1 ⭐
│   └── HelmetLP.pt            # Model 2 ⭐
│
├── 📂 data/                    # Datasets
│   ├── _stage1_motorcyclist/
│   ├── _stage2_helmet_lp_crops/
│   └── _stage2_helmet_lp_fullscene/
│
├── 📂 scripts/                 # Training scripts
│   ├── train_model1_motorcyclist.py
│   ├── train_model2_crops.py
│   ├── resume_model1_training.py
│   └── merge_and_prepare_datasets.py
│
├── 📂 runs/                    # Training outputs
├── 📂 img/                     # Results
│
├── 📂 docs/                  #  documents cua du an
|   ├── 📄 LY_THUYET_VA_GIAI_THICH_CODE.md  # ⭐ Lý thuyết gồm 3 phần
|   ├── 📄 INDEX_TAI_LIEU.md
|   ├── 📄 KE_HOACH_CHI_TIET_NHOM.md
|   ├── 📄 quick_start_ui.py
|
└── 📄 README.md
└── 📄 README_UI.md
```

---

## 📚 Tài liệu

| File                                                                                    | Nội dung                       |
| --------------------------------------------------------------------------------------- | ------------------------------ |
| **[README.md](README.md)**                                                              | Tổng quan dự án (file này)     |
| **[HUONG_DAN_TRAIN_2_MODELS.md](HUONG_DAN_TRAIN_2_MODELS.md)**                          | Hướng dẫn training chi tiết    |
| **[LY_THUYET_VA_GIAI_THICH_CODE_Phan1.md](docs\LY_THUYET_VA_GIAI_THICH_CODE_PHAN1.md)** | 📖 Lý thuyết & giải thích code |
| **[README_UI.md](README_UI.md)**                                                        | Hướng dẫn Web UI               |
| **[INDEX_TAI_LIEU.md](INDEX_TAI_LIEU.md)**                                              | Index tài liệu                 |

---

## 📊 Kết quả

### Model 1: Motorcyclist Detection

| Metric    | Value    |
| --------- | -------- |
| mAP50     | **0.87** |
| mAP50-95  | 0.61     |
| Precision | 0.84     |
| Recall    | 0.83     |

### Model 2: Helmet/LP (ROI Crops)

| Class        | Precision | Recall   | mAP50    |
| ------------ | --------- | -------- | -------- |
| Helmet       | 0.82      | 0.79     | 0.81     |
| NoHelmet     | 0.78      | 0.75     | 0.77     |
| LicensePlate | 0.85      | 0.80     | 0.83     |
| **Average**  | **0.82**  | **0.78** | **0.80** |

### Performance

- **Speed**: 15-25 FPS (RTX 3050 6GB)
- **Accuracy**: ~85% overall
- **False Positive Rate**: <10%

---

## 👥 Đội ngũ phát triển

### Original Authors

1. **NGUYEN DINH THANH SAN**

   - Major: Artificial Intelligence
   - GitHub: [@ThanhSan97](https://github.com/ThanhSan97)
   - Email: samnguyen0907@gmail.com

2. **NGUYEN HUYNH CHI KHANG**

   - Major: Artificial Intelligence
   - GitHub: [@Khang1405](https://github.com/Khang1405)
   - Email: chikhang1235202@gmail.com

3. **NGUYEN PHAN DUC THANH**
   - Major: Artificial Intelligence
   - GitHub: [@NguyenPhanDucThanh](https://github.com/NguyenPhanDucThanh)
   - Email: thanhnguyen1802dn@gmail.com

### Current Maintainer

- **GitHub**: [@phucmanh1310](https://github.com/phucmanh1310)
- **Email**: phucmanhtran08@gmail.com

- **GitHub**: [@]()
- **Email**:

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **Ultralytics Team** - YOLOv8
- **Gradio Team** - Web UI framework
- **JaidedAI** - EasyOCR
- **Roboflow** - Dataset management

---

## 🔗 Links

- **Original Project**: [ThanhSan97/Helmet-Violation-Detection](https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16)
- **Datasets**:
  - Traffic: https://universe.roboflow.com/cdio-zmfmj/motobike-detection
  - Helmet/LP: https://universe.roboflow.com/cdio-zmfmj/helmet-lincense-plate-detection-gevlq

---

<div align="center">

**⭐ Nếu project hữu ích, đừng quên cho star! ⭐**

Made with ❤️ by Computer Vision Team

[⬆ Back to Top](#-helmet-violation-detection-using-yolo---2-stage-detection-system)

</div>
