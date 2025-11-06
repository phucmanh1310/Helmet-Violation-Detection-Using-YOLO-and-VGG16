# 🚀 HƯỚNG DẪN TRAIN 2 MODELS - KIẾN TRÚC 2-STAGE

> **Đã chuẩn bị xong datasets!** Bây giờ train 2 models riêng biệt.

---

## 📊 TỔNG QUAN DATASETS ĐÃ TẠO

### ✅ Datasets có sẵn:

```
data/
├── _merged_all/                      # Dataset gốc merged (4 classes)
│   ├── train/  (17,402 images)
│   ├── valid/  (2,925 images)
│   └── test/   (1,676 images)
│
├── _stage1_motorcyclist/             # 🎯 MODEL 1 - Motorcyclist Detection
│   ├── train/  (11,996 images)       # Chỉ ảnh có motorcyclist
│   ├── valid/  (1,955 images)
│   ├── test/   (1,252 images)
│   └── data.yaml  (1 class: motorcyclist)
│
├── _stage2_helmet_lp_fullscene/      # 🎯 MODEL 2 Option A - Full Scene
│   ├── train/  (17,340 images)       # Ảnh gốc với 3 classes
│   ├── valid/  (2,917 images)
│   ├── test/   (1,675 images)
│   └── data.yaml  (3 classes: helmet, nohelmet, licenseplate)
│
└── _stage2_helmet_lp_crops/          # 🎯 MODEL 2 Option B - ROI Crops (KHUYẾN NGHỊ)
    ├── train/  (crop images)         # Crops từ motorcyclist ROI
    ├── valid/  (crop images)
    ├── test/   (crop images)
    └── data.yaml  (3 classes: helmet, nohelmet, licenseplate)
```

---

## 🎯 KIẾN TRÚC 2-STAGE

### **MODEL 1: Motorcyclist Detection**

- **Input**: Ảnh gốc (full scene)
- **Output**: Bounding boxes của motorcyclist (người + xe)
- **Dataset**: `data/_stage1_motorcyclist/`
- **Classes**: 1 (motorcyclist)

### **MODEL 2: Helmet/NoHelmet/LicensePlate Detection**

- **Input**:
  - Option A: Ảnh gốc (full scene)
  - Option B: ROI crops từ Model 1 (**Khuyến nghị**)
- **Output**: Bounding boxes của helmet, nohelmet, licenseplate
- **Dataset**:
  - Option A: `data/_stage2_helmet_lp_fullscene/`
  - Option B: `data/_stage2_helmet_lp_crops/` (**Khuyến nghị**)
- **Classes**: 3 (helmet, nohelmet, licenseplate)

### **Pipeline Inference:**

```
Input Image
    ↓
[Model 1] → Detect motorcyclist boxes
    ↓
Crop ROI từ motorcyclist boxes
    ↓
[Model 2] → Detect helmet/nohelmet/licenseplate trong ROI
    ↓
Output: Violations detected
```

---

## 🏋️ BƯỚC 1: TRAIN MODEL 1 - MOTORCYCLIST DETECTION

### **Lệnh Train (PowerShell):**

```powershell
# Vào thư mục project
cd "d:\hoctap\Năm 4\HK1\Python_CoTrang\project\Helmet-Violation-Detection-Using-YOLO-and-VGG16"

# Train Model 1 (với PyTorch 2.6 fix)
py -3.13 scripts/train_model1_motorcyclist.py
```

**Lưu ý**: Script đã tích hợp fix cho PyTorch 2.6 `weights_only` error.

### **Giải thích tham số:**

- `model=yolov8n.pt`: Pretrained YOLOv8 nano (nhẹ, nhanh)
- `epochs=100`: Train 100 epochs
- `imgsz=640`: Resize ảnh về 640x640
- `batch=16`: Batch size (điều chỉnh theo GPU của bạn)
- `optimizer=AdamW`: Optimizer tốt cho YOLO
- `lr0=0.01`: Learning rate ban đầu
- `patience=50`: Early stopping nếu không cải thiện sau 50 epochs
- `device=0`: Dùng GPU 0

### **Thời gian ước tính:**

- YOLOv8n: ~2-3 giờ (GPU RTX 3060)
- YOLOv8m: ~4-6 giờ

### **Kết quả:**

- Model tốt nhất: `runs/detect/model1_motorcyclist/weights/best.pt`
- Metrics: `runs/detect/model1_motorcyclist/results.csv`
- Confusion matrix: `runs/detect/model1_motorcyclist/confusion_matrix.png`

### **Đánh giá Model 1:**

```powershell
# Validate
yolo detect val `
  model=runs/detect/model1_motorcyclist/weights/best.pt `
  data=data/_stage1_motorcyclist/data.yaml

# Test trên 1 ảnh
yolo detect predict `
  model=runs/detect/model1_motorcyclist/weights/best.pt `
  source=img/test/test1.jpg `
  conf=0.4
```

---

## 🏋️ BƯỚC 2: TRAIN MODEL 2 - HELMET/NOHELMET/LP DETECTION

### **Option A: Full Scene (Đơn giản, baseline)**

```powershell
# Train Model 2 Option A (với PyTorch 2.6 fix)
py -3.13 scripts/train_model2_fullscene.py
```

### **Option B: ROI Crops (KHUYẾN NGHỊ - Chính xác hơn)**

```powershell
# Train Model 2 Option B (với PyTorch 2.6 fix)
py -3.13 scripts/train_model2_crops.py
```

**Tại sao imgsz=768?**

- Helmet và license plate là objects nhỏ
- Độ phân giải cao hơn giúp detect tốt hơn
- Trade-off: chậm hơn một chút

### **Thời gian ước tính:**

- Full Scene: ~3-4 giờ
- Crops: ~4-6 giờ (có nhiều crops hơn)

### **Kết quả:**

- Model Full Scene: `runs/detect/model2_fullscene/weights/best.pt`
- Model Crops: `runs/detect/model2_crops/weights/best.pt`

### **Đánh giá Model 2:**

```powershell
# Validate Full Scene
yolo detect val `
  model=runs/detect/model2_fullscene/weights/best.pt `
  data=data/_stage2_helmet_lp_fullscene/data.yaml

# Validate Crops
yolo detect val `
  model=runs/detect/model2_crops/weights/best.pt `
  data=data/_stage2_helmet_lp_crops/data.yaml
```

---

## 📊 BƯỚC 3: SO SÁNH VÀ CHỌN MODEL TỐT NHẤT

### **Metrics quan trọng:**

1. **mAP50** (mean Average Precision @ IoU=0.5)

   - Độ chính xác tổng thể
   - Target: >0.7 là tốt, >0.8 là rất tốt

2. **mAP50-95** (mAP @ IoU=0.5-0.95)

   - Độ chính xác ở nhiều IoU threshold
   - Target: >0.5 là tốt

3. **Precision** (Độ chính xác)

   - Trong số dự đoán positive, bao nhiêu % đúng?
   - Target: >0.75

4. **Recall** (Độ phủ)
   - Trong số ground truth, detect được bao nhiêu %?
   - Target: >0.7

### **So sánh Model 2:**

| Metric               | Full Scene | Crops | Winner |
| -------------------- | ---------- | ----- | ------ |
| mAP50                | ???        | ???   | ?      |
| mAP50-95             | ???        | ???   | ?      |
| Precision (helmet)   | ???        | ???   | ?      |
| Recall (helmet)      | ???        | ???   | ?      |
| Precision (nohelmet) | ???        | ???   | ?      |
| Recall (nohelmet)    | ???        | ???   | ?      |
| Precision (LP)       | ???        | ???   | ?      |
| Recall (LP)          | ???        | ???   | ?      |
| FPS (inference)      | ???        | ???   | ?      |

**Ghi chú**: Điền kết quả sau khi train xong.

### **Quyết định:**

- Nếu Crops tốt hơn đáng kể (mAP50 > +5%) → Dùng Crops
- Nếu Full Scene tốt tương đương nhưng nhanh hơn → Dùng Full Scene
- **Khuyến nghị**: Thường Crops tốt hơn 10-15% mAP cho helmet/nohelmet

---

## 🎨 BƯỚC 4: EXPORT MODELS VÀO THƯU MỤC `models/`

```powershell
# Copy Model 1
Copy-Item "runs/detect/model1_motorcyclist/weights/best.pt" "models/Motov10l.pt" -Force

# Copy Model 2 (chọn 1 trong 2)
# Option A: Full Scene
Copy-Item "runs/detect/model2_fullscene/weights/best.pt" "models/HelmetLP.pt" -Force

# Option B: Crops (Khuyến nghị)
Copy-Item "runs/detect/model2_crops/weights/best.pt" "models/HelmetLP.pt" -Force
```

---

## 🧪 BƯỚC 5: TEST TOÀN BỘ PIPELINE

### **Test Script (Python):**

```python
from ultralytics import YOLO
import cv2

# Load models
model1 = YOLO('models/Motov10l.pt')  # Motorcyclist
model2 = YOLO('models/HelmetLP.pt')   # Helmet/NoHelmet/LP

# Test image
img_path = 'img/test/test1.jpg'
img = cv2.imread(img_path)

# Stage 1: Detect motorcyclists
results1 = model1.predict(img, conf=0.4, verbose=False)

# Stage 2: For each motorcyclist, crop and detect helmet/LP
for box in results1[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Padding
    pad = 0.1
    w, h = x2-x1, y2-y1
    x1 = max(0, x1 - int(w*pad))
    y1 = max(0, y1 - int(h*pad))
    x2 = min(img.shape[1], x2 + int(w*pad))
    y2 = min(img.shape[0], y2 + int(h*pad))

    # Crop
    crop = img[y1:y2, x1:x2]

    # Detect in crop
    results2 = model2.predict(crop, conf=0.3, verbose=False)

    # Check violation
    has_helmet = False
    has_nohelmet = False

    for box2 in results2[0].boxes:
        cls = int(box2.cls[0])
        if cls == 0:  # helmet
            has_helmet = True
        elif cls == 1:  # nohelmet
            has_nohelmet = True

    if has_nohelmet or (not has_helmet and not has_nohelmet):
        print("🚨 VI PHẠM: Không đội mũ bảo hiểm!")
    else:
        print("✅ OK: Có đội mũ bảo hiểm")

print("Test hoàn tất!")
```

### **Chạy test:**

```powershell
py -3.13 test_pipeline.py
```

---

## 📈 BƯỚC 6: TỐI ƯU HÓA (Nếu kết quả chưa tốt)

### **Nếu Model 1 (Motorcyclist) kém:**

1. **Thu thập thêm data** - đặc biệt góc khó, xa, che khuất
2. **Tăng epochs** → 150-200
3. **Thử model lớn hơn** - yolov8m.pt hoặc yolov8l.pt
4. **Augmentation mạnh hơn**:
   ```
   degrees=15 translate=0.2 scale=0.7 hsv_h=0.02
   ```

### **Nếu Model 2 (Helmet/LP) kém:**

1. **Dùng Crops thay vì Full Scene** (tăng 10-15% mAP)
2. **Tăng imgsz** → 832 hoặc 1024 (cho objects nhỏ)
3. **Thu thập thêm data** - đặc biệt:
   - Góc nghiêng
   - Ánh sáng xấu (tối, ngược sáng)
   - LP bị mờ, che khuất
4. **Class weights** nếu imbalance:
   ```python
   # Trong YAML, thêm:
   class_weights: [1.0, 1.5, 1.2]  # helmet, nohelmet, LP
   ```
5. **Ensemble models** - train nhiều models, vote kết quả

### **Hyperparameter tuning:**

```powershell
# Auto-tune hyperparameters (chạy 100 epochs thử)
yolo detect tune `
  model=yolov8n.pt `
  data=data/_stage2_helmet_lp_crops/data.yaml `
  epochs=100 `
  iterations=30
```

---

## 🎯 KẾT QUẢ MONG ĐỢI

### **Model 1 - Motorcyclist:**

- mAP50: >0.85 (tốt)
- mAP50-95: >0.60
- FPS: >30 (real-time)

### **Model 2 - Helmet/NoHelmet/LP:**

- **Full Scene:**

  - mAP50: 0.65-0.75
  - mAP50-95: 0.40-0.50
  - FPS: >25

- **Crops (Khuyến nghị):**
  - mAP50: 0.75-0.85 ⬆️ (+10-15%)
  - mAP50-95: 0.50-0.60 ⬆️
  - FPS: >20 (chậm hơn vì 2 stages)

### **Overall Pipeline:**

- Accuracy: >80% violations detected
- False Positives: <10%
- FPS: 15-25 (real-time trên video)

---

## 📝 CHECKLIST HOÀN THÀNH

### **Setup:**

- [x] Merge 2 datasets → `_merged_all`
- [x] Tạo Stage 1 view → `_stage1_motorcyclist`
- [x] Tạo Stage 2 Full Scene view → `_stage2_helmet_lp_fullscene`
- [x] Tạo Stage 2 Crops view → `_stage2_helmet_lp_crops`

### **Training:**

- [ ] Train Model 1 (Motorcyclist) - 100 epochs
- [ ] Validate Model 1
- [ ] Train Model 2 Option A (Full Scene) - 150 epochs
- [ ] Train Model 2 Option B (Crops) - 150 epochs
- [ ] Validate cả 2 versions Model 2
- [ ] So sánh kết quả, chọn tốt nhất

### **Integration:**

- [ ] Export models vào `models/`
- [ ] Update `Source/main_app.py` với models mới
- [ ] Update `Source/ui_app.py` với models mới
- [ ] Test toàn bộ pipeline
- [ ] Test trên video thực tế

### **Documentation:**

- [ ] Ghi nhận metrics (mAP, precision, recall)
- [ ] Tạo comparison table
- [ ] Chụp screenshots kết quả
- [ ] Viết báo cáo kết quả training

---

## 🚨 LƯU Ý QUAN TRỌNG

### **GPU Memory:**

- YOLOv8n batch=16 imgsz=640: ~4GB VRAM
- YOLOv8n batch=16 imgsz=768: ~6GB VRAM
- Nếu out of memory, giảm batch size:
  ```
  batch=8  # hoặc batch=4
  ```

### **Training Time:**

- Không tắt máy giữa chừng
- Dùng `patience=50` để auto early stop
- Theo dõi loss giảm đều
- Nếu loss không giảm sau 20 epochs → có vấn đề

### **Data Quality:**

- Kiểm tra labels trước khi train:
  ```powershell
  # Visualize labels
  yolo detect val data=data/_stage1_motorcyclist/data.yaml split=train max_det=10
  ```
- Xóa ảnh/labels lỗi

### **Backup:**

- Backup models định kỳ
- Lưu `runs/detect/` folder
- Git commit thường xuyên

---

## 🎓 HỌC THÊM

### **YOLO Documentation:**

- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/modes/val/
- https://docs.ultralytics.com/modes/predict/

### **Hyperparameter Tuning:**

- https://docs.ultralytics.com/guides/hyperparameter-tuning/

### **Model Export:**

- https://docs.ultralytics.com/modes/export/

---

## ✅ BẮTT ĐẦU NGAY!

```powershell
# 1. Train Model 1 (với PyTorch 2.6 fix)
py -3.13 scripts/train_model1_motorcyclist.py

# 2. Train Model 2 - Option B: Crops (Recommended, với PyTorch 2.6 fix)
py -3.13 scripts/train_model2_crops.py

# 3. Copy models
Copy-Item "runs/detect/model1_motorcyclist/weights/best.pt" "models/Motov10l.pt" -Force
Copy-Item "runs/detect/model2_crops/weights/best.pt" "models/HelmetLP.pt" -Force
```

**Chúc bạn train thành công! 🚀**
