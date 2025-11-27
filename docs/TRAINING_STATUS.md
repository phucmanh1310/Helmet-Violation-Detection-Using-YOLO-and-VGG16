# ✅ TRAINING ĐÃ BẮT ĐẦU!

## 🚀 Tình trạng hiện tại

### ✅ **Đã hoàn thành:**

1. ✅ Merge 2 datasets → `data/_merged_all/`
2. ✅ Tạo 3 dataset views:
   - `data/_stage1_motorcyclist/` (Model 1)
   - `data/_stage2_helmet_lp_fullscene/` (Model 2 Option A)
   - `data/_stage2_helmet_lp_crops/` (Model 2 Option B)
3. ✅ Fix PyTorch 2.6 `weights_only` error
4. ✅ Tạo 3 training scripts với patch tích hợp:
   - `scripts/train_model1_motorcyclist.py`
   - `scripts/train_model2_fullscene.py`
   - `scripts/train_model2_crops.py`
5. ✅ Fix data.yaml paths (absolute paths cho Windows)
6. ✅ **ĐANG CHẠY**: Train Model 1 (Motorcyclist Detection)

---

## 📊 Training Progress

### **Model 1: Motorcyclist Detection**

- **Status**: 🟢 ĐANG TRAIN
- **Dataset**: `data/_stage1_motorcyclist/`
- **Config**:
  - Model: YOLOv8n
  - Epochs: 100
  - Batch: 16
  - Image size: 640
  - Device: GPU 0 (RTX 3050 6GB)
- **Thời gian ước tính**: 2-3 giờ
- **Output**: `runs/detect/model1_motorcyclist/`

### **Model 2: Helmet/LP Detection**

- **Status**: ⏳ CHỜ Model 1 hoàn thành
- **Có 2 options**:
  - Option A: Full Scene
  - Option B: ROI Crops (**Khuyến nghị**)

---

## 📌 Theo dõi Training

### **Xem tiến độ real-time:**

```powershell
# Xem logs
Get-Content runs/detect/model1_motorcyclist/results.csv -Tail 20 -Wait

# Xem tensorboard (nếu có)
tensorboard --logdir runs/detect/model1_motorcyclist
```

### **Kiểm tra GPU usage:**

```powershell
nvidia-smi
```

### **Files quan trọng sẽ được tạo:**

```
runs/detect/model1_motorcyclist/
├── weights/
│   ├── best.pt          ← Model tốt nhất (copy sang models/)
│   └── last.pt          ← Model checkpoint cuối
├── results.csv          ← Metrics theo epoch
├── confusion_matrix.png ← Confusion matrix
├── F1_curve.png
├── P_curve.png
├── R_curve.png
├── PR_curve.png
└── train_batch*.jpg     ← Ảnh training visualization
```

---

## ⏭️ Sau khi Model 1 hoàn thành

### **1. Validate Model 1:**

```powershell
yolo detect val `
  model=runs/detect/model1_motorcyclist/weights/best.pt `
  data=data/_stage1_motorcyclist/data.yaml
```

### **2. Test thử:**

```powershell
yolo detect predict `
  model=runs/detect/model1_motorcyclist/weights/best.pt `
  source=img/test/ `
  conf=0.4 `
  save=True
```

### **3. Copy Model 1 vào thư mục models:**

```powershell
Copy-Item "runs/detect/model1_motorcyclist/weights/best.pt" "models/Motov10l.pt" -Force
```

### **4. Train Model 2 (chọn 1 trong 2):**

**Option A - Full Scene:**

```powershell
py -3.13 scripts/train_model2_fullscene.py
```

**Option B - ROI Crops (Khuyến nghị):**

```powershell
py -3.13 scripts/train_model2_crops.py
```

---

## 🎯 Kết quả mong đợi

### **Model 1:**

- mAP50: >0.85
- mAP50-95: >0.60
- Precision: >0.80
- Recall: >0.75

Nếu đạt được metrics này → Chuyển sang train Model 2

---

## 🚨 Nếu gặp lỗi

### **Out of Memory:**

- Giảm batch size trong script: `batch=8` hoặc `batch=4`

### **Training quá chậm:**

- Kiểm tra GPU đang được dùng: `nvidia-smi`
- Đảm bảo `device=0` trong script

### **Loss không giảm:**

- Đợi thêm 20-30 epochs
- Nếu vẫn không giảm → có vấn đề với data

### **Training bị dừng:**

- Chạy lại script, YOLO sẽ tự động resume từ checkpoint cuối

---

## 📞 Commands hữu ích

```powershell
# Xem processes Python
Get-Process python

# Kill training nếu cần
Stop-Process -Name python -Force

# Xem disk space
Get-PSDrive C

# Backup runs folder
Copy-Item -Recurse runs runs_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')
```

---

**Cập nhật**: November 6, 2025 - Training started successfully! 🚀
**Next**: Đợi 2-3 giờ cho Model 1, sau đó train Model 2.
