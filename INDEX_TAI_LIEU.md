# 📚 TÀI LIỆU HƯỚNG DẪN - HELMET VIOLATION DETECTION

> Tổng hợp tất cả tài liệu quan trọng trong dự án

---

## 📖 CÁC FILE HƯỚNG DẪN CHÍNH

### 🚀 **1. HUONG_DAN_TRAIN_2_MODELS.md** (MỚI NHẤT - ƯU TIÊN)

**Mục đích**: Hướng dẫn chi tiết train 2 models theo kiến trúc 2-stage

**Nội dung**:

- ✅ Cách merge 2 datasets
- ✅ Tạo views cho Model 1 (Motorcyclist) và Model 2 (Helmet/LP)
- ✅ Lệnh train đầy đủ với giải thích tham số
- ✅ So sánh 2 options: Full Scene vs ROI Crops
- ✅ Metrics đánh giá và tối ưu hóa
- ✅ Checklist hoàn thành

**Khi nào dùng**: Khi cần train lại models từ đầu với datasets mới

---

### 🎯 **2. HUONG_DAN_CHAY.md**

**Mục đích**: Hướng dẫn chạy ứng dụng sau khi đã có models

**Nội dung**:

- ✅ Chạy CLI app (`main_app.py`)
- ✅ Chạy Gradio UI (`ui_app.py`)
- ✅ Fix lỗi PyTorch 2.6 compatibility

**Khi nào dùng**: Khi muốn test/demo ứng dụng với models đã train

---

### 🖥️ **3. Huong_dan_UI.md** & **README_UI.md**

**Mục đích**: Hướng dẫn sử dụng giao diện web Gradio

**Nội dung**:

- ✅ Upload ảnh/video
- ✅ Xem kết quả detection
- ✅ Bảng vi phạm
- ✅ Launcher scripts (CHAY_UI.bat, quick_start_ui.py)

**Khi nào dùng**: Khi cần demo cho giáo viên hoặc người dùng cuối

---

### 📅 **4. KE_HOACH_CHI_TIET_NHOM.md**

**Mục đích**: Kế hoạch học tập chi tiết cho nhóm 4 người (4 tuần)

**Nội dung**:

- ✅ Phân công vai trò: Theory Lead, Dev Lead, Data Leads
- ✅ Lịch trình từng tuần, từng ngày
- ✅ Mục tiêu cụ thể cho mỗi thành viên
- ✅ Checklist theo dõi tiến độ

**Khi nào dùng**: Khi cần lên kế hoạch học/làm việc nhóm

---

### 📄 **5. README.md**

**Mục đích**: Tổng quan dự án

**Nội dung**:

- ✅ Giới thiệu dự án
- ✅ Yêu cầu hệ thống
- ✅ Cách cài đặt
- ✅ Cấu trúc thư mục
- ✅ Credits

**Khi nào dùng**: Đọc đầu tiên khi mới vào dự án

---

## 🗂️ CẤU TRÚC THƯ MỤC DỰ ÁN

```
Helmet-Violation-Detection-Using-YOLO-and-VGG16/
│
├── 📚 Tài liệu
│   ├── README.md                        # Tổng quan
│   ├── HUONG_DAN_TRAIN_2_MODELS.md     # Train models (MỚI NHẤT)
│   ├── HUONG_DAN_CHAY.md               # Chạy app
│   ├── Huong_dan_UI.md                 # Hướng dẫn UI
│   ├── README_UI.md                    # Hướng dẫn UI (backup)
│   ├── KE_HOACH_CHI_TIET_NHOM.md      # Kế hoạch nhóm
│   └── INDEX_TAI_LIEU.md              # File này
│
├── 🗄️ Dữ liệu
│   └── data/
│       ├── _merged_all/                 # Dataset merged (4 classes)
│       ├── _stage1_motorcyclist/        # Model 1 dataset (1 class)
│       ├── _stage2_helmet_lp_fullscene/ # Model 2 Option A
│       ├── _stage2_helmet_lp_crops/     # Model 2 Option B (Recommended)
│       ├── Helmet_detect.v2i.yolov8/    # Dataset gốc 1
│       └── helmet-detection-and-license-plate-recognition.v6i.yolov8/ # Dataset gốc 2
│
├── 🤖 Models
│   └── models/
│       ├── Motov10l.pt                 # Model 1: Motorcyclist Detection
│       ├── HelmetLP.pt                 # Model 2: Helmet/LP Detection
│       ├── yolo11n.pt                  # Pretrained backup
│       └── yolov8n.pt                  # Pretrained backup
│
├── 💻 Source Code
│   └── Source/
│       ├── main_app.py                 # CLI application
│       ├── ui_app.py                   # Gradio web UI
│       ├── _Motobike.py               # Model 1 logic
│       ├── _LP_Helmet.py              # Model 2 logic
│       └── _myFunc.py                 # Utilities
│
├── 🛠️ Scripts
│   └── scripts/
│       ├── merge_and_prepare_datasets.py      # Merge datasets + create views
│       ├── make_roi_crops_from_class.py       # Create ROI crops
│       └── filter_labels_by_classes.py        # Filter labels
│
├── 📊 Training Results
│   └── runs/
│       └── detect/
│           ├── model1_motorcyclist/    # Model 1 training results
│           ├── model2_fullscene/       # Model 2 Option A results
│           └── model2_crops/           # Model 2 Option B results
│
└── 🚀 Launchers
    ├── CHAY_UI.bat                     # Windows batch launcher
    └── quick_start_ui.py               # Python launcher
```

---

## 🎯 WORKFLOW THÔNG THƯỜNG

### **Kịch bản 1: Bắt đầu dự án mới**

1. Đọc `README.md` - Hiểu tổng quan
2. Đọc `KE_HOACH_CHI_TIET_NHOM.md` - Lên kế hoạch nhóm
3. Chạy theo `HUONG_DAN_TRAIN_2_MODELS.md` - Train models
4. Test theo `HUONG_DAN_CHAY.md` - Chạy app

### **Kịch bản 2: Đã có models, cần demo**

1. Đọc `HUONG_DAN_CHAY.md` - Setup environment
2. Đọc `Huong_dan_UI.md` - Chạy Gradio UI
3. Demo cho giáo viên

### **Kịch bản 3: Cải thiện models**

1. Xem `HUONG_DAN_TRAIN_2_MODELS.md` → Phần "Tối ưu hóa"
2. Thu thập thêm data
3. Train lại với hyperparameters tốt hơn
4. So sánh metrics

### **Kịch bản 4: Báo cáo/Thuyết trình**

1. Đọc `KE_HOACH_CHI_TIET_NHOM.md` → Phần deliverables
2. Chuẩn bị slides từ các docs
3. Demo UI từ `Huong_dan_UI.md`

---

## ⚠️ LƯU Ý QUAN TRỌNG

### **Files đã XÓA (không dùng nữa):**

- ❌ `HUONG_DAN_TRAIN.md` - Đã thay bằng `HUONG_DAN_TRAIN_2_MODELS.md`
- ❌ `HUONG_DAN_CAI_THIEN_MODEL.md` - Quá dài, không còn phù hợp

### **Thứ tự ưu tiên đọc:**

1. 🥇 `README.md` - Bắt buộc đọc đầu tiên
2. 🥈 `HUONG_DAN_TRAIN_2_MODELS.md` - Nếu cần train
3. 🥉 `HUONG_DAN_CHAY.md` - Nếu cần chạy app
4. 📚 Còn lại - Tùy nhu cầu

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:

1. Xem lại file hướng dẫn tương ứng
2. Check logs trong `runs/detect/`
3. Kiểm tra compatibility Python 3.13 + PyTorch 2.6

---

**Cập nhật lần cuối**: November 6, 2025  
**Tình trạng**: ✅ All documents up-to-date với kiến trúc 2-model mới
