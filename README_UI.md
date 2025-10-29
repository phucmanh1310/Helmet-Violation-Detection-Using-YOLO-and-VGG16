# 🛵 Hệ Thống Phát Hiện Vi Phạm Mũ Bảo Hiểm - GIAO DIỆN WEB UI

## 🎯 Giới thiệu

Đây là phiên bản cải tiến của dự án **Helmet Violation Detection** với giao diện web hiện đại được xây dựng bằng **Gradio**. Hệ thống cho phép upload ảnh/video và hiển thị kết quả phát hiện vi phạm một cách trực quan với bảng thống kê chi tiết.

## ✨ Tính năng mới

### 🌐 Giao diện Web (UI)

- **Upload ảnh/video** dễ dàng qua trình duyệt web
- **Hiển thị kết quả** trực quan với bounding boxes màu sắc:
  - 🔴 **Màu đỏ**: Vi phạm (không đội mũ bảo hiểm)
  - 🟢 **Màu xanh**: Tuân thủ (đội mũ bảo hiểm)
- **Bảng thống kê chi tiết** gồm:
  - STT
  - Họ và tên (có thể mở rộng)
  - Gmail (có thể mở rộng)
  - Biển số xe (tự động OCR)
  - Thời gian phát hiện
  - ID vi phạm
- **Xử lý video** với progress bar hiển thị tiến độ
- **Responsive design** - Truy cập từ mọi thiết bị

### 🔥 Cải tiến

- Không cần cài đặt PyQt5
- Giao diện thân thiện, dễ sử dụng
- Hỗ trợ nhiều định dạng ảnh/video
- Có thể truy cập từ xa qua mạng nội bộ
- Xử lý real-time với feedback

## 🚀 Cài đặt và Chạy

### Yêu cầu hệ thống

- Python >= 3.8
- CUDA compatible GPU (khuyến nghị) hoặc CPU
- RAM >= 8GB
- Windows/Linux/MacOS

### Bước 1: Clone repository

```bash
git clone https://github.com/phucmanh1310/Helmet-Violation-Detection-Using-YOLO-and-VGG16.git
cd Helmet-Violation-Detection-Using-YOLO-and-VGG16
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Chạy giao diện web

Có 2 cách để chạy:

**Cách 1: Quick Start (Khuyến nghị)**

```bash
python quick_start_ui.py
```

**Cách 2: Chạy trực tiếp**

```bash
cd Source
python ui_app.py
```

### Bước 4: Truy cập giao diện

Sau khi chạy thành công, mở trình duyệt và truy cập:

- **Local**: http://127.0.0.1:7860
- **Network**: http://0.0.0.0:7860

## 📖 Hướng dẫn sử dụng

### 1️⃣ Phát hiện trên Ảnh

1. Chuyển sang tab **"📷 Phát hiện trên Ảnh"**
2. Click vào khu vực upload và chọn ảnh
3. Nhấn nút **"🔍 Phát hiện vi phạm"**
4. Xem kết quả:
   - Ảnh với bounding boxes
   - Bảng danh sách vi phạm
   - Thống kê tóm tắt

### 2️⃣ Phát hiện trên Video

1. Chuyển sang tab **"🎥 Phát hiện trên Video"**
2. Click vào khu vực upload và chọn video
3. Nhấn nút **"🔍 Phát hiện vi phạm"**
4. Đợi hệ thống xử lý (có thanh progress)
5. Xem video kết quả và bảng thống kê

## 📊 Cấu trúc Project

```
Helmet-Violation-Detection-Using-YOLO-and-VGG16/
│
├── Source/
│   ├── ui_app.py              # 🆕 Giao diện web Gradio
│   ├── main_app.py            # Script CLI gốc
│   ├── _LP_Helmet.py          # Module xử lý LP và Helmet
│   ├── _Motobike.py           # Module phát hiện xe máy
│   └── _myFunc.py             # Các hàm tiện ích
│
├── models/
│   ├── Motov10l.pt            # Model phát hiện xe máy
│   └── HelmetLP.pt            # Model phát hiện mũ & biển số
│
├── data/                      # Datasets
├── img/                       # Thư mục lưu kết quả
├── quick_start_ui.py          # 🆕 Script khởi động nhanh
├── requirements.txt           # 🆕 Đã thêm gradio
├── HUONG_DAN_UI.md           # 🆕 Hướng dẫn chi tiết UI
└── README_UI.md              # 🆕 File này
```

## ⚙️ Cấu hình

Bạn có thể tùy chỉnh các thông số trong `Source/ui_app.py`:

```python
# Ngưỡng confidence
MOTO_CONF = 0.4              # Phát hiện xe máy
HELMET_LP_CONF = 0.4         # Phát hiện mũ/biển số

# Xử lý video
process_every_n_frames = 5   # Xử lý mỗi N frames

# Server settings
server_port = 7860           # Port web server
share = False                # True để tạo public link
```

## 🎨 Screenshots

### Giao diện chính

![UI Main](https://via.placeholder.com/800x400?text=UI+Main+Screen)

### Kết quả phát hiện trên ảnh

![Result Image](https://via.placeholder.com/800x400?text=Detection+Result)

### Bảng thống kê vi phạm

![Table](https://via.placeholder.com/800x200?text=Violation+Table)

## 🔧 Troubleshooting

### ❌ Lỗi: Models không tìm thấy

```
FileNotFoundError: Helmet/LP detection model not found
```

**Giải pháp**: Đảm bảo các file model tồn tại:

- `models/Motov10l.pt`
- `models/HelmetLP.pt`

### ❌ Lỗi: CUDA/GPU

```
CUDA error
```

**Giải pháp**: Sửa trong `ui_app.py`:

```python
reader = easyocr.Reader(['en'], gpu=False)  # Đổi thành False
```

### ❌ Lỗi: Port đã được sử dụng

```
OSError: [Errno 98] Address already in use
```

**Giải pháp**: Đổi port khác trong code hoặc tắt ứng dụng đang dùng port 7860

### ❌ Lỗi: Out of Memory

**Giải pháp**:

- Tăng `process_every_n_frames` trong code (xử lý ít frame hơn)
- Giảm độ phân giải video đầu vào
- Đóng các ứng dụng khác

## 🎯 Roadmap - Tính năng tương lai

- [ ] Tích hợp nhận diện khuôn mặt
- [ ] Kết nối database để lưu lịch sử
- [ ] Gửi email thông báo tự động
- [ ] Dashboard thống kê theo thời gian
- [ ] Xử lý video real-time từ camera
- [ ] REST API để tích hợp với hệ thống khác
- [ ] Mobile app (React Native/Flutter)
- [ ] Xuất báo cáo PDF/Excel

## 📚 Tài liệu tham khảo

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)

## 📝 Changelog

### Version 2.0 (2024) - UI Update

- ✅ Thêm giao diện web Gradio
- ✅ Hỗ trợ upload ảnh/video qua browser
- ✅ Bảng thống kê vi phạm
- ✅ Progress bar cho xử lý video
- ✅ Responsive design
- ✅ Quick start script

### Version 1.0 - Original

- Phát hiện xe máy bằng YOLOv8
- Phát hiện mũ bảo hiểm và biển số
- OCR đọc biển số
- Giao diện PyQt5

## 👥 Contributors

- **Nguyễn Trọng Thụy** - samnguyen0510@gmail.com
- **Development Team** - Đồ án Thị giác Máy tính

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO team for the amazing object detection framework
- Gradio team for the easy-to-use UI library
- EasyOCR for Vietnamese license plate recognition
- Community contributors

---

**⭐ Nếu project hữu ích, hãy cho một star nhé!**

📧 Contact: samnguyen0510@gmail.com

🔗 GitHub: [Helmet-Violation-Detection](https://github.com/phucmanh1310/Helmet-Violation-Detection-Using-YOLO-and-VGG16)
