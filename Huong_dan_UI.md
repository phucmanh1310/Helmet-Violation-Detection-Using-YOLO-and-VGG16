# 🚀 HƯỚNG DẪN BẮT ĐẦU SỬ DỤNG

## ✅ Đã cài đặt thành công!

Hệ thống giao diện web UI để phát hiện vi phạm mũ bảo hiểm đã sẵn sàng!

---

## 🎯 CÁCH CHẠY NHANH NHẤT

### Cách 1: Chạy bằng file BAT (Khuyến nghị)
**Double-click vào file:**
```
CHAY_UI.bat
```

### Cách 2: Chạy từ PowerShell/Terminal
```powershell
cd "d:\hoctap\Năm 4\HK1\Python_CoTrang\project\Helmet-Violation-Detection-Using-YOLO-and-VGG16"
& py -3.13 quick_start_ui.py
```

### Cách 3: Chạy trực tiếp
```powershell
cd Source
& py -3.13 ui_app.py
```

---

## 🌐 TRUY CẬP GIAO DIỆN

Sau khi chạy, mở trình duyệt web và truy cập:

### 🏠 Trên máy tính của bạn:
```
http://127.0.0.1:7860
```

### 📱 Từ thiết bị khác trong cùng mạng:
```
http://0.0.0.0:7860
```

hoặc

```
http://<IP-may-tinh-cua-ban>:7860
```

---

## 📖 HƯỚNG DẪN SỬ DỤNG

### Tab 1: 📷 Phát hiện trên Ảnh

1. Click vào khu vực **"Upload ảnh"**
2. Chọn ảnh chứa xe máy từ máy tính
3. Nhấn nút **"🔍 Phát hiện vi phạm"**
4. Xem kết quả:
   - Ảnh với bounding box (đỏ = vi phạm, xanh = tuân thủ)
   - Bảng danh sách vi phạm
   - Thống kê tóm tắt

### Tab 2: 🎥 Phát hiện trên Video

1. Click vào khu vực **"Upload video"**
2. Chọn video giao thông từ máy tính
3. Nhấn nút **"🔍 Phát hiện vi phạm"**
4. Đợi hệ thống xử lý (có progress bar)
5. Xem video kết quả và bảng danh sách vi phạm

### Tab 3: ℹ️ Hướng dẫn

Xem thêm thông tin chi tiết về cách sử dụng

---

## 🎨 GIẢI THÍCH MÀU SẮC

- 🟢 **Màu xanh lá**: Tuân thủ (đội mũ bảo hiểm)
- 🔴 **Màu đỏ**: Vi phạm (không đội mũ bảo hiểm)

---

## 📊 BẢNG THỐNG KÊ VI PHẠM

Bảng hiển thị các cột:
- **STT**: Số thứ tự vi phạm
- **Họ và tên**: Chưa xác định (có thể tích hợp sau)
- **Gmail**: Chưa có
- **Biển số**: Tự động nhận diện bằng OCR
- **Thời gian**: Thời gian phát hiện
- **ID**: Mã định danh vi phạm (MVxxx)

---

## ⚙️ CẤU HÌNH HỆ THỐNG

### Phiên bản đang sử dụng:
- **Python**: 3.13
- **Lệnh chạy**: `& py -3.13`

### Các thư viện chính:
- ✅ opencv-python: 4.12.0.88
- ✅ ultralytics: 8.0.196
- ✅ easyocr: 1.7.2
- ✅ gradio: 5.49.1
- ✅ torch: 2.6.0+cu124

### Models:
- ✅ `models/Motov10l.pt` - Phát hiện xe máy
- ✅ `models/HelmetLP.pt` - Phát hiện mũ bảo hiểm và biển số

---

## 🔧 XỬ LÝ LỖI

### ❌ Lỗi: Port đã được sử dụng
**Giải pháp**:
1. Đóng ứng dụng đang sử dụng port 7860
2. Hoặc sửa port trong file `Source/ui_app.py`:
```python
server_port=7861  # Đổi thành port khác
```

### ❌ Lỗi: Models không tìm thấy
**Giải pháp**:
Đảm bảo các file model tồn tại tại:
- `models/Motov10l.pt`
- `models/HelmetLP.pt`

### ❌ Lỗi: CUDA/GPU
**Giải pháp**:
Nếu không có GPU NVIDIA, sửa trong `Source/ui_app.py`:
```python
reader = easyocr.Reader(['en'], gpu=False)  # Đổi thành False
```

### ❌ Lỗi: Out of Memory
**Giải pháp**:
Sửa trong `Source/ui_app.py`:
```python
process_every_n_frames = 10  # Tăng số này để xử lý ít frame hơn
```

---

## 📁 CẤU TRÚC PROJECT

```
Helmet-Violation-Detection-Using-YOLO-and-VGG16/
│
├── CHAY_UI.bat                    # ⭐ File chạy nhanh
├── BAT_DAU_SU_DUNG.md             # ⭐ File này
├── quick_start_ui.py              # Script khởi động
├── HUONG_DAN_UI.md                # Hướng dẫn chi tiết
├── README_UI.md                   # README phiên bản UI
│
├── Source/
│   ├── ui_app.py                  # ⭐ Giao diện web chính
│   ├── main_app.py                # Script CLI gốc
│   ├── _LP_Helmet.py              # Module xử lý
│   ├── _Motobike.py               # Module phát hiện
│   └── _myFunc.py                 # Hàm tiện ích
│
├── models/
│   ├── Motov10l.pt                # Model xe máy
│   └── HelmetLP.pt                # Model mũ & biển số
│
├── data/                          # Datasets
├── img/                           # Kết quả
└── requirements.txt               # Dependencies
```

---

## 🎯 TÍNH NĂNG NỔI BẬT

### ✨ Giao diện hiện đại
- Upload ảnh/video qua trình duyệt
- Không cần cài đặt phần mềm phức tạp
- Responsive - chạy mọi thiết bị

### 🤖 AI Detection
- Phát hiện xe máy tự động
- Nhận diện mũ bảo hiểm
- OCR đọc biển số xe

### 📊 Báo cáo chi tiết
- Bảng thống kê vi phạm
- Bounding boxes màu sắc rõ ràng
- Thông tin tóm tắt

### ⚡ Hiệu suất
- Xử lý nhanh với GPU
- Progress bar cho video
- Hỗ trợ nhiều định dạng

---

## 🚀 TÍNH NĂNG TƯƠNG LAI

- [ ] Nhận diện khuôn mặt người vi phạm
- [ ] Lưu database lịch sử vi phạm
- [ ] Gửi email thông báo tự động
- [ ] Dashboard thống kê theo thời gian
- [ ] Xử lý real-time từ camera
- [ ] REST API
- [ ] Mobile app

---

## 💡 MẸO SỬ DỤNG

### Để có kết quả tốt nhất:
1. **Ảnh/Video chất lượng cao**: Độ phân giải càng cao càng tốt
2. **Góc nhìn rõ ràng**: Camera nhìn thẳng hoặc nghiêng 45 độ
3. **Ánh sáng tốt**: Tránh ảnh quá tối hoặc quá sáng
4. **Biển số rõ ràng**: Không bị che khuất hoặc mờ

### Để xử lý nhanh hơn:
1. Giảm độ phân giải video trước khi upload
2. Tăng `process_every_n_frames` trong code
3. Sử dụng GPU nếu có

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
1. Xem phần "🔧 Xử lý lỗi" ở trên
2. Đọc file `HUONG_DAN_UI.md` để biết chi tiết
3. Kiểm tra console/terminal để xem thông báo lỗi

---

## 📝 GHI CHÚ QUAN TRỌNG

- ⚠️ Server chạy ở **port 7860**
- ⚠️ Nhấn **Ctrl+C** trong terminal để dừng server
- ⚠️ Sử dụng **Python 3.13** (lệnh: `& py -3.13`)
- ⚠️ Cần **GPU NVIDIA** để chạy nhanh (không bắt buộc)

---

## ✅ CHECKLIST TRƯỚC KHI CHẠY

- [ ] Python 3.13 đã cài đặt
- [ ] Đã cài đặt các thư viện (gradio, opencv-python, ultralytics, easyocr)
- [ ] File models tồn tại (`models/Motov10l.pt`, `models/HelmetLP.pt`)
- [ ] Port 7860 chưa được sử dụng

---

**🎉 Chúc bạn sử dụng thành công!**

📧 Email: samnguyen0510@gmail.com
🔗 GitHub: https://github.com/phucmanh1310/Helmet-Violation-Detection-Using-YOLO-and-VGG16

---

**Version**: 2.0 - UI Update
**Ngày**: 2024
**Đồ án**: Thị giác Máy tính
