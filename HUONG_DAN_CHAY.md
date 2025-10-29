# Hướng dẫn chạy dự án Helmet Violation Detection (Full Local)

Tài liệu này hướng dẫn bạn cách chạy ứng dụng phát hiện vi phạm mũ bảo hiểm sau khi đã được tái cấu trúc để chạy hoàn toàn trên máy cá nhân (local).

### I. Kiến trúc và Công nghệ

Ứng dụng sử dụng một pipeline xử lý gồm 3 bước, tất cả đều chạy local:

1.  **Model 1 (YOLOv8)**: Phát hiện xe máy trong ảnh đầu vào. Sử dụng model `models/Motov10l.pt`.
2.  **Model 2 (YOLOv8)**: Từ ảnh xe máy đã được cắt ra, phát hiện các đối tượng: `helmet` (mũ bảo hiểm), `no helmet` (không mũ), và `LP` (biển số). Sử dụng model `models/HelmetLP.pt`.
3.  **OCR (EasyOCR)**: Nếu phát hiện được biển số (`LP`), ảnh biển số sẽ được cắt ra và đưa vào EasyOCR để đọc và nhận dạng ký tự.

### II. Yêu cầu trước khi chạy

1.  **Cài đặt thư viện**: Đảm bảo bạn đã cài tất cả các thư viện cần thiết.

    ```powershell
    # Mở PowerShell tại thư mục gốc dự án
    & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
    ```

2.  **Train và chuẩn bị Models**: Đây là bước quan trọng nhất. Bạn phải train và có đủ 2 file model YOLOv8.

    - `models/Motov10l.pt` (phát hiện xe máy)
    - `models/HelmetLP.pt` (phát hiện mũ/biển số)

    ➡️ **Để biết cách train chi tiết, vui lòng đọc kỹ file: `HUONG_DAN_TRAIN.md`**

### III. Cách chạy ứng dụng

File thực thi chính của dự án là `Source/main_app.py`. Mọi thao tác sẽ được thực hiện qua file này từ dòng lệnh PowerShell tại thư mục gốc của dự án.

1.  **Chạy phát hiện trên một ảnh (Khuyến nghị)**

    Đây là cách tốt nhất để kiểm tra pipeline. Thay thế đường dẫn ảnh bằng ảnh của bạn.

    ```powershell
    & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" .\Source\main_app.py --image "đường\dẫn\tới\ảnh\của\bạn.jpg"
    ```

    **Ví dụ:**

    ```powershell
    & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" .\Source\main_app.py --image ".\img\test\test1.jpg"
    ```

2.  **Dọn dẹp thư mục tạm**

    Lệnh này sẽ xóa các ảnh đã được cắt ra trong các lần chạy trước (`img/Moto_Crops/`, `img/LP_Crops/`).

    ```powershell
    & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" .\Source\main_app.py --image "your_image.jpg" --clean
    ```

### IV. Kết quả đầu ra

Khi chạy, chương trình sẽ in kết quả ra màn hình console:

- `🚨 VIOLATION`: Nếu phát hiện `no helmet`.
- `✅ Compliance`: Nếu phát hiện `helmet`.
- `⚠️ Could not determine`: Nếu không phát hiện được mũ bảo hiểm.

Biển số xe (nếu đọc được) sẽ được đính kèm với mỗi kết quả.

Các ảnh cắt ra để xử lý (xe máy, biển số) sẽ được lưu trong các thư mục `img/Moto_Crops/` và `img/LP_Crops/` để bạn có thể kiểm tra trực quan.

### V. Cấu trúc thư mục dự án (sau khi tái cấu trúc)

```
Helmet-Violation-Detection-Using-YOLO-and-VGG16/
├── Source/
│   ├── main_app.py         # << FILE CHẠY CHÍNH
│   ├── _Motobike.py        # (Logic được tích hợp vào main_app)
│   ├── _LP_Helmet.py       # (Logic được tích hợp vào main_app)
│   └── _myFunc.py          # (Chứa các hàm phụ trợ)
├── models/
│   ├── Motov10l.pt         # Model YOLOv8 phát hiện xe máy (cần train)
│   └── HelmetLP.pt         # Model YOLOv8 phát hiện mũ/biển số (cần train)
├── img/
│   ├── test/               # Chứa ảnh để bạn kiểm tra
│   ├── Moto_Crops/         # Lưu ảnh xe máy được cắt ra tự động
│   └── LP_Crops/           # Lưu ảnh biển số được cắt ra tự động
├── data/                   # Chứa datasets để train models
├── Video_Demo/             # Chứa video demo (hiện không dùng)
├── requirements.txt
├── HUONG_DAN_CHAY.md       # << File hướng dẫn này
└── HUONG_DAN_TRAIN.md      # File hướng dẫn train model
```

### VI. Xử lý lỗi thường gặp

- **Lỗi `FileNotFoundError` cho model**: Đảm bảo bạn đã train và đặt file `.pt` vào đúng thư mục `models/` với tên chính xác.
- **Không phát hiện được gì**: Thử giảm ngưỡng tin cậy (`CONF`) trong file `Source/main_app.py`. Hoặc, chất lượng model của bạn chưa đủ tốt, hãy train lại với nhiều epochs hơn hoặc dataset chất lượng hơn.
- **Lỗi `No module named 'ultralytics'`**: Bạn đang chạy sai môi trường Python. Hãy luôn sử dụng đường dẫn đầy đủ đến file `python.exe` đã cài thư viện, ví dụ: `C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe`.
