# Hướng dẫn chạy dự án Helmet Violation Detection (Local, gọn gàng, chi tiết)

Tài liệu này hướng dẫn chạy dự án trên máy cá nhân theo cách tối giản, tránh sinh nhiều file rác, và có thể test nhanh bằng ảnh hoặc video.

I. Kiến trúc và công nghệ sử dụng

- YOLOv8 (Ultralytics): phát hiện xe máy (model 1). Bạn đã train nhanh và dùng file models/Motov10l.pt.
- Roboflow Inference API: phát hiện mũ bảo hiểm, không mũ và biển số (model 2) dựa trên project lp-helmet (workspace object-detection-project-fn2vr, version 2). Không cần tải .pt.
- EasyOCR: đọc text biển số trực tiếp từ ảnh biển số cắt ra (không dùng VGG16 .h5).
- OpenCV + NumPy: xử lý ảnh và I/O cơ bản.

II. Chuẩn bị môi trường

1. Cài thư viện

- Mở PowerShell ở thư mục dự án, chạy:
  pip install -r requirements.txt

2. Kiểm tra/tạo cấu hình Roboflow

- Tạo file config nếu chưa có:
  python main.py --setup
- Mở config/roboflow_config.json đảm bảo:
  {
  "ROBOFLOW_API_KEY": "<api_key_của_bạn>",
  "ROBOFLOW_WORKSPACE_ID": "object-detection-project-fn2vr",
  "ROBOFLOW_PROJECT_ID": "lp-helmet",
  "ROBOFLOW_VERSION_NUMBER": 2,
  "ROBOFLOW_SIZE": 640
  }

3. Model YOLO xe máy (offline)

- Đặt model đã train vào:
  models/Motov10l.pt
- Nếu bạn vừa train bằng yolo detect train…, copy:
  Copy-Item "runs\detect\train2\weights\best.pt" "models\Motov10l.pt" -Force

III. Cách chạy (khuyến nghị dùng Python 3.11/3.13, PowerShell)

1. Test nhanh với ảnh (không cần video)

- Lệnh:
  python main.py --image path\to\your_image.jpg
- Hệ thống sẽ:
  - Detect xe máy bằng YOLOv8, crop ảnh xe máy vào img/Moto
  - Gọi Roboflow để detect helmet/no-helmet + LP
  - Đọc biển số bằng EasyOCR (nếu có ảnh LP)
  - In ra Vi phạm/Tuân thủ và biển số (UNKNOWN nếu không đọc được)

2. Chạy với video (tùy chọn)

- Lệnh:
  python main.py --video "Video_Demo\Video1.mp4"
- Hệ thống sẽ trích frame theo chu kỳ và áp dụng pipeline tương tự như ảnh.

3. Dọn dẹp file tạm (tránh sinh file lung tung)

- Lệnh:
  python main.py --clean
- Hệ thống sẽ xóa runs/detect và toàn bộ ảnh tạm trong img/\*.

IV. Thay đổi mặc định để kết quả dễ ra hơn (đã áp dụng trong mã)

- \_Motobike.py:
  - conf YOLO từ 0.7 xuống 0.4 để bớt bỏ sót.
  - Không lưu ảnh dự đoán của YOLO vào runs/detect nữa, làm trực tiếp trên frame hiện tại.
- \_LP_Helmet.py:
  - confidence từ 70 xuống 40.
  - Không lưu ảnh kết quả Roboflow ra ổ đĩa để tránh rác.
- main.py:
  - Nếu phát hiện “no-helmet” thì ghi nhận vi phạm luôn, kể cả chưa đọc được biển số (LP sẽ là "UNKNOWN").
  - Thêm chế độ chạy ảnh (--image) và dọn dẹp (--clean).

V. Cấu trúc thư mục chạy local (sau setup)
Helmet-Violation-Detection-Using-YOLO-and-VGG16/
├─ main.py
├─ requirements.txt
├─ HUONG_DAN_CHAY.md
├─ README.md
├─ Source/
│ ├─ \_Motobike.py # detect xe máy (YOLOv8, conf=0.4, không ghi runs)
│ ├─ \_LP_Helmet.py # detect helmet/no-helmet + LP (Roboflow, conf=40)
│ ├─ \_ReadLP.py # (giữ nguyên, hiện không dùng VGG16)
│ └─ \_myFunc.py # tiện ích (xóa/ tạo folder, lấy thông tin giả lập)
├─ models/
│ └─ Motov10l.pt # model YOLO xe máy đã train
├─ config/
│ └─ roboflow_config.json
├─ img/
│ ├─ Traffic/ # ảnh frame gốc (video)
│ ├─ Moto/ # ảnh xe máy đã crop
│ ├─ LP/ # (không dùng để lưu tự động nữa)
│ └─ Thresh_result/ # đường kẻ ngưỡng (nếu cần)
├─ data/
│ ├─ Motobike Detection.v18i.yolov8/ # dataset bạn dùng để train YOLO
│ └─ LP-Helmet.v2i.yolov8/ # dataset tham khảo của Roboflow
└─ Video_Demo/
├─ Video1.mp4
├─ Video2.mp4
└─ Video3_mail.mp4

VI. Lý do thiết kế (giải thích để tự tìm hiểu sâu hơn)

- Vì pipeline cần 2 nhiệm vụ: (1) tìm xe; (2) tìm mũ/LP và đọc biển số, nên tách model 1 và 2:
  - Model 1 (YOLO offline) đảm bảo chạy local nhanh, không phụ thuộc mạng.
  - Model 2 (Roboflow Hosted) giúp bạn không cần train thêm model helmet/LP, giảm độ phức tạp lúc mới chạy.
- EasyOCR thay thế VGG16 giúp không cần tệp .h5 và cho phép thử nghiệm nhanh. Khi đã quen, bạn có thể quay lại train VGG16 theo notebook VGG_Training.ipynb để tối ưu theo font/biển VN.
- Giảm conf giúp tăng Recall (bắt được nhiều hơn), phù hợp cho demo ban đầu. Khi cần độ chính xác cao hơn, hãy tăng lại conf và/hoặc đặt quy tắc lọc sau phát hiện.
- Không lưu ảnh dự đoán YOLO/Roboflow để tránh sinh nhiều file rác.

VII. Các lệnh tham khảo (Windows PowerShell)

- Cài thư viện:
  pip install -r requirements.txt
- Tạo config mẫu + thư mục cần thiết:
  python main.py --setup
- Dọn dẹp file tạm:
  python main.py --clean
- Test nhanh bằng ảnh:
  python main.py --image path\to\your_image.jpg
- Chạy với video demo:
  python main.py --video "Video_Demo\Video1.mp4"
- Train YOLOv8 cho motobike (GPU), ví dụ nhanh:
  yolo detect train model=yolov8n.pt data="data\Motobike Detection.v18i.yolov8\data.yaml" epochs=20 imgsz=640 device=0
  → Copy runs\detect\trainX\weights\best.pt → models\Motov10l.pt

VIII. Troubleshooting (hay gặp)

- Roboflow: “Version not found/Authentication failed”
  - Kiểm tra workspace/project/version đúng như data.yaml của lp-helmet (v2) và API key còn hiệu lực.
- Không thấy vi phạm nào:
  - Hạ conf xuống nữa (YOLO: 0.35; Roboflow: confidence 30), thử ảnh rõ hơn/cận hơn.
  - Đảm bảo ảnh crop xe máy trong img/Moto đủ lớn, thấy rõ người + mũ.
- PyTorch GPU lỗi bộ nhớ/WinError 1455:
  - Train với batch nhỏ hơn, workers=0, imgsz giảm (416/512).
  - Tăng Windows Page File (Virtual Memory) hoặc dùng CPU để train nhanh demo.

IX. Nâng cấp tiếp theo (nếu muốn xây GUI như demo gốc)

- Dùng PyQt5 tạo UI: chọn file ảnh/video, bấm Detect, show danh sách vi phạm và ảnh minh họa.
- Chạy inference đa luồng để mượt hơn, hiển thị thanh tiến trình.
- Thay Roboflow bằng model YOLOv8 (offline) cho helmet+LP để chạy hoàn toàn offline.

Chúc bạn thành công. Nếu cần, hãy mở issue mô tả log/lỗi để tôi hỗ trợ tiếp.
