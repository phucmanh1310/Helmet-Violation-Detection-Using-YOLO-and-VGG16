# Hướng dẫn TRAIN mô hình (Full Local) trước khi kiểm tra ảnh/video

Tài liệu này hướng dẫn bạn train 2 mô hình YOLOv8 để chạy hoàn toàn local:

- Model 1: Motobike Detection (phát hiện xe máy) → file đầu ra: models/Motov10l.pt
- Model 2: Helmet + LP Detection (mũ bảo hiểm, không mũ, biển số) → file đầu ra: models/HelmetLP.pt
- OCR: dùng EasyOCR (không cần train)

Lưu ý công cụ và môi trường

- Windows + PowerShell
- Python đã cài Ultralytics (yolo.exe), Torch GPU (khuyến nghị) và các lib trong requirements.txt
- Luôn dùng đúng Python 3.13 (hoặc 3.11) mà bạn đã cài thư viện
  - Ví dụ dùng tuyệt đối:
    - C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe
    - C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\Scripts\yolo.exe

I. Chuẩn bị thư mục dự án

1. Mở PowerShell tại thư mục dự án
   VD: D:\hoctap\Năm 4\HK1\Python_CoTrang\project\Helmet-Violation-Detection-Using-YOLO-and-VGG16

2. Tạo sẵn một số thư mục
   mkdir data
   mkdir models

3. Cài thư viện (nếu chưa)
   & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt

II. Chuẩn bị dataset (2 cách)
A. Tải thủ công từ Roboflow Universe (đơn giản nhất)

- Motobike Detection:

  - Truy cập: https://universe.roboflow.com/cdio-zmfmj/motobike-detection
  - Chọn "Download YOLOv8"
  - Giải nén vào: data\motobike_yolo\
  - Kiểm tra có file: data\motobike_yolo\data.yaml và các thư mục train/, valid/, test/

- LP-Helmet:
  - Truy cập: https://universe.roboflow.com/object-detection-project-fn2vr/lp-helmet
  - Chọn "Download YOLOv8"
  - Giải nén vào: data\lp_helmet_yolo\
  - Kiểm tra có file: data\lp_helmet_yolo\data.yaml và các thư mục train/, valid/, test/

B. Tải tự động bằng Roboflow SDK (yêu cầu API key)

- Tạo file download_datasets.py với nội dung:
  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_API_KEY")

  # Motobike

  rf.workspace("cdio-zmfmj").project("motobike-detection").version(18).download("yolov8", location="data/motobike_yolo")

  # LP-Helmet

  rf.workspace("object-detection-project-fn2vr").project("lp-helmet").version(2).download("yolov8", location="data/lp_helmet_yolo")

- Chạy:
  & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" .\download_datasets.py

C. Sửa đường dẫn trong data.yaml (nếu cần)

- Mở file data.yaml trong mỗi dataset, đảm bảo 3 dòng đầu:
  train: train/images
  val: valid/images
  test: test/images
- Nếu đang là dạng ../train/images, hãy sửa lại như trên để tránh lỗi đường dẫn.

III. Train YOLOv8 cho Motobike (Model 1)

1. Lệnh khuyến nghị (GPU)
   & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\Scripts\yolo.exe" detect train ^
   model=yolov8n.pt ^
   data="data\motobike_yolo\data.yaml" ^
   epochs=50 imgsz=640 device=0

2. Nếu GPU thiếu bộ nhớ (WinError 1455, OOM), dùng cấu hình nhẹ hơn
   & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\Scripts\yolo.exe" detect train ^
   model=yolov8n.pt ^
   data="data\motobike_yolo\data.yaml" ^
   epochs=30 imgsz=512 device=0 batch=8 workers=0 amp=False

3. Sau khi train xong, copy model tốt nhất
   New-Item -ItemType Directory -Force -Path models | Out-Null
   Copy-Item "runs\detect\train\weights\best.pt" "models\Motov10l.pt" -Force
   (Nếu Ultralytics tạo train2/train3…, hãy thay đúng đường dẫn.)

4. (Tùy chọn) Đánh giá nhanh model
   "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\Scripts\yolo.exe" detect val ^
   model=models\Motov10l.pt data="data\motobike_yolo\data.yaml"

IV. Train YOLOv8 cho Helmet + LP (Model 2)

1. Lệnh khuyến nghị (GPU)
   & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\Scripts\yolo.exe" detect train ^
   model=yolov8n.pt ^
   data="data\lp_helmet_yolo\data.yaml" ^
   epochs=50 imgsz=640 device=0

2. Nếu GPU thiếu bộ nhớ
   & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\Scripts\yolo.exe" detect train ^
   model=yolov8n.pt ^
   data="data\lp_helmet_yolo\data.yaml" ^
   epochs=30 imgsz=512 device=0 batch=8 workers=0 amp=False

3. Copy model tốt nhất vào models/HelmetLP.pt
   Copy-Item "runs\detect\train\weights\best.pt" "models\HelmetLP.pt" -Force

4. (Tùy chọn) Đánh giá nhanh model
   "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\Scripts\yolo.exe" detect val ^
   model=models\HelmetLP.pt data="data\lp_helmet_yolo\data.yaml"

V. Tích hợp 2 model vào pipeline (chạy full local)

- Sau khi bạn có:
  - models/Motov10l.pt (Motobike)
  - models/HelmetLP.pt (Helmet + LP)
- (Nếu chưa được tích hợp sẵn) Trong mã nguồn main.py/\_LP_Helmet.py sẽ cần:
  - Load YOLO("models/HelmetLP.pt") thay vì Roboflow API.
  - Parse kết quả YOLO: r.boxes.xyxy, r.boxes.cls và r.names để xác định class "LP", "helmet", "no helmet" theo data.yaml.
- Trong bản code hiện tại, tôi có thể patch giúp nếu bạn yêu cầu (để chạy hoàn toàn offline).

VI. Kiểm tra sau khi train (chạy ảnh/video)

- Dọn rác (tùy chọn):
  & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" .\main.py --clean

- Test 1 ảnh (khuyến nghị cho newbie):
  & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" .\main.py --image "path\to\your_image.jpg"

- Chạy video (tùy chọn):
  & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" .\main.py --video "Video_Demo\Video1.mp4"

VII. Mẹo và xử lý lỗi thường gặp

1. WinError 1455 / OOM khi train GPU

- Giảm batch (16 → 8 → 4), đặt workers=0, imgsz=512/416, tắt AMP (amp=False).
- Tăng Windows Page File (Virtual Memory): Control Panel → System → Advanced → Performance → Advanced → Virtual memory → Custom size (Initial 8–16 GB, Max 16–32 GB).

2. data.yaml lỗi đường dẫn

- Sửa thành:
  train: train/images
  val: valid/images
  test: test/images
- Đảm bảo thư mục labels tương ứng tồn tại.

3. Python không nhận ultralytics/yolo

- Luôn dùng đúng interpreter tuyệt đối:
  & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\python.exe" …
  & "C:\Users\PhucManh\AppData\Local\Programs\Python\Python313\Scripts\yolo.exe" …

4. Kết quả detect yếu/ít

- Tăng epochs (50 → 100+), tăng imgsz 640 → 896 (nếu đủ VRAM), hoặc tinh chỉnh conf trong code.

5. Xuất model (tùy chọn)

- Có thể export sang onnx/engine nếu cần:
  yolo export model=models\Motov10l.pt format=onnx
  yolo export model=models\HelmetLP.pt format=onnx

—
Sau khi hoàn thành 2 bước train và copy best.pt vào models/, bạn có thể quay lại HUONG_DAN_CHAY.md để chạy test ảnh/video. Nếu muốn tôi patch code để dùng YOLO local cho Helmet+LP (bỏ Roboflow), hãy báo tôi để thực hiện ngay.
