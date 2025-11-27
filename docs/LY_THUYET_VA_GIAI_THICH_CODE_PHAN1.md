# 📚 LÝ THUYẾT VÀ GIẢI THÍCH CODE - PHẦN 1

> **Tài liệu lý thuyết và giải thích code cho dự án Helmet Violation Detection**  
> **Phần 1: Lý thuyết cơ bản & Thư viện**

---

## 📋 Mục lục Phần 1

1. [Lý thuyết về Object Detection](#1-lý-thuyết-về-object-detection)
2. [Kiến trúc YOLO](#2-kiến-trúc-yolo)
3. [Thư viện PyTorch](#3-thư-viện-pytorch)
4. [Thư viện Ultralytics](#4-thư-viện-ultralytics)
5. [Thư viện OpenCV](#5-thư-viện-opencv)
6. [Thư viện EasyOCR](#6-thư-viện-easyocr)
7. [Thư viện Gradio](#7-thư-viện-gradio)

---

## 1. Lý thuyết về Object Detection

### 1.1 Object Detection là gì?

**Object Detection** (Phát hiện đối tượng) là một bài toán trong Computer Vision, có nhiệm vụ:

- **Phát hiện** (Detection): Tìm vị trí của đối tượng trong ảnh
- **Phân loại** (Classification): Xác định class của đối tượng
- **Định vị** (Localization): Vẽ bounding box xung quanh đối tượng

**Output**: Bounding box (x, y, w, h) + Class label + Confidence score

### 1.2 Các phương pháp Object Detection

```
┌─────────────────────────────────────────┐
│     Object Detection Approaches        │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────┐    ┌────────────────┐ │
│  │ Two-Stage   │    │  One-Stage     │ │
│  │ Detectors   │    │  Detectors     │ │
│  ├─────────────┤    ├────────────────┤ │
│  │ R-CNN       │    │  YOLO          │ │
│  │ Fast R-CNN  │    │  SSD           │ │
│  │ Faster R-CNN│    │  RetinaNet     │ │
│  │ Mask R-CNN  │    │  EfficientDet  │ │
│  └─────────────┘    └────────────────┘ │
│                                         │
│  High Accuracy       High Speed        │
│  Slow (2-40 FPS)     Fast (30-150 FPS) │
└─────────────────────────────────────────┘
```

#### Two-Stage Detectors

**Quy trình**:

1. **Stage 1**: Region Proposal Network (RPN) → Đề xuất các vùng có khả năng chứa object
2. **Stage 2**: Classification & Refinement → Phân loại và tinh chỉnh bounding box

**Ví dụ**: Faster R-CNN

- **Ưu điểm**: Độ chính xác cao (mAP > 0.90)
- **Nhược điểm**: Chậm (~5-7 FPS), không real-time

#### One-Stage Detectors

**Quy trình**: Trực tiếp dự đoán bounding box + class trong một lần forward pass

**Ví dụ**: YOLO (You Only Look Once)

- **Ưu điểm**: Rất nhanh (30-150 FPS), real-time
- **Nhược điểm**: Accuracy thấp hơn Two-Stage một chút

### 1.3 Các khái niệm quan trọng

#### Intersection over Union (IoU)

IoU đo độ overlap giữa predicted box và ground truth box. Là giá trị thể hiện sự giao thoa giữa hai box bất kỳ. Nếu hai box giao thoa càng nhiều thì giá trị này càng lớn.

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{A \cap B}{A \cup B}
$$

```
Ví dụ:
Ground Truth: [10, 10, 50, 50]
Prediction:   [15, 15, 55, 55]

IoU = 0.68 (overlap tốt)
```

**Ứng dụng**:

- IoU > 0.5: Prediction "tốt"
- IoU > 0.7: Prediction "rất tốt"
- IoU < 0.5: Prediction "kém"

#### Non-Maximum Suppression (NMS)

**Vấn đề**: Model có thể dự đoán nhiều bounding box cho cùng 1 đối tượng

**Giải pháp**: NMS loại bỏ các box trùng lặp

**Thuật toán NMS**:

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: List of bounding boxes [x1, y1, x2, y2]
    scores: Confidence scores
    iou_threshold: Ngưỡng IoU
    """
    # Sắp xếp boxes theo scores giảm dần
    sorted_indices = scores.argsort()[::-1]

    keep = []
    while len(sorted_indices) > 0:
        # Lấy box có score cao nhất
        current = sorted_indices[0]
        keep.append(current)

        # Tính IoU với các box còn lại
        ious = compute_iou(boxes[current], boxes[sorted_indices[1:]])

        # Loại bỏ boxes có IoU > threshold
        sorted_indices = sorted_indices[1:][ious <= iou_threshold]

    return keep
```

#### Precision, Recall, mAP

**Precision** (Độ chính xác):

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- **TP** (True Positive): Dự đoán đúng object
- **FP** (False Positive): Dự đoán sai (detect nhầm)

**Recall** (Độ phủ):

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- **FN** (False Negative): Bỏ sót object

**mAP** (mean Average Precision): là điểm số thể hiện mức độ chính xác của một mô hình học sâu khi thực hiện các tác vụ liên quan đến truy xuất thông tin thị giác, chẳng hạn như phát hiện và nhận dạng các đối tượng khác nhau trong một hình ảnh.

$$
\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} AP_i
$$

- **mAP@0.5**: Average Precision ở IoU threshold = 0.5
- **mAP@0.5:0.95**: Average Precision ở nhiều IoU thresholds

**Ví dụ**:

```
Model A: mAP@0.5 = 0.87 → Tốt
Model B: mAP@0.5 = 0.92 → Rất tốt
```

---

## 2. Kiến trúc YOLO

### 2.1 YOLO là gì?

**YOLO** (You Only Look Once) là một họ các model object detection **one-stage**, nổi tiếng về tốc độ real-time.

**Lịch sử phát triển**:

```
YOLOv1 (2015) → YOLOv2 (2016) → YOLOv3 (2018) → YOLOv4 (2020)
                                                     ↓
                                    YOLOv5 (2020, Ultralytics)
                                                     ↓
                                    YOLOv8 (2023, Ultralytics) ⭐
                                                     ↓
                                    YOLO11 (2024, Ultralytics)
```

### 2.2 YOLOv8 Architecture

Dự án này sử dụng **YOLOv8** từ Ultralytics.

#### Kiến trúc tổng quan

```
┌────────────────────────────────────────────────────────────┐
│                    YOLOv8 Architecture                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input Image (640x640)                                     │
│       ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  BACKBONE (Feature Extraction)                      │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │  - CSPDarknet (Conv + C2f blocks)                   │  │
│  │  - Extract features at multiple scales              │  │
│  │  - Output: Feature maps P3, P4, P5                  │  │
│  └─────────────────────────────────────────────────────┘  │
│       ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  NECK (Feature Fusion)                              │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │  - FPN (Feature Pyramid Network)                    │  │
│  │  - PAN (Path Aggregation Network)                   │  │
│  │  - Fuse multi-scale features                        │  │
│  └─────────────────────────────────────────────────────┘  │
│       ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  HEAD (Detection)                                   │  │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  │
│  │  - Decoupled head (separate box + class branches)  │  │
│  │  - Anchor-free design                               │  │
│  │  - Output: Bounding boxes + Classes + Confidence   │  │
│  └─────────────────────────────────────────────────────┘  │
│       ↓                                                    │
│  Post-processing (NMS)                                     │
│       ↓                                                    │
│  Final Detections                                          │
└────────────────────────────────────────────────────────────┘
```

#### Các cải tiến của YOLOv8

| Feature      | YOLOv5       | YOLOv8               | Lợi ích               |
| ------------ | ------------ | -------------------- | --------------------- |
| **Backbone** | CSPDarknet   | CSPDarknet + C2f     | Faster, lighter       |
| **Neck**     | PANet        | FPN + PAN            | Better feature fusion |
| **Head**     | Coupled      | **Decoupled**        | Higher accuracy       |
| **Anchor**   | Anchor-based | **Anchor-free**      | Simpler, faster       |
| **Loss**     | BCE + CIoU   | **BCE + DFL + CIoU** | Better box regression |

### 2.3 YOLOv8 Model Variants

YOLOv8 có nhiều variants phục vụ các use case khác nhau:

| Model       | Parameters | mAP@0.5 | Speed (RTX 3050) | Use Case             |
| ----------- | ---------- | ------- | ---------------- | -------------------- |
| **YOLOv8n** | 3.2M       | 0.37    | ~80 FPS          | Mobile, edge devices |
| **YOLOv8s** | 11.2M      | 0.44    | ~60 FPS          | Balanced             |
| **YOLOv8m** | 25.9M      | 0.50    | ~40 FPS          | Good accuracy        |
| **YOLOv8l** | 43.7M      | 0.53    | ~25 FPS          | High accuracy ⭐     |
| **YOLOv8x** | 68.2M      | 0.54    | ~15 FPS          | Maximum accuracy     |

**Dự án này sử dụng**:

- **Model 1**: YOLOv8l variant (Motov10l.pt)
- **Model 2**: YOLOv8 medium/large (HelmetLP.pt)

### 2.4 YOLO Training Process

#### Loss Function

YOLOv8 sử dụng **composite loss**:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}
$$

**1. Box Loss** (CIoU):

$$
\mathcal{L}_{\text{box}} = 1 - \text{CIoU}(b, \hat{b})
$$

- **CIoU** (Complete IoU): IoU + distance + aspect ratio

**2. Classification Loss** (BCE):

$$
\mathcal{L}_{\text{cls}} = -\sum_{i} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$

**3. Distribution Focal Loss** (DFL):

- Cải thiện box regression accuracy

#### Data Augmentation

YOLOv8 sử dụng augmentation để tránh overfitting:

```python
augmentation_techniques = {
    'mosaic': True,        # Ghép 4 ảnh thành 1
    'mixup': True,         # Trộn 2 ảnh với alpha blending
    'copy_paste': True,    # Copy-paste objects
    'random_perspective': {
        'rotation': 10,    # Xoay ±10°
        'translation': 0.1,# Dịch chuyển 10%
        'scale': 0.5,      # Scale ±50%
        'shear': 10        # Biến dạng ±10°
    },
    'hsv_augment': {
        'hue': 0.015,      # Thay đổi màu sắc
        'saturation': 0.7, # Độ bão hòa
        'value': 0.4       # Độ sáng
    },
    'flip': 0.5            # Lật ngang 50%
}
```

**Ví dụ Mosaic Augmentation**:

```
┌─────────┬─────────┐
│  Img 1  │  Img 2  │
│  (moto) │ (helmet)│
├─────────┼─────────┤
│  Img 3  │  Img 4  │
│  (LP)   │  (moto) │
└─────────┴─────────┘
→ Merged into 1 training image
```

---

## 3. Thư viện PyTorch

### 3.1 PyTorch là gì?

**PyTorch** là một open-source machine learning framework phát triển bởi Meta (Facebook AI).

**Đặc điểm**:

- Dynamic computational graph (define-by-run)
- Pythonic API, dễ học
- GPU acceleration (CUDA)
- Ecosystem phong phú (TorchVision, TorchAudio, etc.)

### 3.2 Tensor - Cấu trúc dữ liệu cơ bản

**Tensor** tương tự NumPy array nhưng có thể chạy trên GPU:

```python
import torch

# Tạo tensor
x = torch.tensor([1, 2, 3])
print(x.shape)  # torch.Size([3])

# Tensor 2D (matrix)
matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix.shape)  # torch.Size([2, 2])

# Tensor 3D (ảnh RGB)
image = torch.randn(3, 640, 640)  # [C, H, W]
print(image.shape)  # torch.Size([3, 640, 640])

# Tensor 4D (batch images)
batch = torch.randn(16, 3, 640, 640)  # [B, C, H, W]
print(batch.shape)  # torch.Size([16, 3, 640, 640])
```

#### Tensor Operations

```python
# Arithmetic
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
c = a + b  # [4.0, 6.0]
d = a * b  # [3.0, 8.0]

# Matrix multiplication
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.matmul(A, B)  # [2, 4]

# Reshape
x = torch.randn(12)
y = x.view(3, 4)  # [3, 4]
z = x.reshape(2, 6)  # [2, 6]

# Indexing
img = torch.randn(3, 640, 640)
red_channel = img[0]  # [640, 640]
crop = img[:, 100:200, 100:200]  # [3, 100, 100]
```

### 3.3 GPU Acceleration

```python
# Kiểm tra GPU
print(torch.cuda.is_available())  # True nếu có GPU
print(torch.cuda.get_device_name(0))  # "NVIDIA GeForce RTX 3050"

# Chuyển tensor sang GPU
x = torch.randn(1000, 1000)
x_gpu = x.to('cuda')  # hoặc x.cuda()

# Operations trên GPU
y_gpu = x_gpu @ x_gpu.T  # Matrix multiply trên GPU
y_cpu = y_gpu.to('cpu')  # Chuyển về CPU

# Context manager
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(100, 100, device=device)
```

### 3.4 Autograd - Automatic Differentiation

PyTorch tự động tính gradient cho backpropagation:

```python
# Tạo tensor với requires_grad=True
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = x^2
y.backward()  # Tính dy/dx
print(x.grad)  # tensor([4.0]) = 2*x tại x=2

# Ví dụ training loop đơn giản
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    # Forward pass
    pred = model(inputs)
    loss = torch.nn.functional.mse_loss(pred, targets)

    # Backward pass
    optimizer.zero_grad()  # Reset gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
```

### 3.5 Lưu và Load Models

```python
# Lưu toàn bộ model
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# Lưu chỉ weights (khuyến nghị)
torch.save(model.state_dict(), 'weights.pth')
model.load_state_dict(torch.load('weights.pth'))

# Lưu checkpoint training
checkpoint = {
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.25
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**⚠️ PyTorch 2.6 Breaking Change**:

Từ PyTorch 2.6, `torch.load()` mặc định `weights_only=True` (chỉ load weights, không load arbitrary Python objects) để bảo mật.

```python
# ❌ Lỗi trong PyTorch 2.6
model = torch.load('model.pth')  # UnpicklingError!

# ✅ Cách fix
model = torch.load('model.pth', weights_only=False)

# Hoặc monkey-patch (dùng trong dự án này)
import torch
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, weights_only=False, **kwargs)
```

---

## 4. Thư viện Ultralytics

### 4.1 Ultralytics là gì?

**Ultralytics** là công ty phát triển YOLO framework hiện đại nhất, bao gồm:

- YOLOv5 (2020)
- YOLOv8 (2023) ⭐
- YOLO11 (2024)

**Cài đặt**:

```bash
pip install ultralytics==8.0.196
```

### 4.2 Sử dụng YOLO để Inference

#### Load model

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')  # nano variant

# Hoặc load custom trained model
model = YOLO('models/Motov10l.pt')
```

#### Inference trên ảnh

```python
# Dự đoán trên 1 ảnh
results = model('image.jpg')

# Dự đoán trên nhiều ảnh
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Dự đoán với config
results = model(
    'image.jpg',
    conf=0.4,      # Confidence threshold
    iou=0.5,       # NMS IoU threshold
    imgsz=640,     # Input size
    device='cuda'  # GPU
)
```

#### Xử lý Results

```python
results = model('traffic.jpg')

# Lấy boxes
for result in results:
    boxes = result.boxes  # Boxes object

    # Thông tin boxes
    xyxy = boxes.xyxy    # [N, 4] tensor (x1, y1, x2, y2)
    conf = boxes.conf    # [N] confidence scores
    cls = boxes.cls      # [N] class indices

    # Iterate boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

#### Visualize kết quả

```python
# Plot với boxes
annotated = results[0].plot()  # numpy array (BGR)

# Lưu ảnh
cv2.imwrite('result.jpg', annotated)

# Hoặc dùng built-in save
results[0].save('result.jpg')

# Show ảnh
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.show()
```

### 4.3 Training YOLO

#### Cấu trúc dataset YOLO

```
dataset/
├── data.yaml           # Config file
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**data.yaml**:

```yaml
path: D:/datasets/motorcyclist
train: train/images
val: valid/images
test: test/images

nc: 1 # Number of classes
names:
  0: motorcyclist
```

**Label format** (img1.txt):

```
# class_id x_center y_center width height (normalized 0-1)
0 0.512 0.345 0.156 0.287
0 0.723 0.521 0.123 0.198
```

#### Training code

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8l.pt')  # Large variant

# Train
results = model.train(
    data='data/motorcyclist/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda',
    workers=2,
    project='runs/detect',
    name='motorcyclist_v1',

    # Optimizer
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,

    # Augmentation
    mosaic=1.0,
    mixup=0.5,
    degrees=10,
    translate=0.1,
    scale=0.5,

    # Other
    patience=50,  # Early stopping
    save=True,
    save_period=10,
    verbose=True
)
```

#### Resume training

```python
# Resume từ checkpoint
model = YOLO('runs/detect/motorcyclist_v1/weights/last.pt')
model.train(resume=True)
```

### 4.4 Validation & Metrics

```python
# Validate model
metrics = model.val()

# Metrics
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

### 4.5 Export Model

```python
# Export sang ONNX
model.export(format='onnx')

# Export sang TensorRT (faster inference)
model.export(format='engine')

# Export sang TFLite (mobile)
model.export(format='tflite')
```

---

## 5. Thư viện OpenCV

### 5.1 OpenCV là gì?

**OpenCV** (Open Source Computer Vision Library) là thư viện xử lý ảnh và computer vision phổ biến nhất.

**Cài đặt**:

```bash
pip install opencv-python opencv-python-headless
```

### 5.2 Đọc và hiển thị ảnh

```python
import cv2

# Đọc ảnh
img = cv2.imread('image.jpg')  # BGR format
print(img.shape)  # (H, W, 3)

# Đọc ảnh grayscale
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
print(gray.shape)  # (H, W)

# Hiển thị ảnh
cv2.imshow('Image', img)
cv2.waitKey(0)  # Chờ phím bất kỳ
cv2.destroyAllWindows()

# Lưu ảnh
cv2.imwrite('output.jpg', img)
```

### 5.3 Color Space Conversion

```python
# BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BGR to HSV (Hue, Saturation, Value)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Useful cho threshold màu sắc
# Ví dụ: detect màu đỏ
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
```

### 5.4 Vẽ Annotations

```python
# Vẽ bounding box
x1, y1, x2, y2 = 100, 100, 300, 400
cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Vẽ text
cv2.putText(
    img,
    text='Violation',
    org=(x1, y1 - 10),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.8,
    color=(0, 0, 255),
    thickness=2
)

# Vẽ circle
cv2.circle(img, center=(200, 200), radius=50, color=(255, 0, 0), thickness=-1)

# Vẽ line
cv2.line(img, pt1=(0, 0), pt2=(640, 640), color=(255, 255, 0), thickness=3)
```

### 5.5 Resize và Crop

```python
# Resize
resized = cv2.resize(img, (640, 640))  # (W, H)

# Resize giữ aspect ratio
h, w = img.shape[:2]
scale = 640 / max(h, w)
new_h, new_w = int(h * scale), int(w * scale)
resized = cv2.resize(img, (new_w, new_h))

# Crop
crop = img[y1:y2, x1:x2]  # [H, W] slicing
```

### 5.6 Video Processing

```python
# Đọc video
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    # ...

    # Hiển thị
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Ghi video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))

for frame in frames:
    out.write(frame)

out.release()
```

### 5.7 Image Preprocessing cho OCR

```python
def preprocess_license_plate(img):
    """Preprocessing cho OCR biển số"""
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur để giảm noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph
```

---

## 6. Thư viện EasyOCR

### 6.1 EasyOCR là gì?

**EasyOCR** là thư viện OCR (Optical Character Recognition) dựa trên deep learning, hỗ trợ 80+ ngôn ngữ.

**Đặc điểm**:

- Deep learning-based (PyTorch)
- Hỗ trợ tiếng Việt, tiếng Anh, ...
- Dễ sử dụng
- GPU acceleration

**Cài đặt**:

```bash
pip install easyocr
```

### 6.2 Sử dụng EasyOCR

#### Basic usage

```python
import easyocr

# Khởi tạo Reader
reader = easyocr.Reader(['en'], gpu=True)  # English

# Đọc text từ ảnh
results = reader.readtext('license_plate.jpg')

# Results format
for bbox, text, confidence in results:
    print(f"Text: {text}, Confidence: {confidence:.2f}")
    print(f"BBox: {bbox}")
```

#### Advanced usage cho biển số xe

```python
def read_license_plate(img):
    """
    Đọc biển số xe từ ảnh crop

    Args:
        img: numpy array (BGR)

    Returns:
        str: Biển số xe hoặc "Unknown"
    """
    # Khởi tạo reader (cache lại để tránh reload)
    if not hasattr(read_license_plate, 'reader'):
        read_license_plate.reader = easyocr.Reader(
            ['en'],
            gpu=True,
            verbose=False
        )

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OCR
    results = read_license_plate.reader.readtext(
        gray,
        allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ',  # Ký tự hợp lệ
        detail=1,
        paragraph=False
    )

    if not results:
        return "Unknown"

    # Lấy text có confidence cao nhất
    best = max(results, key=lambda x: x[2])  # x[2] = confidence
    text = best[1]

    # Post-processing
    text = text.replace(' ', '').replace('-', '').replace('.', '')

    return text if len(text) >= 5 else "Unknown"
```

### 6.3 Tối ưu performance

```python
# 1. Cache reader instance
class LicensePlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)

    def read(self, img):
        results = self.reader.readtext(img)
        return self._postprocess(results)

# 2. Batch processing
reader = easyocr.Reader(['en'], gpu=True)
results = reader.readtext_batched(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# 3. Giảm model size (nếu GPU nhỏ)
reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='models/')
```

---

## 7. Thư viện Gradio

### 7.1 Gradio là gì?

**Gradio** là thư viện Python để tạo web UI cho machine learning models một cách nhanh chóng.

**Đặc điểm**:

- API đơn giản
- Auto-generate UI từ function signature
- Share link (gradio.live)
- Tích hợp dễ dàng với ML models

**Cài đặt**:

```bash
pip install gradio
```

### 7.2 Hello World Example

```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

demo.launch()
```

Mở browser: `http://127.0.0.1:7860`

### 7.3 Input/Output Components

#### Common Inputs

```python
import gradio as gr

# Text input
text = gr.Textbox(label="Tên", placeholder="Nhập tên...")

# Image input
image = gr.Image(type="numpy", label="Upload ảnh")

# File input
file = gr.File(label="Upload video")

# Slider
slider = gr.Slider(minimum=0, maximum=100, value=50, label="Confidence")

# Dropdown
dropdown = gr.Dropdown(choices=["A", "B", "C"], label="Chọn")

# Checkbox
checkbox = gr.Checkbox(label="Hiển thị boxes")
```

#### Common Outputs

```python
# Text output
text_output = gr.Textbox(label="Kết quả")

# Image output
image_output = gr.Image(type="numpy", label="Ảnh đã detect")

# DataFrame (table)
dataframe = gr.Dataframe(headers=["STT", "Biển số", "Thời gian"])

# JSON
json_output = gr.JSON()
```

### 7.4 Advanced Interface

```python
def detect_violations(image, conf_threshold, show_boxes):
    """
    Detect helmet violations

    Args:
        image: numpy array
        conf_threshold: float
        show_boxes: bool

    Returns:
        annotated_image, report_table
    """
    # Detection logic here
    annotated = detect(image, conf=conf_threshold)

    report = [
        ["1", "59A-12345", "2024-01-15 10:30"],
        ["2", "51B-67890", "2024-01-15 10:35"]
    ]

    return annotated, report

# Create interface
demo = gr.Interface(
    fn=detect_violations,
    inputs=[
        gr.Image(type="numpy", label="Upload ảnh giao thông"),
        gr.Slider(0.1, 0.9, value=0.4, label="Confidence Threshold"),
        gr.Checkbox(value=True, label="Hiển thị bounding boxes")
    ],
    outputs=[
        gr.Image(type="numpy", label="Kết quả detection"),
        gr.Dataframe(headers=["STT", "Biển số", "Thời gian"], label="Báo cáo vi phạm")
    ],
    title="🛵 Helmet Violation Detection",
    description="Upload ảnh giao thông để phát hiện vi phạm không đội mũ bảo hiểm",
    examples=[
        ["examples/traffic1.jpg", 0.4, True],
        ["examples/traffic2.jpg", 0.5, True]
    ]
)

demo.launch(share=False)
```

### 7.5 Tabs Layout

```python
with gr.Blocks() as demo:
    gr.Markdown("# Helmet Violation Detection System")

    with gr.Tab("Xử lý ảnh"):
        with gr.Row():
            img_input = gr.Image()
            img_output = gr.Image()
        detect_btn = gr.Button("Detect")

    with gr.Tab("Xử lý video"):
        with gr.Row():
            vid_input = gr.Video()
            vid_output = gr.Video()
        process_btn = gr.Button("Process")

    with gr.Tab("Thống kê"):
        stats = gr.DataFrame()

demo.launch()
```

### 7.6 Progress Bar

```python
import gradio as gr
import time

def long_process(image, progress=gr.Progress()):
    progress(0, desc="Đang tải...")
    time.sleep(1)

    progress(0.3, desc="Đang detect xe máy...")
    # ... Model 1
    time.sleep(1)

    progress(0.7, desc="Đang detect mũ bảo hiểm...")
    # ... Model 2
    time.sleep(1)

    progress(1.0, desc="Hoàn thành!")
    return result

demo = gr.Interface(fn=long_process, inputs="image", outputs="image")
demo.launch()
```

---

## 📝 Tóm tắt Phần 1

**Đã học**:

1. ✅ Object Detection cơ bản (IoU, NMS, mAP)
2. ✅ YOLO architecture (YOLOv8 variants, loss functions)
3. ✅ PyTorch (tensors, autograd, GPU, save/load)
4. ✅ Ultralytics (inference, training, validation)
5. ✅ OpenCV (image processing, video, annotations)
6. ✅ EasyOCR (license plate reading)
7. ✅ Gradio (web UI development)

**Tiếp theo (Phần 2)**:

- Giải thích code chi tiết từng module
- 2-Stage detection pipeline
- Training workflow
- Deployment strategies

---

👉 **Tiếp tục đọc**: [LY_THUYET_VA_GIAI_THICH_CODE_PHAN2.md](LY_THUYET_VA_GIAI_THICH_CODE_PHAN2.md)
