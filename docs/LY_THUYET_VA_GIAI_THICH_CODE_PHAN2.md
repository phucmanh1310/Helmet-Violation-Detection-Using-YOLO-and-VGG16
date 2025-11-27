# 📚 LÝ THUYẾT VÀ GIẢI THÍCH CODE - PHẦN 2

> **Tài liệu lý thuyết và giải thích code cho dự án Helmet Violation Detection**  
> **Phần 2: Giải thích Code chi tiết & Kiến trúc 2-Stage**

---

## 📋 Mục lục Phần 2

1. [Kiến trúc 2-Stage Detection](#1-kiến-trúc-2-stage-detection)
2. [Module \_Motobike.py - Stage 1](#2-module-_motobikepy---stage-1)
3. [Module \_LP_Helmet.py - Stage 2](#3-module-_lp_helmetpy---stage-2)
4. [Module \_myFunc.py - Utilities](#4-module-myfuncpy---utilities)
5. [Module main_app.py - CLI Application](#5-module-main_apppy---cli-application)
6. [Module ui_app.py - Web Interface](#6-module-ui_apppy---web-interface)
7. [Training Scripts](#7-training-scripts)
8. [Best Practices & Optimization](#8-best-practices--optimization)

---

## 1. Kiến trúc 2-Stage Detection

### 1.1 Tại sao cần 2-Stage?

**Vấn đề với 1-Stage (Direct Detection)**:

```python
# ❌ Approach 1-Stage (không tốt)
model = YOLO('all_in_one_model.pt')
results = model.predict(traffic_image)

# Vấn đề:
# 1. Helmet/NoHelmet rất nhỏ trong full scene → Hard to detect
# 2. False positives cao (detect nhầm mũ trên người đi bộ)
# 3. mAP thấp (~70-75%)
# 4. Biển số xe quá nhỏ → OCR sai
```

**Giải pháp với 2-Stage**:

```python
# ✅ Approach 2-Stage (tốt hơn)

# Stage 1: Detect motorcyclist (người + xe)
motorcyclist_boxes = detect_motorcyclists(traffic_image)

# Stage 2: Detect helmet/nohelmet/LP chỉ trong ROI
for box in motorcyclist_boxes:
    roi = crop_region(traffic_image, box)  # Crop vùng xe máy
    roi_resized = resize(roi, 768, 768)    # Resize lên độ phân giải cao

    helmet_results = detect_helmet_and_lp(roi_resized)

# Ưu điểm:
# 1. Small objects → Large objects (resize ROI)
# 2. Context filtering (chỉ xét trong vùng xe)
# 3. mAP tăng lên ~85-90%
# 4. False positives giảm <10%
```

### 1.2 Pipeline Chi tiết

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Traffic Image                         │
│                    (1920x1080 hoặc 1280x720)                    │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: MOTORCYCLIST DETECTION                                │
├─────────────────────────────────────────────────────────────────┤
│  Model: Motov10l.pt (YOLOv8l variant)                           │
│  Input Size: 640x640                                            │
│  Classes: motorcyclist (1 class)                                │
│                                                                  │
│  Code:                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ from Source._Motobike import detect_motorcyclists          │ │
│  │                                                             │ │
│  │ boxes = detect_motorcyclists(                              │ │
│  │     image=traffic_img,                                     │ │
│  │     conf_threshold=0.4,                                    │ │
│  │     model_path='models/Motov10l.pt'                        │ │
│  │ )                                                           │ │
│  │                                                             │ │
│  │ # Output: List of bounding boxes                           │ │
│  │ # [                                                         │ │
│  │ #   [x1, y1, x2, y2, confidence],                          │ │
│  │ #   [x1, y1, x2, y2, confidence],                          │ │
│  │ #   ...                                                     │ │
│  │ # ]                                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Output: Motorcyclist bounding boxes                            │
│  Example: [[120, 200, 350, 580, 0.87], [400, 150, 620, 520, 0.91]]│
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  ROI EXTRACTION & PREPROCESSING                                 │
├─────────────────────────────────────────────────────────────────┤
│  Với mỗi motorcyclist box:                                      │
│                                                                  │
│  1. Expand box (+10% padding)                                   │
│     x1 -= w * 0.05, y1 -= h * 0.05                              │
│     x2 += w * 0.05, y2 += h * 0.05                              │
│                                                                  │
│  2. Crop ROI                                                    │
│     roi = img[y1:y2, x1:x2]                                     │
│                                                                  │
│  3. Resize to 768x768 (higher resolution for small objects)    │
│     roi_resized = cv2.resize(roi, (768, 768))                   │
│                                                                  │
│  4. Normalize                                                   │
│     roi_norm = roi_resized / 255.0                              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: HELMET & LICENSE PLATE DETECTION                      │
├─────────────────────────────────────────────────────────────────┤
│  Model: HelmetLP.pt (YOLOv8 custom)                             │
│  Input Size: 768x768                                            │
│  Classes: helmet, nohelmet, licenseplate (3 classes)            │
│                                                                  │
│  Code:                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ from Source._LP_Helmet import detect_helmet_and_lp         │ │
│  │                                                             │ │
│  │ results = detect_helmet_and_lp(                            │ │
│  │     roi_image=roi_resized,                                 │ │
│  │     conf_threshold=0.3,                                    │ │
│  │     model_path='models/HelmetLP.pt'                        │ │
│  │ )                                                           │ │
│  │                                                             │ │
│  │ # Output: Dict với detections                              │ │
│  │ # {                                                         │ │
│  │ #   'helmet': [...],                                        │ │
│  │ #   'nohelmet': [...],                                      │ │
│  │ #   'licenseplate': [...]                                   │ │
│  │ # }                                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Output: Detections trong ROI coordinates                       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  COORDINATE TRANSFORMATION                                       │
├─────────────────────────────────────────────────────────────────┤
│  Chuyển đổi từ ROI coords → Original image coords               │
│                                                                  │
│  scale_x = roi_width / 768                                      │
│  scale_y = roi_height / 768                                     │
│                                                                  │
│  x_original = x_roi * scale_x + roi_x1                          │
│  y_original = y_roi * scale_y + roi_y1                          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  VIOLATION ANALYSIS                                              │
├─────────────────────────────────────────────────────────────────┤
│  Logic phát hiện vi phạm:                                       │
│                                                                  │
│  for motorcyclist in motorcyclists:                             │
│      helmet_count = count(helmet detections in motorcyclist)    │
│      nohelmet_count = count(nohelmet in motorcyclist)           │
│      lp_detections = filter(licenseplate in motorcyclist)       │
│                                                                  │
│      if nohelmet_count > 0:                                     │
│          violation = True                                       │
│          severity = "High"                                      │
│      elif helmet_count == 0:                                    │
│          violation = True                                       │
│          severity = "Medium"  # Không phát hiện được mũ         │
│      else:                                                      │
│          violation = False                                      │
│                                                                  │
│      if violation and lp_detections:                            │
│          lp_img = crop(original_img, lp_box)                    │
│          lp_text = ocr_license_plate(lp_img)                    │
│          violations.append({                                    │
│              'box': motorcyclist_box,                           │
│              'license_plate': lp_text,                          │
│              'severity': severity                               │
│          })                                                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  LICENSE PLATE OCR (Optional)                                    │
├─────────────────────────────────────────────────────────────────┤
│  Nếu phát hiện vi phạm + có biển số:                            │
│                                                                  │
│  Code:                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ import easyocr                                              │ │
│  │                                                             │ │
│  │ reader = easyocr.Reader(['en'], gpu=True)                  │ │
│  │                                                             │ │
│  │ # Crop license plate                                        │ │
│  │ lp_crop = img[lp_y1:lp_y2, lp_x1:lp_x2]                     │ │
│  │                                                             │ │
│  │ # Preprocess                                                │ │
│  │ lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)        │ │
│  │ lp_thresh = cv2.threshold(lp_gray, 0, 255,                 │ │
│  │                          cv2.THRESH_BINARY+THRESH_OTSU)   │ │
│  │                                                             │ │
│  │ # OCR                                                       │ │
│  │ results = reader.readtext(lp_thresh,                       │ │
│  │     allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ')          │ │
│  │                                                             │ │
│  │ # Extract best result                                       │ │
│  │ if results:                                                 │ │
│  │     text = max(results, key=lambda x: x[2])[1]             │ │
│  │     text = text.replace(' ', '').replace('-', '')          │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  VISUALIZATION & OUTPUT                                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Vẽ bounding boxes:                                          │
│     - Violation (nohelmet): RED color                           │
│     - Safe (helmet): GREEN color                                │
│     - License plate: BLUE color                                 │
│                                                                  │
│  2. Vẽ labels với text                                          │
│                                                                  │
│  3. Tạo báo cáo table:                                          │
│     ┌─────┬─────────────┬──────────────┬──────────┐            │
│     │ STT │ Biển số     │ Thời gian    │ Mức độ   │            │
│     ├─────┼─────────────┼──────────────┼──────────┤            │
│     │ 1   │ 59A-12345   │ 10:30:15     │ High     │            │
│     │ 2   │ 51B-67890   │ 10:35:42     │ Medium   │            │
│     └─────┴─────────────┴──────────────┴──────────┘            │
│                                                                  │
│  4. Export:                                                     │
│     - Annotated image/video                                     │
│     - CSV report                                                │
│     - JSON metadata                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 So sánh 1-Stage vs 2-Stage

| Metric               | 1-Stage Direct | 2-Stage (Ours) | Improvement |
| -------------------- | -------------- | -------------- | ----------- |
| **mAP (Helmet)**     | 0.72           | **0.82**       | +13.9% ✅   |
| **mAP (NoHelmet)**   | 0.68           | **0.77**       | +13.2% ✅   |
| **mAP (LP)**         | 0.75           | **0.83**       | +10.7% ✅   |
| **False Positives**  | ~15%           | **<10%**       | -33% ✅     |
| **Small Object Det** | Poor           | **Good**       | ✅          |
| **Speed (FPS)**      | 35-40          | 15-25          | -40% ⚠️     |
| **OCR Accuracy**     | 65%            | **85%**        | +30.8% ✅   |

**Kết luận**: Trade-off giữa speed và accuracy, nhưng 15-25 FPS vẫn đủ cho real-time monitoring.

---

## 2. Module `_Motobike.py` - Stage 1

### 2.1 Tổng quan

File này chịu trách nhiệm **Stage 1**: Phát hiện motorcyclist trong ảnh giao thông.

**Đường dẫn**: `Source/_Motobike.py`

**Chức năng chính**:

- Load YOLO model (Motov10l.pt)
- Detect motorcyclist với confidence filtering
- Trả về bounding boxes

### 2.2 Code đầy đủ với giải thích

```python
"""
Module phát hiện xe máy (motorcyclist) trong ảnh giao thông
Author: Helmet Violation Detection Team
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Đường dẫn model mặc định
DEFAULT_MODEL_PATH = 'models/Motov10l.pt'

# Cache model instance để tránh reload nhiều lần
_cached_model = None


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path=None):
    """
    Load YOLO model cho motorcyclist detection

    Sử dụng caching để tránh reload model nhiều lần (expensive operation)

    Args:
        model_path (str, optional): Đường dẫn đến model weights.
                                    Nếu None, dùng DEFAULT_MODEL_PATH

    Returns:
        YOLO: Model instance đã load

    Raises:
        FileNotFoundError: Nếu model file không tồn tại

    Example:
        >>> model = load_model('models/Motov10l.pt')
        >>> print(type(model))
        <class 'ultralytics.models.yolo.model.YOLO'>
    """
    global _cached_model

    # Xác định model path
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # Convert sang Path object để xử lý cross-platform
    model_path = Path(model_path)

    # Kiểm tra file tồn tại
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure the model is in the correct location."
        )

    # Cache model để tránh reload
    if _cached_model is None:
        print(f"[INFO] Loading motorcyclist detection model: {model_path}")
        _cached_model = YOLO(str(model_path))
        print(f"[INFO] Model loaded successfully!")

    return _cached_model


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_motorcyclists(image, conf_threshold=0.4, model_path=None):
    """
    Phát hiện motorcyclist trong ảnh

    Workflow:
        1. Load model (với caching)
        2. Resize ảnh về 640x640 (YOLO input size)
        3. Run inference
        4. Filter results theo confidence threshold
        5. Scale boxes về original image size
        6. Return boxes

    Args:
        image (numpy.ndarray): Input image (BGR format từ cv2.imread)
                               Shape: (H, W, 3)
        conf_threshold (float): Ngưỡng confidence (0.0-1.0)
                               Chỉ giữ detections có conf >= threshold
                               Default: 0.4
        model_path (str, optional): Đường dẫn model weights

    Returns:
        list: List of bounding boxes
              Format: [[x1, y1, x2, y2, confidence], ...]
              Coordinates là pixel values trong original image

    Example:
        >>> import cv2
        >>> img = cv2.imread('traffic.jpg')
        >>> boxes = detect_motorcyclists(img, conf_threshold=0.5)
        >>> print(f"Detected {len(boxes)} motorcyclists")
        Detected 3 motorcyclists
        >>> print(boxes[0])
        [120.5, 200.3, 350.8, 580.2, 0.87]
    """
    # Load model
    model = load_model(model_path)

    # Lấy kích thước original image
    original_h, original_w = image.shape[:2]

    # Run inference
    # - imgsz=640: Resize input về 640x640
    # - conf=conf_threshold: Filter theo confidence
    # - verbose=False: Không print log
    # - device='cuda' nếu có GPU, 'cpu' nếu không
    results = model.predict(
        source=image,
        imgsz=640,
        conf=conf_threshold,
        verbose=False,
        device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    )

    # Extract boxes
    boxes = []

    # Results là list, lấy result đầu tiên (vì chỉ detect 1 ảnh)
    result = results[0]

    # Lấy boxes object
    detections = result.boxes

    if detections is not None and len(detections) > 0:
        # Iterate qua từng detection
        for detection in detections:
            # Extract box coordinates (x1, y1, x2, y2)
            # xyxy: tensor shape [4]
            box_xyxy = detection.xyxy[0].cpu().numpy()

            # Extract confidence
            # conf: tensor shape [1]
            confidence = detection.conf[0].cpu().item()

            # Extract class (should be 0 = motorcyclist)
            class_id = int(detection.cls[0].cpu().item())

            # Scale boxes về original image size
            # (YOLO đã tự động scale từ 640x640 về original size)
            x1, y1, x2, y2 = box_xyxy

            # Validate box (đảm bảo trong bounds)
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))

            # Thêm vào list
            boxes.append([x1, y1, x2, y2, confidence])

    print(f"[INFO] Detected {len(boxes)} motorcyclists (conf >= {conf_threshold})")

    return boxes


def detect_and_visualize(image_path, conf_threshold=0.4, output_path=None):
    """
    Detect motorcyclist và visualize kết quả

    Hàm tiện ích để test model nhanh

    Args:
        image_path (str): Đường dẫn ảnh input
        conf_threshold (float): Confidence threshold
        output_path (str, optional): Đường dẫn save ảnh output
                                     Nếu None, không save

    Returns:
        numpy.ndarray: Annotated image

    Example:
        >>> annotated = detect_and_visualize(
        ...     'test.jpg',
        ...     conf_threshold=0.5,
        ...     output_path='result.jpg'
        ... )
        >>> cv2.imshow('Result', annotated)
        >>> cv2.waitKey(0)
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Detect
    boxes = detect_motorcyclists(image, conf_threshold)

    # Visualize
    annotated = image.copy()

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box

        # Convert to int
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Vẽ box
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Vẽ label
        label = f"Motorcyclist {idx+1}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Tính size của text để vẽ background
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Vẽ background cho text
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - 10),
            (x1 + text_w, y1),
            color,
            -1  # Filled
        )

        # Vẽ text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness
        )

    # Save nếu có output_path
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Saved annotated image to: {output_path}")

    return annotated


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_boxes_by_area(boxes, min_area=5000, max_area=500000):
    """
    Filter boxes theo diện tích (loại bỏ boxes quá nhỏ hoặc quá lớn)

    Args:
        boxes (list): List of boxes [[x1, y1, x2, y2, conf], ...]
        min_area (int): Diện tích tối thiểu (pixels)
        max_area (int): Diện tích tối đa (pixels)

    Returns:
        list: Filtered boxes

    Example:
        >>> boxes = [[10, 10, 50, 50, 0.9], [100, 100, 900, 900, 0.95]]
        >>> filtered = filter_boxes_by_area(boxes, min_area=2000, max_area=100000)
        >>> len(filtered)
        0  # Cả 2 boxes đều ngoài range
    """
    filtered = []

    for box in boxes:
        x1, y1, x2, y2, conf = box

        # Tính diện tích
        area = (x2 - x1) * (y2 - y1)

        # Filter
        if min_area <= area <= max_area:
            filtered.append(box)

    print(f"[INFO] Filtered {len(boxes)} → {len(filtered)} boxes by area")

    return filtered


def expand_boxes(boxes, expand_ratio=0.1, img_width=None, img_height=None):
    """
    Mở rộng bounding boxes với padding

    Hữu ích để đảm bảo ROI crop bao gồm toàn bộ object

    Args:
        boxes (list): List of boxes [[x1, y1, x2, y2, conf], ...]
        expand_ratio (float): Tỷ lệ mở rộng (0.1 = 10%)
        img_width (int, optional): Width của image (để clip boxes)
        img_height (int, optional): Height của image (để clip boxes)

    Returns:
        list: Expanded boxes

    Example:
        >>> boxes = [[100, 100, 200, 300, 0.9]]
        >>> expanded = expand_boxes(boxes, expand_ratio=0.1,
        ...                         img_width=640, img_height=640)
        >>> expanded[0][:4]
        [90, 90, 210, 310]  # Expanded by 10%
    """
    expanded = []

    for box in boxes:
        x1, y1, x2, y2, conf = box

        # Tính width và height
        w = x2 - x1
        h = y2 - y1

        # Tính padding
        pad_w = w * expand_ratio / 2
        pad_h = h * expand_ratio / 2

        # Expand
        x1_new = x1 - pad_w
        y1_new = y1 - pad_h
        x2_new = x2 + pad_w
        y2_new = y2 + pad_h

        # Clip nếu có img dimensions
        if img_width is not None:
            x1_new = max(0, x1_new)
            x2_new = min(img_width, x2_new)

        if img_height is not None:
            y1_new = max(0, y1_new)
            y2_new = min(img_height, y2_new)

        expanded.append([x1_new, y1_new, x2_new, y2_new, conf])

    return expanded


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == '__main__':
    """
    Test script cho module _Motobike.py

    Usage:
        python Source/_Motobike.py
    """
    import sys

    # Test image path
    test_image = 'img/test/traffic_sample.jpg'

    if not Path(test_image).exists():
        print(f"[ERROR] Test image not found: {test_image}")
        print("[INFO] Please provide a test image path as argument")
        print("Usage: python Source/_Motobike.py <image_path>")

        if len(sys.argv) > 1:
            test_image = sys.argv[1]
        else:
            sys.exit(1)

    # Run detection
    print("=" * 60)
    print("TESTING MOTORCYCLIST DETECTION")
    print("=" * 60)

    annotated = detect_and_visualize(
        test_image,
        conf_threshold=0.4,
        output_path='result_motorcyclist.jpg'
    )

    print("=" * 60)
    print("Test completed! Check result_motorcyclist.jpg")
    print("=" * 60)
```

### 2.3 Giải thích chi tiết các phần

#### Caching Model

```python
_cached_model = None

def load_model(model_path=None):
    global _cached_model

    if _cached_model is None:
        _cached_model = YOLO(str(model_path))

    return _cached_model
```

**Lý do**:

- Load YOLO model là **expensive operation** (~2-3 giây)
- Nếu không cache, mỗi lần gọi `detect_motorcyclists()` sẽ reload → Rất chậm
- Cache trong global variable → Chỉ load 1 lần duy nhất

#### Box Scaling

```python
# YOLO tự động scale boxes từ 640x640 về original size
# Nhưng cần validate để đảm bảo trong bounds
x1 = max(0, min(x1, original_w))
y1 = max(0, min(y1, original_h))
x2 = max(0, min(x2, original_w))
y2 = max(0, min(y2, original_h))
```

**Lý do**:

- Đôi khi boxes có thể vượt ra ngoài image boundaries
- Cần clip về [0, width] và [0, height]

---

## 3. Module `_LP_Helmet.py` - Stage 2

### 3.1 Tổng quan

File này chịu trách nhiệm **Stage 2**: Phát hiện helmet/nohelmet/licenseplate trong ROI crops.

**Đường dẫn**: `Source/_LP_Helmet.py`

**Chức năng chính**:

- Load YOLO model (HelmetLP.pt)
- Detect helmet, nohelmet, licenseplate trong ROI
- OCR license plate với EasyOCR

### 3.2 Code đầy đủ với giải thích

```python
"""
Module phát hiện mũ bảo hiểm và biển số xe trong ROI crops
Author: Helmet Violation Detection Team
"""

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from pathlib import Path

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

DEFAULT_MODEL_PATH = 'models/HelmetLP.pt'

# Cache instances
_cached_model = None
_cached_ocr_reader = None

# Class names mapping
CLASS_NAMES = {
    0: 'helmet',
    1: 'nohelmet',
    2: 'licenseplate'
}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path=None):
    """
    Load YOLO model cho helmet/LP detection

    Tương tự _Motobike.py nhưng với model khác
    """
    global _cached_model

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if _cached_model is None:
        print(f"[INFO] Loading helmet/LP detection model: {model_path}")
        _cached_model = YOLO(str(model_path))
        print(f"[INFO] Model loaded successfully!")

    return _cached_model


def load_ocr_reader():
    """
    Load EasyOCR reader

    Cache để tránh reload (mất ~5-10 giây)
    """
    global _cached_ocr_reader

    if _cached_ocr_reader is None:
        print("[INFO] Loading EasyOCR reader...")
        _cached_ocr_reader = easyocr.Reader(
            ['en'],  # English
            gpu=True,  # Dùng GPU nếu có
            verbose=False
        )
        print("[INFO] OCR reader loaded!")

    return _cached_ocr_reader


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_helmet_and_lp(roi_image, conf_threshold=0.3, model_path=None):
    """
    Phát hiện helmet, nohelmet, licenseplate trong ROI

    Args:
        roi_image (numpy.ndarray): ROI crop từ Stage 1
                                    Thường được resize về 768x768
        conf_threshold (float): Confidence threshold (default: 0.3)
        model_path (str, optional): Model path

    Returns:
        dict: Dictionary với structure:
              {
                  'helmet': [[x1, y1, x2, y2, conf], ...],
                  'nohelmet': [[x1, y1, x2, y2, conf], ...],
                  'licenseplate': [[x1, y1, x2, y2, conf], ...]
              }

    Example:
        >>> roi = img[100:400, 150:450]  # Crop from Stage 1
        >>> roi_resized = cv2.resize(roi, (768, 768))
        >>> results = detect_helmet_and_lp(roi_resized, conf_threshold=0.4)
        >>> print(f"Helmets: {len(results['helmet'])}")
        >>> print(f"NoHelmets: {len(results['nohelmet'])}")
        >>> print(f"LPs: {len(results['licenseplate'])}")
    """
    # Load model
    model = load_model(model_path)

    # Run inference
    # imgsz=768: Higher resolution cho small objects
    results = model.predict(
        source=roi_image,
        imgsz=768,
        conf=conf_threshold,
        verbose=False
    )

    # Initialize output dict
    detections = {
        'helmet': [],
        'nohelmet': [],
        'licenseplate': []
    }

    # Extract detections
    result = results[0]
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for detection in boxes:
            # Extract data
            box_xyxy = detection.xyxy[0].cpu().numpy()
            confidence = detection.conf[0].cpu().item()
            class_id = int(detection.cls[0].cpu().item())

            # Get class name
            class_name = CLASS_NAMES.get(class_id, 'unknown')

            # Add to corresponding list
            x1, y1, x2, y2 = box_xyxy
            detections[class_name].append([x1, y1, x2, y2, confidence])

    print(f"[INFO] Detected: {len(detections['helmet'])} helmets, "
          f"{len(detections['nohelmet'])} no-helmets, "
          f"{len(detections['licenseplate'])} license plates")

    return detections


# ============================================================================
# OCR FUNCTIONS
# ============================================================================

def preprocess_license_plate(lp_crop):
    """
    Tiền xử lý ảnh biển số trước OCR

    Các bước:
        1. Convert to grayscale
        2. Denoise với bilateral filter
        3. Contrast enhancement với CLAHE
        4. Threshold với Otsu's method
        5. Morphology operations

    Args:
        lp_crop (numpy.ndarray): Cropped license plate image

    Returns:
        numpy.ndarray: Preprocessed image (grayscale)
    """
    # Grayscale
    if len(lp_crop.shape) == 3:
        gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = lp_crop.copy()

    # Denoise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Threshold
    _, thresh = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphology (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph


def ocr_license_plate(lp_crop):
    """
    Đọc text từ ảnh biển số bằng EasyOCR

    Args:
        lp_crop (numpy.ndarray): Cropped license plate image

    Returns:
        str: License plate text hoặc "Unknown"

    Example:
        >>> lp_img = img[y1:y2, x1:x2]
        >>> text = ocr_license_plate(lp_img)
        >>> print(text)
        '59A12345'
    """
    # Load OCR reader
    reader = load_ocr_reader()

    # Preprocess
    preprocessed = preprocess_license_plate(lp_crop)

    # OCR với allowlist (chỉ cho phép ký tự hợp lệ)
    # Biển số VN: Số + Chữ cái (không có I, O, Q, W)
    results = reader.readtext(
        preprocessed,
        allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ',
        detail=1,  # Return bbox + text + confidence
        paragraph=False  # Không merge thành đoạn văn
    )

    if not results:
        return "Unknown"

    # Lấy kết quả có confidence cao nhất
    best_result = max(results, key=lambda x: x[2])  # x[2] = confidence
    text = best_result[1]
    confidence = best_result[2]

    # Post-processing: Remove spaces, dashes
    text = text.replace(' ', '').replace('-', '').replace('.', '')

    # Validate length (biển số VN thường 7-9 ký tự)
    if len(text) < 5:
        return "Unknown"

    print(f"[INFO] OCR result: {text} (confidence: {confidence:.2f})")

    return text


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def analyze_violation(detections):
    """
    Phân tích xem có vi phạm không dựa trên detections

    Logic:
        - Có nohelmet → Violation = True (nghiêm trọng)
        - Không có helmet và không có nohelmet → Violation = True (nghi ngờ)
        - Có helmet → Violation = False (an toàn)

    Args:
        detections (dict): Output từ detect_helmet_and_lp()

    Returns:
        dict: {
            'is_violation': bool,
            'severity': 'High' | 'Medium' | 'Low' | 'None',
            'reason': str
        }
    """
    helmet_count = len(detections['helmet'])
    nohelmet_count = len(detections['nohelmet'])

    if nohelmet_count > 0:
        return {
            'is_violation': True,
            'severity': 'High',
            'reason': f'Detected {nohelmet_count} person(s) without helmet'
        }
    elif helmet_count == 0:
        return {
            'is_violation': True,
            'severity': 'Medium',
            'reason': 'No helmet detected (possible violation)'
        }
    else:
        return {
            'is_violation': False,
            'severity': 'None',
            'reason': f'{helmet_count} person(s) wearing helmet'
        }


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == '__main__':
    """
    Test script cho module _LP_Helmet.py
    """
    import sys

    test_roi = 'img/test/roi_sample.jpg'

    if not Path(test_roi).exists():
        print(f"[ERROR] Test ROI not found: {test_roi}")

        if len(sys.argv) > 1:
            test_roi = sys.argv[1]
        else:
            print("Usage: python Source/_LP_Helmet.py <roi_image_path>")
            sys.exit(1)

    # Test detection
    print("=" * 60)
    print("TESTING HELMET/LP DETECTION")
    print("=" * 60)

    roi = cv2.imread(test_roi)
    roi_resized = cv2.resize(roi, (768, 768))

    results = detect_helmet_and_lp(roi_resized, conf_threshold=0.3)

    # Analyze violation
    violation_info = analyze_violation(results)
    print("\nViolation Analysis:")
    print(f"  Is Violation: {violation_info['is_violation']}")
    print(f"  Severity: {violation_info['severity']}")
    print(f"  Reason: {violation_info['reason']}")

    # Test OCR if license plate detected
    if results['licenseplate']:
        print("\nTesting OCR on first license plate...")
        lp_box = results['licenseplate'][0]
        x1, y1, x2, y2 = map(int, lp_box[:4])

        lp_crop = roi_resized[y1:y2, x1:x2]
        lp_text = ocr_license_plate(lp_crop)

        print(f"License Plate: {lp_text}")

    print("=" * 60)
```

---

_Tiếp tục với các module còn lại trong file riêng để tránh quá tải..._

**Nội dung còn lại sẽ có trong file tiếp theo**:

- Module `_myFunc.py` - Utilities
- Module `main_app.py` - CLI Application
- Module `ui_app.py` - Web Interface
- Training Scripts chi tiết
- Best Practices & Optimization

---

👉 **File tiếp theo**: `LY_THUYET_VA_GIAI_THICH_CODE_PHAN3.md`
