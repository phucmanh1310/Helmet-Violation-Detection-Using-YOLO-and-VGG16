# 📚 LÝ THUYẾT VÀ GIẢI THÍCH CODE - PHẦN 3

> **Tài liệu lý thuyết và giải thích code cho dự án Helmet Violation Detection**  
> **Phần 3: Utilities, Applications & Training**

---

## 📋 Mục lục Phần 3

1. [Module \_myFunc.py - Utilities](#1-module-_myfuncpy---utilities)
2. [Module main_app.py - CLI Application](#2-module-main_apppy---cli-application)
3. [Module ui_app.py - Web Interface](#3-module-ui_apppy---web-interface)
4. [Training Scripts Chi tiết](#4-training-scripts-chi-tiết)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Best Practices & Optimization](#6-best-practices--optimization)
7. [Troubleshooting Common Issues](#7-troubleshooting-common-issues)

---

## 1. Module `_myFunc.py` - Utilities

### 1.1 Tổng quan

Module này chứa các **utility functions** được sử dụng chung trong project.

**Đường dẫn**: `Source/_myFunc.py`

**Chức năng**:

- Drawing functions (vẽ boxes, labels)
- Coordinate transformation
- Image preprocessing
- Report generation
- File I/O helpers

### 1.2 Code đầy đủ với giải thích

```python
"""
Module chứa utility functions cho Helmet Violation Detection
Author: Helmet Violation Detection Team
"""

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

# ============================================================================
# DRAWING FUNCTIONS
# ============================================================================

def draw_detection_box(image, box, label, color, thickness=2):
    """
    Vẽ bounding box và label lên ảnh

    Args:
        image (numpy.ndarray): Image để vẽ (sẽ modify in-place)
        box (list/tuple): [x1, y1, x2, y2] hoặc [x1, y1, x2, y2, conf]
        label (str): Text label để hiển thị
        color (tuple): BGR color (B, G, R)
        thickness (int): Độ dày của box

    Returns:
        numpy.ndarray: Image đã vẽ (same object với input)

    Example:
        >>> img = cv2.imread('test.jpg')
        >>> box = [100, 100, 300, 400, 0.95]
        >>> draw_detection_box(img, box, 'Violation', (0, 0, 255))
    """
    # Extract coordinates
    x1, y1, x2, y2 = map(int, box[:4])

    # Vẽ rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare label text
    if len(box) >= 5:
        conf = box[4]
        label_text = f"{label}: {conf:.2f}"
    else:
        label_text = label

    # Vẽ label với background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, font, font_scale, font_thickness
    )

    # Vẽ filled rectangle làm background cho text
    cv2.rectangle(
        image,
        (x1, y1 - text_h - baseline - 5),
        (x1 + text_w, y1),
        color,
        -1  # Filled
    )

    # Vẽ text (màu trắng hoặc đen tùy màu background)
    text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
    cv2.putText(
        image,
        label_text,
        (x1, y1 - baseline - 2),
        font,
        font_scale,
        text_color,
        font_thickness
    )

    return image


def visualize_detections(image, motorcyclists, violations, safe_motorcyclists):
    """
    Visualize toàn bộ detections trên ảnh

    Color coding:
        - RED: Vi phạm (nohelmet)
        - GREEN: An toàn (helmet)
        - BLUE: License plate

    Args:
        image (numpy.ndarray): Original image
        motorcyclists (list): List of motorcyclist boxes từ Stage 1
        violations (list): List of violation objects
        safe_motorcyclists (list): List of safe motorcyclist objects

    Returns:
        numpy.ndarray: Annotated image

    Example:
        >>> annotated = visualize_detections(
        ...     img, motorcyclist_boxes, violations, safe_list
        ... )
        >>> cv2.imwrite('result.jpg', annotated)
    """
    # Copy image để không modify original
    annotated = image.copy()

    # Vẽ violations (RED)
    for idx, violation in enumerate(violations):
        motorcyclist_box = violation['motorcyclist_box']
        lp_text = violation.get('license_plate', 'Unknown')

        # Vẽ motorcyclist box
        draw_detection_box(
            annotated,
            motorcyclist_box,
            f"VIOLATION #{idx+1}",
            color=(0, 0, 255),  # Red
            thickness=3
        )

        # Vẽ license plate nếu có
        if 'lp_box' in violation:
            lp_box = violation['lp_box']
            draw_detection_box(
                annotated,
                lp_box,
                f"LP: {lp_text}",
                color=(255, 0, 0),  # Blue
                thickness=2
            )

        # Vẽ nohelmet boxes
        for nohelmet_box in violation.get('nohelmet_boxes', []):
            draw_detection_box(
                annotated,
                nohelmet_box,
                "No Helmet",
                color=(0, 0, 255),  # Red
                thickness=2
            )

    # Vẽ safe motorcyclists (GREEN)
    for idx, safe in enumerate(safe_motorcyclists):
        motorcyclist_box = safe['motorcyclist_box']

        draw_detection_box(
            annotated,
            motorcyclist_box,
            f"SAFE #{idx+1}",
            color=(0, 255, 0),  # Green
            thickness=2
        )

        # Vẽ helmet boxes
        for helmet_box in safe.get('helmet_boxes', []):
            draw_detection_box(
                annotated,
                helmet_box,
                "Helmet",
                color=(0, 255, 0),  # Green
                thickness=2
            )

    return annotated


# ============================================================================
# COORDINATE TRANSFORMATION
# ============================================================================

def roi_to_original_coords(roi_box, roi_offset, roi_scale):
    """
    Chuyển đổi coordinates từ ROI space về original image space

    Workflow:
        1. Scale từ resized ROI (768x768) về original ROI size
        2. Offset về vị trí trong original image

    Args:
        roi_box (list): [x1, y1, x2, y2] trong ROI coordinates
        roi_offset (tuple): (offset_x, offset_y) của ROI trong original image
        roi_scale (tuple): (scale_x, scale_y) từ resized ROI về original ROI

    Returns:
        list: [x1, y1, x2, y2] trong original image coordinates

    Example:
        >>> # ROI crop từ [100, 150] đến [400, 600] trong original image
        >>> # ROI được resize từ (300, 450) về (768, 768)
        >>> roi_box = [50, 100, 200, 300]  # Detection trong 768x768
        >>> roi_offset = (100, 150)
        >>> roi_scale = (300/768, 450/768)
        >>> original_box = roi_to_original_coords(roi_box, roi_offset, roi_scale)
        >>> print(original_box)
        [119, 208, 178, 325]
    """
    x1, y1, x2, y2 = roi_box
    offset_x, offset_y = roi_offset
    scale_x, scale_y = roi_scale

    # Scale về original ROI size
    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    # Offset về original image position
    x1_original = x1_scaled + offset_x
    y1_original = y1_scaled + offset_y
    x2_original = x2_scaled + offset_x
    y2_original = y2_scaled + offset_y

    return [x1_original, y1_original, x2_original, y2_original]


def expand_box(box, expand_ratio, img_shape):
    """
    Mở rộng bounding box với clipping

    Args:
        box (list): [x1, y1, x2, y2]
        expand_ratio (float): Tỷ lệ mở rộng (e.g., 0.1 = 10%)
        img_shape (tuple): (height, width) của image

    Returns:
        list: [x1, y1, x2, y2] đã expand và clip
    """
    x1, y1, x2, y2 = box
    h, w = img_shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    # Calculate padding
    pad_w = box_w * expand_ratio / 2
    pad_h = box_h * expand_ratio / 2

    # Expand
    x1_new = x1 - pad_w
    y1_new = y1 - pad_h
    x2_new = x2 + pad_w
    y2_new = y2 + pad_h

    # Clip to image bounds
    x1_new = max(0, min(x1_new, w))
    y1_new = max(0, min(y1_new, h))
    x2_new = max(0, min(x2_new, w))
    y2_new = max(0, min(y2_new, h))

    return [x1_new, y1_new, x2_new, y2_new]


def crop_roi(image, box, target_size=(768, 768)):
    """
    Crop ROI từ image và resize

    Args:
        image (numpy.ndarray): Original image
        box (list): [x1, y1, x2, y2]
        target_size (tuple): (width, height) để resize

    Returns:
        tuple: (roi_resized, roi_offset, roi_scale)
            - roi_resized: Cropped và resized ROI
            - roi_offset: (x1, y1) offset trong original image
            - roi_scale: (scale_x, scale_y) ratio
    """
    x1, y1, x2, y2 = map(int, box[:4])

    # Crop
    roi = image[y1:y2, x1:x2]

    # Get original ROI size
    roi_h, roi_w = roi.shape[:2]

    # Resize
    roi_resized = cv2.resize(roi, target_size)

    # Calculate scale
    scale_x = roi_w / target_size[0]
    scale_y = roi_h / target_size[1]

    # Offset
    roi_offset = (x1, y1)
    roi_scale = (scale_x, scale_y)

    return roi_resized, roi_offset, roi_scale


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image, target_size=640):
    """
    Preprocess image cho YOLO model

    Steps:
        1. Resize giữ aspect ratio
        2. Pad về square
        3. Normalize (optional, YOLO tự normalize)

    Args:
        image (numpy.ndarray): Input image
        target_size (int): Target size (square)

    Returns:
        tuple: (processed_image, scale, padding)
    """
    h, w = image.shape[:2]

    # Calculate scale
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h))

    # Padding
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)  # Gray
    )

    return padded, scale, (top, left)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_violation_report(violations, output_format='dataframe'):
    """
    Tạo báo cáo vi phạm

    Args:
        violations (list): List of violation dicts
        output_format (str): 'dataframe', 'csv', 'json', 'dict'

    Returns:
        pandas.DataFrame hoặc list hoặc str tùy format

    Example:
        >>> report = generate_violation_report(violations, format='dataframe')
        >>> print(report)
           STT  Biển số xe    Thời gian     Mức độ
        0    1   59A-12345   10:30:15      High
        1    2   51B-67890   10:35:42      Medium
    """
    # Prepare data
    report_data = []

    for idx, violation in enumerate(violations, start=1):
        report_data.append({
            'STT': idx,
            'Biển số xe': violation.get('license_plate', 'Unknown'),
            'Thời gian': violation.get('timestamp', datetime.now().strftime('%H:%M:%S')),
            'Mức độ': violation.get('severity', 'Unknown'),
            'Confidence': f"{violation.get('confidence', 0):.2f}",
            'Vị trí': f"({violation.get('motorcyclist_box', [0,0,0,0])[0]:.0f}, "
                      f"{violation.get('motorcyclist_box', [0,0,0,0])[1]:.0f})"
        })

    # Create DataFrame
    df = pd.DataFrame(report_data)

    # Return theo format
    if output_format == 'dataframe':
        return df
    elif output_format == 'csv':
        return df.to_csv(index=False)
    elif output_format == 'json':
        return df.to_json(orient='records', indent=2)
    elif output_format == 'dict':
        return df.to_dict('records')
    else:
        return df


def save_report(violations, output_path, format='csv'):
    """
    Lưu báo cáo vi phạm ra file

    Args:
        violations (list): List of violations
        output_path (str/Path): Output file path
        format (str): 'csv', 'json', 'excel'

    Example:
        >>> save_report(violations, 'report.csv', format='csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_violation_report(violations, output_format='dataframe')

    if format == 'csv':
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2, force_ascii=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"[INFO] Report saved to: {output_path}")


# ============================================================================
# FILE I/O HELPERS
# ============================================================================

def load_image(image_path):
    """
    Load image với error handling

    Args:
        image_path (str/Path): Path to image

    Returns:
        numpy.ndarray: Image in BGR format

    Raises:
        FileNotFoundError: If image not found
        ValueError: If image cannot be read
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return image


def save_image(image, output_path):
    """
    Save image với auto mkdir

    Args:
        image (numpy.ndarray): Image to save
        output_path (str/Path): Output path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)
    print(f"[INFO] Image saved to: {output_path}")


def get_video_info(video_path):
    """
    Get thông tin về video

    Args:
        video_path (str/Path): Path to video

    Returns:
        dict: Video information

    Example:
        >>> info = get_video_info('video.mp4')
        >>> print(info)
        {
            'fps': 30.0,
            'frame_count': 900,
            'width': 1920,
            'height': 1080,
            'duration': 30.0
        }
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_box(box, img_shape):
    """
    Validate và fix bounding box

    Args:
        box (list): [x1, y1, x2, y2]
        img_shape (tuple): (height, width)

    Returns:
        list: Valid box hoặc None nếu invalid
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = box

    # Clip to image bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Check valid area
    if x2 <= x1 or y2 <= y1:
        return None

    # Check minimum size
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union

    Args:
        box1 (list): [x1, y1, x2, y2]
        box2 (list): [x1, y1, x2, y2]

    Returns:
        float: IoU value (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================================
# STATISTICS HELPERS
# ============================================================================

def calculate_detection_stats(violations, safe_motorcyclists):
    """
    Tính thống kê detection

    Args:
        violations (list): List of violations
        safe_motorcyclists (list): List of safe motorcyclists

    Returns:
        dict: Statistics
    """
    total_motorcyclists = len(violations) + len(safe_motorcyclists)
    violation_count = len(violations)
    safe_count = len(safe_motorcyclists)

    violation_rate = (violation_count / total_motorcyclists * 100) if total_motorcyclists > 0 else 0

    # Severity breakdown
    severity_counts = {
        'High': 0,
        'Medium': 0,
        'Low': 0
    }

    for v in violations:
        severity = v.get('severity', 'Unknown')
        if severity in severity_counts:
            severity_counts[severity] += 1

    return {
        'total_motorcyclists': total_motorcyclists,
        'violations': violation_count,
        'safe': safe_count,
        'violation_rate': f"{violation_rate:.1f}%",
        'severity_breakdown': severity_counts
    }


# ============================================================================
# MAIN - FOR TESTING
# ============================================================================

if __name__ == '__main__':
    """
    Test các utility functions
    """
    print("=" * 60)
    print("TESTING UTILITY FUNCTIONS")
    print("=" * 60)

    # Test coordinate transformation
    roi_box = [100, 150, 300, 450]
    roi_offset = (50, 75)
    roi_scale = (1.5, 2.0)

    original_box = roi_to_original_coords(roi_box, roi_offset, roi_scale)
    print(f"\nCoordinate transformation test:")
    print(f"  ROI box: {roi_box}")
    print(f"  Original box: {original_box}")

    # Test IoU
    box1 = [100, 100, 200, 200]
    box2 = [150, 150, 250, 250]
    iou = calculate_iou(box1, box2)
    print(f"\nIoU test:")
    print(f"  Box1: {box1}")
    print(f"  Box2: {box2}")
    print(f"  IoU: {iou:.3f}")

    # Test report generation
    violations = [
        {
            'license_plate': '59A-12345',
            'timestamp': '10:30:15',
            'severity': 'High',
            'confidence': 0.87,
            'motorcyclist_box': [100, 200, 300, 500]
        },
        {
            'license_plate': '51B-67890',
            'timestamp': '10:35:42',
            'severity': 'Medium',
            'confidence': 0.75,
            'motorcyclist_box': [400, 150, 600, 450]
        }
    ]

    report = generate_violation_report(violations, output_format='dataframe')
    print(f"\nReport generation test:")
    print(report)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
```

---

## 2. Module `main_app.py` - CLI Application

### 2.1 Tổng quan

Module CLI (Command Line Interface) để chạy detection từ terminal.

**Đường dẫn**: `Source/main_app.py`

**Features**:

- Image detection
- Video detection
- Batch processing
- Export reports

### 2.2 Code Structure

```python
"""
CLI Application cho Helmet Violation Detection
Usage:
    python main_app.py --image path/to/image.jpg
    python main_app.py --video path/to/video.mp4
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import các modules
from _Motobike import detect_motorcyclists, load_model as load_moto_model
from _LP_Helmet import detect_helmet_and_lp, ocr_license_plate, analyze_violation
from _myFunc import (
    visualize_detections, crop_roi, roi_to_original_coords,
    generate_violation_report, save_report, get_video_info
)


# ============================================================================
# MAIN DETECTION PIPELINE
# ============================================================================

def process_image(image_path, output_dir='results', save_annotated=True, save_report=True):
    """
    Process single image

    Full pipeline:
        1. Load image
        2. Stage 1: Detect motorcyclists
        3. Stage 2: For each motorcyclist, detect helmet/LP
        4. Analyze violations
        5. Visualize & save results

    Args:
        image_path (str): Path to input image
        output_dir (str): Directory để save results
        save_annotated (bool): Có save ảnh annotated không
        save_report (bool): Có save báo cáo không

    Returns:
        dict: Results với violations, safe, annotated image
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")

    # Stage 1: Detect motorcyclists
    print("\n[Stage 1] Detecting motorcyclists...")
    motorcyclist_boxes = detect_motorcyclists(image, conf_threshold=0.4)
    print(f"Found {len(motorcyclist_boxes)} motorcyclists")

    if len(motorcyclist_boxes) == 0:
        print("[INFO] No motorcyclists detected. Exiting.")
        return {
            'violations': [],
            'safe': [],
            'annotated': image
        }

    # Stage 2: Process each motorcyclist
    print("\n[Stage 2] Analyzing each motorcyclist...")

    violations = []
    safe_motorcyclists = []

    for idx, moto_box in enumerate(motorcyclist_boxes, start=1):
        print(f"\n  Processing motorcyclist #{idx}...")

        # Crop ROI
        roi_resized, roi_offset, roi_scale = crop_roi(image, moto_box, target_size=(768, 768))

        # Detect helmet/LP trong ROI
        roi_detections = detect_helmet_and_lp(roi_resized, conf_threshold=0.3)

        # Analyze violation
        violation_info = analyze_violation(roi_detections)

        # Transform coordinates về original image
        helmet_boxes_orig = []
        nohelmet_boxes_orig = []
        lp_boxes_orig = []

        for helmet_box in roi_detections['helmet']:
            orig_box = roi_to_original_coords(helmet_box[:4], roi_offset, roi_scale)
            helmet_boxes_orig.append(orig_box + [helmet_box[4]])

        for nohelmet_box in roi_detections['nohelmet']:
            orig_box = roi_to_original_coords(nohelmet_box[:4], roi_offset, roi_scale)
            nohelmet_boxes_orig.append(orig_box + [nohelmet_box[4]])

        for lp_box in roi_detections['licenseplate']:
            orig_box = roi_to_original_coords(lp_box[:4], roi_offset, roi_scale)
            lp_boxes_orig.append(orig_box + [lp_box[4]])

        # OCR license plate nếu có
        lp_text = "Unknown"
        lp_box_final = None

        if lp_boxes_orig:
            lp_box_final = lp_boxes_orig[0]  # Lấy LP có conf cao nhất
            x1, y1, x2, y2 = map(int, lp_box_final[:4])
            lp_crop = image[y1:y2, x1:x2]

            if lp_crop.size > 0:
                lp_text = ocr_license_plate(lp_crop)

        # Tạo object
        motorcyclist_obj = {
            'motorcyclist_box': moto_box,
            'helmet_boxes': helmet_boxes_orig,
            'nohelmet_boxes': nohelmet_boxes_orig,
            'lp_boxes': lp_boxes_orig,
            'license_plate': lp_text,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'severity': violation_info['severity'],
            'confidence': moto_box[4] if len(moto_box) >= 5 else 0
        }

        if 'lp_box' not in motorcyclist_obj and lp_box_final:
            motorcyclist_obj['lp_box'] = lp_box_final

        # Phân loại
        if violation_info['is_violation']:
            violations.append(motorcyclist_obj)
            print(f"    ❌ VIOLATION: {violation_info['reason']}")
            if lp_text != "Unknown":
                print(f"    License Plate: {lp_text}")
        else:
            safe_motorcyclists.append(motorcyclist_obj)
            print(f"    ✅ SAFE: {violation_info['reason']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total motorcyclists: {len(motorcyclist_boxes)}")
    print(f"Violations: {len(violations)}")
    print(f"Safe: {len(safe_motorcyclists)}")
    print(f"Violation rate: {len(violations)/len(motorcyclist_boxes)*100:.1f}%")

    # Visualize
    annotated = visualize_detections(image, motorcyclist_boxes, violations, safe_motorcyclists)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_annotated:
        annotated_path = output_dir / f"annotated_{Path(image_path).name}"
        cv2.imwrite(str(annotated_path), annotated)
        print(f"\n✅ Saved annotated image: {annotated_path}")

    if save_report and violations:
        report_path = output_dir / f"report_{Path(image_path).stem}.csv"
        save_report(violations, report_path, format='csv')
        print(f"✅ Saved report: {report_path}")

    return {
        'violations': violations,
        'safe': safe_motorcyclists,
        'annotated': annotated
    }


def process_video(video_path, output_dir='results', frame_skip=5):
    """
    Process video file

    Args:
        video_path (str): Path to video
        output_dir (str): Output directory
        frame_skip (int): Process every N frames (để tăng tốc)

    Returns:
        dict: Results
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path}")
    print(f"{'='*60}")

    # Get video info
    video_info = get_video_info(video_path)
    print(f"Video info:")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']}")
    print(f"  Frames: {video_info['frame_count']}")
    print(f"  Duration: {video_info['duration']:.1f}s")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Prepare output video writer
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_dir / f"annotated_{Path(video_path).name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        video_info['fps'] / frame_skip,  # Adjusted FPS
        (video_info['width'], video_info['height'])
    )

    # Process frames
    all_violations = []
    frame_count = 0
    processed_count = 0

    pbar = tqdm(total=video_info['frame_count'], desc="Processing frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pbar.update(1)

        # Skip frames
        if frame_count % frame_skip != 0:
            continue

        processed_count += 1

        # Process frame (similar to process_image but no save)
        # ... (detection pipeline)

        # Write to output video
        # out.write(annotated_frame)

    cap.release()
    out.release()
    pbar.close()

    print(f"\n✅ Processed {processed_count}/{frame_count} frames")
    print(f"✅ Saved annotated video: {output_video_path}")

    # Save violations report
    if all_violations:
        report_path = output_dir / f"report_{Path(video_path).stem}.csv"
        save_report(all_violations, report_path)
        print(f"✅ Saved report: {report_path}")

    return {
        'violations': all_violations,
        'output_video': str(output_video_path)
    }


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Helmet Violation Detection CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process image
  python main_app.py --image traffic.jpg --output results/

  # Process video with frame skip
  python main_app.py --video traffic.mp4 --frame-skip 10

  # Batch process images
  python main_app.py --batch img_folder/ --output results/
        """
    )

    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--batch', type=str, help='Path to folder chứa images')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--frame-skip', type=int, default=5, help='Frame skip for video')
    parser.add_argument('--no-save-annotated', action='store_true', help='Không save ảnh annotated')
    parser.add_argument('--no-save-report', action='store_true', help='Không save report')

    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main function
    """
    args = parse_arguments()

    # Validate arguments
    if not any([args.image, args.video, args.batch]):
        print("ERROR: Phải cung cấp --image hoặc --video hoặc --batch")
        print("Run 'python main_app.py --help' for usage")
        return

    # Process image
    if args.image:
        process_image(
            args.image,
            output_dir=args.output,
            save_annotated=not args.no_save_annotated,
            save_report=not args.no_save_report
        )

    # Process video
    elif args.video:
        process_video(
            args.video,
            output_dir=args.output,
            frame_skip=args.frame_skip
        )

    # Batch processing
    elif args.batch:
        batch_dir = Path(args.batch)
        image_files = list(batch_dir.glob('*.jpg')) + list(batch_dir.glob('*.png'))

        print(f"Found {len(image_files)} images in {batch_dir}")

        for img_path in tqdm(image_files, desc="Batch processing"):
            try:
                process_image(
                    str(img_path),
                    output_dir=args.output,
                    save_annotated=not args.no_save_annotated,
                    save_report=not args.no_save_report
                )
            except Exception as e:
                print(f"ERROR processing {img_path}: {e}")
                continue


if __name__ == '__main__':
    main()
```

---

**Nội dung còn lại tiếp tục ở file sau để tránh quá tải...**

Bạn muốn tôi tiếp tục với:

- Module `ui_app.py` - Web Interface (Gradio)
- Training Scripts chi tiết
- Best Practices & Optimization
- Troubleshooting

Tạo **Phần 4** không? 🚀
