# 📚 LÝ THUYẾT VÀ GIẢI THÍCH CODE - PHẦN 4 (FINAL)

> **Tài liệu lý thuyết và giải thích code cho dự án Helmet Violation Detection**  
> **Phần 4: Web UI, Training, Optimization & Troubleshooting**

---

## 📋 Mục lục Phần 4

1. [Module ui_app.py - Gradio Web Interface](#1-module-ui_apppy---gradio-web-interface)
2. [Training Scripts Chi tiết](#2-training-scripts-chi-tiết)
3. [Dataset Preparation Workflow](#3-dataset-preparation-workflow)
4. [Best Practices & Optimization](#4-best-practices--optimization)
5. [Troubleshooting Common Issues](#5-troubleshooting-common-issues)
6. [Production Deployment](#6-production-deployment)
7. [Future Improvements](#7-future-improvements)

---

## 1. Module `ui_app.py` - Gradio Web Interface

### 1.1 Tổng quan

Module này tạo **Web UI** bằng Gradio để người dùng có thể upload ảnh/video và xem kết quả detection qua browser.

**Đường dẫn**: `Source/ui_app.py`

**Features**:

- Upload ảnh/video qua web browser
- Real-time detection với progress bar
- Hiển thị kết quả annotated
- Bảng báo cáo vi phạm
- Download results

### 1.2 Code đầy đủ với giải thích

```python
"""
Gradio Web UI cho Helmet Violation Detection
Author: Helmet Violation Detection Team

Usage:
    python ui_app.py
    # Mở browser: http://127.0.0.1:7860
"""

import gradio as gr
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime

# Import detection modules
from _Motobike import detect_motorcyclists
from _LP_Helmet import detect_helmet_and_lp, ocr_license_plate, analyze_violation
from _myFunc import (
    crop_roi, roi_to_original_coords, visualize_detections,
    generate_violation_report, calculate_detection_stats
)


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

CONFIG = {
    'moto_model': 'models/Motov10l.pt',
    'helmet_model': 'models/HelmetLP.pt',
    'moto_conf': 0.4,
    'helmet_conf': 0.3,
    'roi_size': 768,
    'max_image_size': 1920  # Resize nếu lớn hơn
}


# ============================================================================
# CORE DETECTION FUNCTION FOR UI
# ============================================================================

def detect_violations_ui(image, moto_conf, helmet_conf, progress=gr.Progress()):
    """
    Main detection function cho Gradio UI

    Args:
        image (numpy.ndarray): Input image từ Gradio (RGB format)
        moto_conf (float): Motorcyclist confidence threshold
        helmet_conf (float): Helmet/LP confidence threshold
        progress (gr.Progress): Gradio progress tracker

    Returns:
        tuple: (annotated_image, report_dataframe, stats_text)
    """
    try:
        # Update progress
        progress(0, desc="Đang khởi tạo...")

        # Convert RGB to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize nếu quá lớn
        h, w = image_bgr.shape[:2]
        if max(h, w) > CONFIG['max_image_size']:
            scale = CONFIG['max_image_size'] / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_bgr = cv2.resize(image_bgr, (new_w, new_h))

        # Stage 1: Detect motorcyclists
        progress(0.2, desc="🏍️ Đang phát hiện xe máy...")
        motorcyclist_boxes = detect_motorcyclists(
            image_bgr,
            conf_threshold=moto_conf
        )

        if len(motorcyclist_boxes) == 0:
            progress(1.0, desc="✅ Hoàn thành!")
            return (
                image,  # Return original image (RGB)
                pd.DataFrame(columns=['STT', 'Biển số xe', 'Thời gian', 'Mức độ']),
                "❌ Không phát hiện xe máy nào trong ảnh"
            )

        # Stage 2: Analyze each motorcyclist
        progress(0.4, desc=f"🔍 Phân tích {len(motorcyclist_boxes)} xe máy...")

        violations = []
        safe_motorcyclists = []

        for idx, moto_box in enumerate(motorcyclist_boxes):
            # Update progress
            progress_val = 0.4 + (0.4 * (idx + 1) / len(motorcyclist_boxes))
            progress(progress_val, desc=f"Đang xử lý xe thứ {idx+1}/{len(motorcyclist_boxes)}...")

            # Crop ROI
            roi_resized, roi_offset, roi_scale = crop_roi(
                image_bgr,
                moto_box,
                target_size=(CONFIG['roi_size'], CONFIG['roi_size'])
            )

            # Detect helmet/LP
            roi_detections = detect_helmet_and_lp(
                roi_resized,
                conf_threshold=helmet_conf
            )

            # Analyze violation
            violation_info = analyze_violation(roi_detections)

            # Transform coordinates
            helmet_boxes_orig = []
            nohelmet_boxes_orig = []
            lp_boxes_orig = []

            for helmet_box in roi_detections['helmet']:
                orig_box = roi_to_original_coords(
                    helmet_box[:4], roi_offset, roi_scale
                )
                helmet_boxes_orig.append(orig_box + [helmet_box[4]])

            for nohelmet_box in roi_detections['nohelmet']:
                orig_box = roi_to_original_coords(
                    nohelmet_box[:4], roi_offset, roi_scale
                )
                nohelmet_boxes_orig.append(orig_box + [nohelmet_box[4]])

            for lp_box in roi_detections['licenseplate']:
                orig_box = roi_to_original_coords(
                    lp_box[:4], roi_offset, roi_scale
                )
                lp_boxes_orig.append(orig_box + [lp_box[4]])

            # OCR license plate
            lp_text = "Unknown"
            lp_box_final = None

            if lp_boxes_orig:
                lp_box_final = lp_boxes_orig[0]
                x1, y1, x2, y2 = map(int, lp_box_final[:4])

                # Ensure valid crop
                if 0 <= y1 < y2 <= image_bgr.shape[0] and 0 <= x1 < x2 <= image_bgr.shape[1]:
                    lp_crop = image_bgr[y1:y2, x1:x2]
                    if lp_crop.size > 0:
                        lp_text = ocr_license_plate(lp_crop)

            # Create object
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

            if lp_box_final:
                motorcyclist_obj['lp_box'] = lp_box_final

            # Classify
            if violation_info['is_violation']:
                violations.append(motorcyclist_obj)
            else:
                safe_motorcyclists.append(motorcyclist_obj)

        # Visualize
        progress(0.9, desc="🎨 Đang tạo hình ảnh kết quả...")
        annotated_bgr = visualize_detections(
            image_bgr,
            motorcyclist_boxes,
            violations,
            safe_motorcyclists
        )

        # Convert back to RGB for Gradio
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Generate report
        report_df = generate_violation_report(violations, output_format='dataframe')

        # Generate stats text
        stats = calculate_detection_stats(violations, safe_motorcyclists)
        stats_text = f"""
📊 **THỐNG KÊ PHÁT HIỆN**

- 🏍️ Tổng số xe máy: {stats['total_motorcyclists']}
- ❌ Vi phạm: {stats['violations']} ({stats['violation_rate']})
- ✅ An toàn: {stats['safe']}

**Chi tiết vi phạm:**
- 🔴 Nghiêm trọng (High): {stats['severity_breakdown']['High']}
- 🟠 Trung bình (Medium): {stats['severity_breakdown']['Medium']}
- 🟡 Nhẹ (Low): {stats['severity_breakdown']['Low']}
        """

        progress(1.0, desc="✅ Hoàn thành!")

        return annotated_rgb, report_df, stats_text

    except Exception as e:
        progress(1.0, desc="❌ Lỗi!")
        error_msg = f"❌ **LỖI**: {str(e)}\n\nVui lòng thử lại hoặc kiểm tra log."
        return image, pd.DataFrame(), error_msg


def process_video_ui(video_path, moto_conf, helmet_conf, frame_skip, progress=gr.Progress()):
    """
    Process video cho Gradio UI

    Args:
        video_path (str): Path to uploaded video
        moto_conf (float): Motorcyclist confidence
        helmet_conf (float): Helmet/LP confidence
        frame_skip (int): Process every N frames
        progress (gr.Progress): Progress tracker

    Returns:
        tuple: (output_video_path, report_dataframe, stats_text)
    """
    try:
        progress(0, desc="Đang mở video...")

        # Open video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None, pd.DataFrame(), "❌ Không thể mở video"

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create temporary output video
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps/frame_skip, (width, height))

        # Process frames
        all_violations = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Update progress
            progress_val = frame_idx / total_frames
            progress(progress_val, desc=f"Đang xử lý frame {frame_idx}/{total_frames}")

            # Skip frames
            if frame_idx % frame_skip != 0:
                continue

            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect (without progress bar for video frames)
            annotated_rgb, _, _ = detect_violations_ui(
                frame_rgb, moto_conf, helmet_conf, progress=lambda *args: None
            )

            # Convert back to BGR
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(annotated_bgr)

        cap.release()
        out.release()

        progress(1.0, desc="✅ Hoàn thành!")

        # Generate report
        report_df = pd.DataFrame()  # Simplified for video
        stats_text = f"✅ Đã xử lý {frame_idx} frames\n📹 Video đã được lưu"

        return output_path, report_df, stats_text

    except Exception as e:
        return None, pd.DataFrame(), f"❌ Lỗi: {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """
    Tạo Gradio interface

    Returns:
        gr.Blocks: Gradio app
    """

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    .gr-button-secondary {
        background: #f3f4f6 !important;
    }
    footer {
        display: none !important;
    }
    """

    with gr.Blocks(css=custom_css, title="Helmet Violation Detection") as app:

        # Header
        gr.Markdown("""
        # 🛵 Helmet Violation Detection System

        > **Hệ thống phát hiện vi phạm không đội mũ bảo hiểm sử dụng YOLOv8 2-Stage Detection**

        ---
        """)

        with gr.Tabs() as tabs:

            # ========== TAB 1: IMAGE DETECTION ==========
            with gr.Tab("🖼️ Xử lý Ảnh"):
                gr.Markdown("### Upload ảnh giao thông để phát hiện vi phạm")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Input
                        image_input = gr.Image(
                            label="📤 Upload ảnh",
                            type="numpy",
                            height=400
                        )

                        # Configuration
                        gr.Markdown("#### ⚙️ Cấu hình")

                        moto_conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.4,
                            step=0.05,
                            label="Confidence - Phát hiện xe máy",
                            info="Ngưỡng confidence cho motorcyclist detection"
                        )

                        helmet_conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Confidence - Phát hiện mũ/biển số",
                            info="Ngưỡng confidence cho helmet/LP detection"
                        )

                        # Buttons
                        with gr.Row():
                            detect_btn = gr.Button(
                                "🔍 Phát hiện Vi phạm",
                                variant="primary",
                                size="lg"
                            )
                            clear_btn = gr.Button(
                                "🗑️ Xóa",
                                variant="secondary"
                            )

                    with gr.Column(scale=1):
                        # Output
                        image_output = gr.Image(
                            label="📊 Kết quả Detection",
                            type="numpy",
                            height=400
                        )

                        stats_output = gr.Markdown(
                            label="📈 Thống kê",
                            value="*Chưa có kết quả*"
                        )

                # Report table
                gr.Markdown("### 📋 Báo cáo Vi phạm")
                report_output = gr.Dataframe(
                    headers=['STT', 'Biển số xe', 'Thời gian', 'Mức độ', 'Confidence', 'Vị trí'],
                    label="Danh sách vi phạm",
                    interactive=False
                )

                # Examples
                gr.Markdown("### 📸 Ảnh mẫu")
                gr.Examples(
                    examples=[
                        # Add paths to example images if available
                    ],
                    inputs=image_input,
                    label="Click để thử với ảnh mẫu"
                )

                # Event handlers
                detect_btn.click(
                    fn=detect_violations_ui,
                    inputs=[image_input, moto_conf_slider, helmet_conf_slider],
                    outputs=[image_output, report_output, stats_output]
                )

                clear_btn.click(
                    fn=lambda: (None, None, pd.DataFrame(), "*Đã xóa*"),
                    outputs=[image_input, image_output, report_output, stats_output]
                )

            # ========== TAB 2: VIDEO DETECTION ==========
            with gr.Tab("🎥 Xử lý Video"):
                gr.Markdown("### Upload video giao thông để phát hiện vi phạm")

                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="📤 Upload video",
                            height=400
                        )

                        gr.Markdown("#### ⚙️ Cấu hình")

                        video_moto_conf = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.4,
                            step=0.05,
                            label="Confidence - Xe máy"
                        )

                        video_helmet_conf = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Confidence - Mũ/Biển số"
                        )

                        frame_skip_slider = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=5,
                            step=1,
                            label="Frame Skip",
                            info="Xử lý mỗi N frames (tăng tốc độ)"
                        )

                        process_video_btn = gr.Button(
                            "▶️ Xử lý Video",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        video_output = gr.Video(
                            label="📹 Video kết quả",
                            height=400
                        )

                        video_stats = gr.Markdown(
                            value="*Chưa có kết quả*"
                        )

                video_report = gr.Dataframe(
                    label="Báo cáo vi phạm"
                )

                # Event handler
                process_video_btn.click(
                    fn=process_video_ui,
                    inputs=[
                        video_input,
                        video_moto_conf,
                        video_helmet_conf,
                        frame_skip_slider
                    ],
                    outputs=[video_output, video_report, video_stats]
                )

            # ========== TAB 3: ABOUT ==========
            with gr.Tab("ℹ️ Giới thiệu"):
                gr.Markdown("""
                ## 🛵 Về Hệ thống

                ### 🎯 Mục đích
                Phát hiện vi phạm không đội mũ bảo hiểm khi tham gia giao thông,
                tự động nhận diện biển số xe vi phạm.

                ### 🏗️ Kiến trúc

                **2-Stage Detection Pipeline:**

                1. **Stage 1**: Phát hiện xe máy (motorcyclist) trong ảnh giao thông
                   - Model: YOLOv8l (Motov10l.pt)
                   - Input: Full scene 640x640
                   - Output: Bounding boxes của xe máy

                2. **Stage 2**: Phát hiện mũ bảo hiểm và biển số trong ROI
                   - Model: YOLOv8 custom (HelmetLP.pt)
                   - Input: ROI crops 768x768
                   - Output: helmet, nohelmet, licenseplate detections

                3. **OCR**: Đọc biển số xe bằng EasyOCR

                ### 📊 Performance

                - **mAP@0.5**: ~0.85
                - **Speed**: 15-25 FPS (RTX 3050 6GB)
                - **Accuracy**: ~85% overall
                - **False Positive Rate**: <10%

                ### 🛠️ Technologies

                - **Deep Learning**: PyTorch 2.6, YOLOv8
                - **Computer Vision**: OpenCV
                - **OCR**: EasyOCR
                - **Web UI**: Gradio

                ### 👥 Đội ngũ phát triển

                - **Nguyen Dinh Thanh San** - [@ThanhSan97](https://github.com/ThanhSan97)
                - **Nguyen Huynh Chi Khang** - [@Khang1405](https://github.com/Khang1405)
                - **Nguyen Phan Duc Thanh** - [@NguyenPhanDucThanh](https://github.com/NguyenPhanDucThanh)

                ### 🔗 Links

                - **GitHub**: [Helmet-Violation-Detection](https://github.com/phucmanh1310/Helmet-Violation-Detection-Using-YOLO-and-VGG16)
                - **Documentation**: [README.md](README.md)

                ### 📄 License

                MIT License - Free to use and modify

                ---

                Made with ❤️ by Computer Vision Team
                """)

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666;">
            <p>Helmet Violation Detection System v2.0 | Powered by YOLOv8 & Gradio</p>
        </div>
        """)

    return app


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Launch Gradio app
    """
    app = create_interface()

    # Launch với cấu hình
    app.launch(
        server_name="0.0.0.0",  # Cho phép access từ mạng LAN
        server_port=7860,
        share=False,  # Set True để tạo public link (gradio.live)
        inbrowser=True,  # Auto mở browser
        favicon_path=None,  # Path to favicon nếu có
        show_error=True
    )


if __name__ == '__main__':
    print("="*60)
    print("STARTING HELMET VIOLATION DETECTION WEB UI")
    print("="*60)
    print("\nLoading models...")
    print("This may take a few seconds...\n")

    main()
```

### 1.3 Giải thích các thành phần Gradio

#### Progress Tracking

```python
def detect_violations_ui(image, moto_conf, helmet_conf, progress=gr.Progress()):
    progress(0, desc="Đang khởi tạo...")
    # ... stage 1
    progress(0.2, desc="🏍️ Đang phát hiện xe máy...")
    # ... stage 2
    progress(0.4, desc=f"🔍 Phân tích {len(motorcyclist_boxes)} xe máy...")
    # ... visualization
    progress(0.9, desc="🎨 Đang tạo hình ảnh kết quả...")
    progress(1.0, desc="✅ Hoàn thành!")
```

**Lý do**: User experience tốt hơn, biết được process đang ở đâu.

#### Tab Organization

```python
with gr.Tabs():
    with gr.Tab("🖼️ Xử lý Ảnh"):
        # Image processing UI

    with gr.Tab("🎥 Xử lý Video"):
        # Video processing UI

    with gr.Tab("ℹ️ Giới thiệu"):
        # About & documentation
```

**Lý do**: Tách biệt chức năng, UI gọn gàng, dễ navigate.

#### Event Handlers

```python
detect_btn.click(
    fn=detect_violations_ui,
    inputs=[image_input, moto_conf_slider, helmet_conf_slider],
    outputs=[image_output, report_output, stats_output]
)
```

**Lý do**: Kết nối UI components với logic functions.

---

## 2. Training Scripts Chi tiết

### 2.1 Script `train_model1_motorcyclist.py`

```python
"""
Training script cho Model 1: Motorcyclist Detection

Đặc điểm:
- PyTorch 2.6 compatible (weights_only fix)
- Windows multiprocessing safe
- Auto-resume từ checkpoint
- Extensive logging

Usage:
    py -3.13 scripts/train_model1_motorcyclist.py
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import multiprocessing

# ============================================================================
# PYTORCH 2.6 COMPATIBILITY FIX
# ============================================================================

# Monkey-patch torch.load để fix weights_only issue
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """
    Wrapper cho torch.load với weights_only=False

    PyTorch 2.6 mặc định weights_only=True → gây lỗi khi load YOLO checkpoints
    """
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

print("[INFO] Applied PyTorch 2.6 compatibility patch (weights_only=False)")


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Dataset
    'data_yaml': 'data/_stage1_motorcyclist/data.yaml',

    # Model
    'base_model': 'yolov8l.pt',  # Large variant

    # Training hyperparameters
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'device': 'cuda',  # 'cuda' hoặc 'cpu'
    'workers': 2,  # QUAN TRỌNG: Workers=2 cho Windows

    # Optimizer
    'optimizer': 'AdamW',
    'lr0': 0.001,  # Initial learning rate
    'lrf': 0.01,   # Final learning rate (lr0 * lrf)
    'momentum': 0.937,
    'weight_decay': 0.0005,

    # Augmentation
    'mosaic': 1.0,
    'mixup': 0.5,
    'copy_paste': 0.5,
    'degrees': 10.0,  # Rotation ±10°
    'translate': 0.1,  # Translation ±10%
    'scale': 0.5,      # Scale ±50%
    'shear': 10.0,     # Shear ±10°
    'perspective': 0.0,
    'flipud': 0.0,     # Flip up-down (không dùng cho giao thông)
    'fliplr': 0.5,     # Flip left-right
    'hsv_h': 0.015,    # Hue augmentation
    'hsv_s': 0.7,      # Saturation
    'hsv_v': 0.4,      # Value/Brightness

    # Regularization
    'dropout': 0.0,
    'label_smoothing': 0.0,

    # Output
    'project': 'runs/detect',
    'name': 'model1_motorcyclist',
    'exist_ok': False,  # Tạo folder mới nếu tồn tại
    'save': True,
    'save_period': 10,  # Save checkpoint mỗi 10 epochs

    # Validation
    'val': True,
    'patience': 50,  # Early stopping patience

    # Logging
    'verbose': True,
    'plots': True
}


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model1():
    """
    Train Model 1: Motorcyclist Detection

    Steps:
        1. Validate paths
        2. Load base model
        3. Configure training
        4. Start training
        5. Validate final model
    """
    print("="*60)
    print("TRAINING MODEL 1: MOTORCYCLIST DETECTION")
    print("="*60)

    # Validate data.yaml
    data_yaml = Path(TRAINING_CONFIG['data_yaml'])
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    print(f"\n📁 Dataset: {data_yaml}")

    # Load base model
    print(f"\n🔧 Loading base model: {TRAINING_CONFIG['base_model']}")
    model = YOLO(TRAINING_CONFIG['base_model'])

    # Print model info
    print(f"\n📊 Model info:")
    print(f"  - Architecture: YOLOv8l")
    print(f"  - Parameters: ~43.7M")
    print(f"  - Task: Object Detection")

    # Training configuration summary
    print(f"\n⚙️ Training configuration:")
    print(f"  - Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  - Batch size: {TRAINING_CONFIG['batch']}")
    print(f"  - Image size: {TRAINING_CONFIG['imgsz']}")
    print(f"  - Device: {TRAINING_CONFIG['device']}")
    print(f"  - Workers: {TRAINING_CONFIG['workers']}")
    print(f"  - Optimizer: {TRAINING_CONFIG['optimizer']}")
    print(f"  - Learning rate: {TRAINING_CONFIG['lr0']} → {TRAINING_CONFIG['lrf']}")

    # Start training
    print(f"\n🚀 Starting training...")
    print(f"{'='*60}\n")

    results = model.train(
        data=str(data_yaml),
        epochs=TRAINING_CONFIG['epochs'],
        imgsz=TRAINING_CONFIG['imgsz'],
        batch=TRAINING_CONFIG['batch'],
        device=TRAINING_CONFIG['device'],
        workers=TRAINING_CONFIG['workers'],

        # Optimizer
        optimizer=TRAINING_CONFIG['optimizer'],
        lr0=TRAINING_CONFIG['lr0'],
        lrf=TRAINING_CONFIG['lrf'],
        momentum=TRAINING_CONFIG['momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay'],

        # Augmentation
        mosaic=TRAINING_CONFIG['mosaic'],
        mixup=TRAINING_CONFIG['mixup'],
        copy_paste=TRAINING_CONFIG['copy_paste'],
        degrees=TRAINING_CONFIG['degrees'],
        translate=TRAINING_CONFIG['translate'],
        scale=TRAINING_CONFIG['scale'],
        shear=TRAINING_CONFIG['shear'],
        perspective=TRAINING_CONFIG['perspective'],
        flipud=TRAINING_CONFIG['flipud'],
        fliplr=TRAINING_CONFIG['fliplr'],
        hsv_h=TRAINING_CONFIG['hsv_h'],
        hsv_s=TRAINING_CONFIG['hsv_s'],
        hsv_v=TRAINING_CONFIG['hsv_v'],

        # Regularization
        dropout=TRAINING_CONFIG['dropout'],
        label_smoothing=TRAINING_CONFIG['label_smoothing'],

        # Output
        project=TRAINING_CONFIG['project'],
        name=TRAINING_CONFIG['name'],
        exist_ok=TRAINING_CONFIG['exist_ok'],
        save=TRAINING_CONFIG['save'],
        save_period=TRAINING_CONFIG['save_period'],

        # Validation
        val=TRAINING_CONFIG['val'],
        patience=TRAINING_CONFIG['patience'],

        # Logging
        verbose=TRAINING_CONFIG['verbose'],
        plots=TRAINING_CONFIG['plots']
    )

    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*60}")

    # Print results
    print(f"\n📊 Final metrics:")
    print(f"  - Box mAP@0.5: {results.box.map50:.4f}")
    print(f"  - Box mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  - Precision: {results.box.mp:.4f}")
    print(f"  - Recall: {results.box.mr:.4f}")

    # Paths
    save_dir = Path(TRAINING_CONFIG['project']) / TRAINING_CONFIG['name']
    print(f"\n📁 Results saved to: {save_dir}")
    print(f"  - Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"  - Last weights: {save_dir / 'weights' / 'last.pt'}")
    print(f"  - Metrics: {save_dir / 'results.csv'}")
    print(f"  - Plots: {save_dir / '*.png'}")

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main function với Windows multiprocessing safety
    """
    # Windows multiprocessing requirement
    multiprocessing.freeze_support()

    # Set start method (IMPORTANT for Windows)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Run training
    train_model1()


if __name__ == '__main__':
    main()
```

### 2.2 Giải thích Training Process

#### Loss Function

YOLOv8 sử dụng **composite loss**:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}
$$

**1. Box Loss (CIoU)**:

```python
# Complete IoU Loss
ciou_loss = 1 - ciou(predicted_box, target_box)

# CIoU considers:
# - IoU (overlap)
# - Distance between centers
# - Aspect ratio consistency
```

**2. Classification Loss (BCE)**:

```python
# Binary Cross Entropy
bce_loss = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

**3. Distribution Focal Loss (DFL)**:

```python
# Improves box regression accuracy
# Treats box regression as classification problem
```

#### Learning Rate Schedule

```python
# Cosine annealing
lr = lr0 * (lrf + (1 - lrf) * (1 + cos(π * epoch / epochs)) / 2)

# Example:
# lr0 = 0.001
# lrf = 0.01
# At epoch 0: lr = 0.001
# At epoch 50: lr ≈ 0.005
# At epoch 100: lr = 0.00001
```

---

## 3. Dataset Preparation Workflow

### 3.1 YOLO Dataset Format

```
dataset/
├── data.yaml           # ⭐ Config file
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt   # Same name với image
│       ├── img2.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

#### data.yaml Structure

```yaml
# Paths (absolute paths khuyến nghị)
path: D:/datasets/motorcyclist
train: train/images
val: valid/images
test: test/images

# Classes
nc: 1 # Number of classes
names:
  0: motorcyclist

# Optional
download: false # Không auto download
```

#### Label Format (.txt)

```
# Format: class_id x_center y_center width height
# All values normalized to [0, 1]

# Example: img1.txt
0 0.512 0.345 0.156 0.287
0 0.723 0.521 0.123 0.198

# Explanation:
# Line 1: Class 0 (motorcyclist) tại center (0.512, 0.345)
#         với width 0.156, height 0.287
```

### 3.2 Data Augmentation Examples

```python
# Mosaic: Ghép 4 ảnh thành 1
┌─────────┬─────────┐
│  Img 1  │  Img 2  │
├─────────┼─────────┤
│  Img 3  │  Img 4  │
└─────────┴─────────┘

# Mixup: Alpha blending 2 ảnh
new_img = α * img1 + (1-α) * img2

# Random Perspective
- Rotation: ±10°
- Translation: ±10%
- Scale: ±50%
- Shear: ±10°

# HSV Augmentation
- Hue shift: ±1.5%
- Saturation: ±70%
- Value/Brightness: ±40%
```

---

## 4. Best Practices & Optimization

### 4.1 Performance Optimization

#### GPU Memory Optimization

```python
# 1. Batch size tuning
# RTX 3050 6GB:
batch_size = 16  # For 640x640
batch_size = 8   # For 768x768
batch_size = 4   # For 1280x1280

# 2. Mixed Precision Training (AMP)
# YOLOv8 tự động dùng AMP nếu CUDA available
# → Faster training, less memory

# 3. Gradient Accumulation (nếu GPU nhỏ)
# Simulate larger batch size
accumulate = 4  # Effective batch = batch_size * accumulate
```

#### Inference Optimization

```python
# 1. Model quantization
model = YOLO('model.pt')
model.export(format='engine')  # TensorRT (fastest)
model.export(format='onnx')    # ONNX (cross-platform)

# 2. Batched inference
images = [img1, img2, img3, ...]
results = model.predict(images, batch=8)  # Process 8 images at once

# 3. Half precision
model = YOLO('model.pt')
results = model.predict(img, half=True)  # FP16 instead of FP32
```

### 4.2 Code Quality Best Practices

```python
# 1. Type hints
def detect_motorcyclists(
    image: np.ndarray,
    conf_threshold: float = 0.4
) -> List[List[float]]:
    """Detect motorcyclists"""
    ...

# 2. Docstrings
def process_image(image_path: str) -> dict:
    """
    Process image with full pipeline.

    Args:
        image_path: Path to input image

    Returns:
        Dictionary with violations, safe motorcyclists, annotated image

    Raises:
        FileNotFoundError: If image not found
        ValueError: If image cannot be read

    Example:
        >>> results = process_image('traffic.jpg')
        >>> print(f"Violations: {len(results['violations'])}")
    """
    ...

# 3. Error handling
try:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read: {image_path}")
except Exception as e:
    logger.error(f"Error processing {image_path}: {e}")
    raise

# 4. Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting detection...")
logger.warning("Low confidence detection")
logger.error("Model failed to load")
```

---

## 5. Troubleshooting Common Issues

### 5.1 PyTorch 2.6 Issues

**Problem**: `UnpicklingError` khi load model

```python
# ❌ Error
model = YOLO('model.pt')
# FutureWarning: You are using `torch.load` with `weights_only=False`

# ✅ Fix
import torch
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, weights_only=False, **kwargs)
```

### 5.2 Windows Multiprocessing Issues

**Problem**: `RuntimeError: An attempt has been made to start a new process...`

```python
# ❌ Error
if __name__ == '__main__':
    model.train(data='data.yaml')  # Missing freeze_support

# ✅ Fix
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)

    model.train(data='data.yaml', workers=2)  # workers=2 for Windows
```

### 5.3 CUDA Out of Memory

**Problem**: `CUDA out of memory`

```python
# ✅ Solutions:
# 1. Giảm batch size
batch = 8  # Thay vì 16

# 2. Giảm image size
imgsz = 640  # Thay vì 1280

# 3. Gradient accumulation
# Simulate batch=16 với batch=8
# (YOLOv8 không support trực tiếp, cần custom trainer)

# 4. Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

### 5.4 Low mAP Issues

**Problem**: Model accuracy thấp

```python
# ✅ Solutions:

# 1. Tăng epochs
epochs = 200  # Thay vì 100

# 2. Data augmentation mạnh hơn
mosaic = 1.0
mixup = 0.5
degrees = 15  # More rotation

# 3. Learning rate tuning
lr0 = 0.001  # Try 0.0005 or 0.002
lrf = 0.01   # Try 0.001

# 4. Freeze backbone (transfer learning)
model.train(freeze=10)  # Freeze first 10 layers

# 5. Validate dataset quality
# - Check labels đúng chưa
# - Check ảnh có noise không
# - Check class imbalance
```

---

## 6. Production Deployment

### 6.1 Docker Deployment

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code và models
COPY Source/ ./Source/
COPY models/ ./models/
COPY quick_start_ui.py .

# Expose port
EXPOSE 7860

# Run app
CMD ["python", "quick_start_ui.py"]
```

```bash
# Build
docker build -t helmet-detection .

# Run
docker run -p 7860:7860 --gpus all helmet-detection
```

### 6.2 REST API với FastAPI

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI()

@app.post("/api/detect")
async def detect_violations(file: UploadFile = File(...)):
    """
    API endpoint để detect violations

    Returns:
        JSON với violations, annotated image (base64)
    """
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process
    results = process_image(image)

    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', results['annotated'])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        'violations': len(results['violations']),
        'annotated_image': img_base64,
        'details': results['violations']
    })

# Run: uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 7. Future Improvements

### 7.1 Model Improvements

```python
# 1. Ensemble models
# Combine predictions từ nhiều models
results1 = model1.predict(img)
results2 = model2.predict(img)
final_results = ensemble([results1, results2], weights=[0.6, 0.4])

# 2. Tracking (video)
# Sử dụng ByteTrack/BoT-SORT để track objects qua frames
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.track(source='video.mp4', tracker='bytetrack.yaml')

# 3. Attention mechanisms
# Thêm attention modules vào backbone để focus vào important regions
```

### 7.2 Feature Additions

```python
# 1. Real-time streaming
# WebRTC hoặc RTSP stream processing

# 2. Database integration
# Lưu violations vào database (PostgreSQL, MongoDB)

# 3. Alert system
# Gửi email/SMS khi phát hiện vi phạm

# 4. Analytics dashboard
# Grafana/Plotly dashboard cho statistics
```

---

## 📝 Tổng kết toàn bộ 4 phần

### Phần 1: Lý thuyết cơ bản

- ✅ Object Detection concepts
- ✅ YOLO architecture
- ✅ PyTorch fundamentals
- ✅ Thư viện: Ultralytics, OpenCV, EasyOCR, Gradio

### Phần 2: Kiến trúc & Code

- ✅ 2-Stage detection pipeline chi tiết
- ✅ Module `_Motobike.py` (Stage 1)
- ✅ Module `_LP_Helmet.py` (Stage 2)

### Phần 3: Applications

- ✅ Module `_myFunc.py` (Utilities)
- ✅ Module `main_app.py` (CLI)

### Phần 4: Advanced Topics

- ✅ Module `ui_app.py` (Web UI)
- ✅ Training scripts chi tiết
- ✅ Dataset preparation
- ✅ Best practices & optimization
- ✅ Troubleshooting
- ✅ Production deployment

---

## 🎓 Học từ dự án này

**Core Concepts**:

1. 2-Stage detection approach → Higher accuracy
2. ROI-based processing → Reduce false positives
3. Coordinate transformation → Multi-scale detection
4. PyTorch 2.6 compatibility → Production readiness
5. Windows multiprocessing → Cross-platform development

**Technical Skills**:

- Deep Learning: YOLOv8 training, inference, optimization
- Computer Vision: Detection, OCR, visualization
- Python: OOP, type hints, error handling
- Web Development: Gradio UI, FastAPI
- DevOps: Docker, deployment strategies

---

<div align="center">

### 🎉 HOÀN THÀNH TÀI LIỆU LÝ THUYẾT

**Đã cover toàn bộ kiến thức từ cơ bản đến nâng cao!**

📚 Tổng cộng: **4 phần** | **~500+ dòng giải thích** | **30+ code examples**

[⬆ Quay lại Phần 1](LY_THUYET_VA_GIAI_THICH_CODE_PHAN1.md) |
[Phần 2](LY_THUYET_VA_GIAI_THICH_CODE_PHAN2.md) |
[Phần 3](LY_THUYET_VA_GIAI_THICH_CODE_PHAN3.md)

---

Made with ❤️ for learning purposes

</div>
