# UI Application for Helmet Violation Detection
# Sử dụng Gradio để tạo giao diện web đẹp mắt

import cv2
import numpy as np
import easyocr
from pathlib import Path
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
import tempfile
import os
import torch

# Fix cho PyTorch 2.6+ weights_only issue
# Patch ultralytics để sử dụng weights_only=False
import ultralytics.nn.tasks as tasks
original_torch_safe_load = tasks.torch_safe_load

def patched_torch_safe_load(weight):
    try:
        file = str(weight)
        return torch.load(file, map_location='cpu', weights_only=False), file
    except Exception as e:
        print(f"Error loading {weight}: {e}")
        raise

tasks.torch_safe_load = patched_torch_safe_load

from ultralytics import YOLO

# --- Configuration ---
# Đường dẫn tương đối từ thư mục Source
MOTO_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'Motov10l.pt'
HELMET_LP_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'HelmetLP.pt'
MOTO_CONF = 0.4
HELMET_LP_CONF = 0.4

# Khởi tạo models globally
print("🔄 Đang khởi tạo models...")
moto_model = YOLO(MOTO_MODEL_PATH)
helmet_lp_model = YOLO(HELMET_LP_MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)
print("✅ Models đã sẵn sàng!")

# Biến toàn cục để lưu trữ kết quả
violation_records = []

def draw_results_on_image(image, detections):
    """Vẽ bounding boxes và labels lên ảnh"""
    img_draw = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = (0, 0, 255) if det['violation'] else (0, 255, 0)  # Đỏ nếu vi phạm, xanh nếu tuân thủ
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
        
        # Vẽ label
        label = f"{'VI PHAM' if det['violation'] else 'TUAN THU'}"
        cv2.putText(img_draw, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Vẽ biển số nếu có
        if det['license_plate'] != 'UNKNOWN':
            cv2.putText(img_draw, det['license_plate'], (x1, y2+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return img_draw

def create_violation_table(detections):
    """Tạo bảng thống kê vi phạm"""
    if not detections:
        return pd.DataFrame(columns=['STT', 'Họ và tên', 'Gmail', 'Biển số', 'Thời gian', 'Địa điểm', 'ID'])
    
    data = []
    for idx, det in enumerate(detections):
        if det['violation']:  # Chỉ hiển thị xe vi phạm
            data.append({
                'STT': idx + 1,
                'Họ và tên': 'Chưa xác định',  # Có thể tích hợp nhận diện khuôn mặt sau
                'Gmail': '-',
                'Biển số': det['license_plate'],
                'Thời gian': det['timestamp'],
                'Địa điểm': '-',  # Không hiển thị địa điểm như yêu cầu
                'ID': det['id']
            })
    
    return pd.DataFrame(data)

def process_image_detection(input_image):
    """Xử lý phát hiện vi phạm trên ảnh"""
    if input_image is None:
        return None, pd.DataFrame(), "Vui lòng upload ảnh!"
    
    # Chuyển đổi từ PIL sang OpenCV
    if isinstance(input_image, Image.Image):
        frame = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    else:
        frame = input_image
    
    detections = []
    violation_count = 0
    
    # 1. Phát hiện xe máy
    moto_results = moto_model.predict(frame, conf=MOTO_CONF, verbose=False)
    
    if not moto_results or len(moto_results[0].boxes) == 0:
        return frame, pd.DataFrame(), "❌ Không phát hiện xe máy nào trong ảnh"
    
    # 2. Xử lý từng xe máy
    for i, box in enumerate(moto_results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        moto_crop = frame[y1:y2, x1:x2]
        
        # 3. Phát hiện mũ bảo hiểm và biển số
        helmet_lp_results = helmet_lp_model.predict(moto_crop, conf=HELMET_LP_CONF, verbose=False)[0]
        class_names = helmet_lp_results.names
        
        has_helmet = False
        has_no_helmet = False
        lp_text = "UNKNOWN"
        
        if len(helmet_lp_results.boxes) > 0:
            for det_box in helmet_lp_results.boxes:
                cls_id = int(det_box.cls.item())
                class_name = class_names[cls_id]
                
                if class_name == 'helmet':
                    has_helmet = True
                elif class_name == 'nohelmet':
                    has_no_helmet = True
                elif class_name == 'licenseplate':
                    # Cắt và OCR biển số
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, det_box.xyxy[0])
                    lp_crop = moto_crop[lp_y1:lp_y2, lp_x1:lp_x2]
                    
                    try:
                        ocr_result = reader.readtext(lp_crop, detail=0, paragraph=False)
                        if ocr_result:
                            lp_text = ''.join(ocr_result).replace(" ", "").upper()
                    except Exception as e:
                        print(f"OCR Error: {e}")
        
        # 4. Lưu kết quả phát hiện
        is_violation = has_no_helmet
        if is_violation:
            violation_count += 1
            
        detection = {
            'id': f'MV{i+1:03d}',
            'bbox': (x1, y1, x2, y2),
            'violation': is_violation,
            'has_helmet': has_helmet,
            'has_no_helmet': has_no_helmet,
            'license_plate': lp_text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        detections.append(detection)
    
    # 5. Vẽ kết quả lên ảnh
    result_image = draw_results_on_image(frame, detections)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # 6. Tạo bảng thống kê
    table = create_violation_table(detections)
    
    # 7. Tạo thông tin tóm tắt
    summary = f"""
    📊 **KẾT QUẢ PHÁT HIỆN:**
    - Tổng số xe máy: {len(detections)}
    - Vi phạm: {violation_count} 🚨
    - Tuân thủ: {len(detections) - violation_count} ✅
    """
    
    return result_image, table, summary

def process_video_detection(input_video, progress=gr.Progress()):
    """Xử lý phát hiện vi phạm trên video"""
    if input_video is None:
        return None, pd.DataFrame(), "Vui lòng upload video!"
    
    # Mở video
    cap = cv2.VideoCapture(input_video)
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tạo file output tạm
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frame_count = 0
    process_every_n_frames = 5  # Xử lý mỗi 5 frames để tăng tốc
    
    progress(0, desc="Đang xử lý video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Xử lý frame
        if frame_count % process_every_n_frames == 0:
            # Phát hiện xe máy
            moto_results = moto_model.predict(frame, conf=MOTO_CONF, verbose=False)
            
            if moto_results and len(moto_results[0].boxes) > 0:
                frame_detections = []
                
                for i, box in enumerate(moto_results[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    moto_crop = frame[y1:y2, x1:x2]
                    
                    # Phát hiện mũ bảo hiểm và biển số
                    helmet_lp_results = helmet_lp_model.predict(moto_crop, conf=HELMET_LP_CONF, verbose=False)[0]
                    class_names = helmet_lp_results.names
                    
                    has_helmet = False
                    has_no_helmet = False
                    lp_text = "UNKNOWN"
                    
                    if len(helmet_lp_results.boxes) > 0:
                        for det_box in helmet_lp_results.boxes:
                            cls_id = int(det_box.cls.item())
                            class_name = class_names[cls_id]
                            
                            if class_name == 'helmet':
                                has_helmet = True
                            elif class_name == 'no helmet':
                                has_no_helmet = True
                            elif class_name == 'LP':
                                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, det_box.xyxy[0])
                                lp_crop = moto_crop[lp_y1:lp_y2, lp_x1:lp_x2]
                                
                                try:
                                    ocr_result = reader.readtext(lp_crop, detail=0, paragraph=False)
                                    if ocr_result:
                                        lp_text = ''.join(ocr_result).replace(" ", "").upper()
                                except:
                                    pass
                    
                    is_violation = has_no_helmet
                    
                    detection = {
                        'id': f'MV{len(all_detections)+1:03d}',
                        'bbox': (x1, y1, x2, y2),
                        'violation': is_violation,
                        'has_helmet': has_helmet,
                        'has_no_helmet': has_no_helmet,
                        'license_plate': lp_text,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'frame': frame_count
                    }
                    frame_detections.append(detection)
                    
                    # Lưu vi phạm duy nhất
                    if is_violation and lp_text != "UNKNOWN":
                        # Kiểm tra xem biển số này đã được ghi nhận chưa
                        if not any(d['license_plate'] == lp_text for d in all_detections):
                            all_detections.append(detection)
                
                # Vẽ kết quả lên frame
                frame = draw_results_on_image(frame, frame_detections)
        
        # Ghi frame vào video output
        out.write(frame)
        
        # Cập nhật progress
        progress(frame_count / total_frames, desc=f"Đang xử lý frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    # Tạo bảng thống kê
    table = create_violation_table(all_detections)
    
    # Tạo thông tin tóm tắt
    summary = f"""
    📊 **KẾT QUẢ PHÁT HIỆN VIDEO:**
    - Tổng số frame: {total_frames}
    - Số vi phạm phát hiện: {len([d for d in all_detections if d['violation']])} 🚨
    """
    
    return output_path, table, summary

# Tạo giao diện Gradio
with gr.Blocks(title="Hệ Thống Phát Hiện Vi Phạm Mũ Bảo Hiểm", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛵 HỆ THỐNG PHÁT HIỆN VI PHẠM MŨ BẢO HIỂM
        ### Sử dụng YOLO và VGG16 để phát hiện xe máy và vi phạm không đội mũ bảo hiểm
        """
    )
    
    with gr.Tabs():
        # Tab 1: Xử lý ảnh
        with gr.TabItem("📷 Phát hiện trên Ảnh"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload ảnh")
                    image_button = gr.Button("🔍 Phát hiện vi phạm", variant="primary", size="lg")
                
                with gr.Column():
                    image_output = gr.Image(label="Kết quả")
            
            with gr.Row():
                image_summary = gr.Markdown(label="Thông tin")
            
            with gr.Row():
                image_table = gr.Dataframe(
                    headers=['STT', 'Họ và tên', 'Gmail', 'Biển số', 'Thời gian', 'ID'],
                    label="📋 Danh sách vi phạm",
                    wrap=True
                )
            
            image_button.click(
                fn=process_image_detection,
                inputs=[image_input],
                outputs=[image_output, image_table, image_summary]
            )
        
        # Tab 2: Xử lý video
        with gr.TabItem("🎥 Phát hiện trên Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload video")
                    video_button = gr.Button("🔍 Phát hiện vi phạm", variant="primary", size="lg")
                
                with gr.Column():
                    video_output = gr.Video(label="Kết quả")
            
            with gr.Row():
                video_summary = gr.Markdown(label="Thông tin")
            
            with gr.Row():
                video_table = gr.Dataframe(
                    headers=['STT', 'Họ và tên', 'Gmail', 'Biển số', 'Thời gian', 'ID'],
                    label="📋 Danh sách vi phạm",
                    wrap=True
                )
            
            video_button.click(
                fn=process_video_detection,
                inputs=[video_input],
                outputs=[video_output, video_table, video_summary]
            )
        
        # Tab 3: Hướng dẫn
        with gr.TabItem("ℹ️ Hướng dẫn"):
            gr.Markdown(
                """
                ## Hướng dẫn sử dụng
                
                ### 1. Phát hiện trên ảnh
                - Upload ảnh chứa xe máy
                - Nhấn nút "Phát hiện vi phạm"
                - Xem kết quả với bounding box và bảng thống kê
                
                ### 2. Phát hiện trên video
                - Upload video giao thông
                - Nhấn nút "Phát hiện vi phạm"
                - Chờ hệ thống xử lý (có progress bar)
                - Xem video kết quả và bảng thống kê
                
                ### 3. Chú thích màu sắc
                - 🟢 **Xanh lá**: Tuân thủ (đội mũ bảo hiểm)
                - 🔴 **Đỏ**: Vi phạm (không đội mũ bảo hiểm)
                
                ### 4. Thông tin bảng
                - **STT**: Số thứ tự
                - **Biển số**: Biển số xe vi phạm (tự động OCR)
                - **Thời gian**: Thời gian phát hiện
                - **ID**: Mã định danh vi phạm
                
                ---
                **Lưu ý**: Hệ thống sử dụng AI để phát hiện, độ chính xác phụ thuộc vào chất lượng ảnh/video.
                """
            )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Khởi động giao diện web...")
    print("="*50 + "\n")
    demo.launch(
        server_name="127.0.0.1",  # Local only
        server_port=None,  # Tự động tìm port trống
        share=False,  # Đặt True nếu muốn chia sẻ public
        show_error=True
    )
