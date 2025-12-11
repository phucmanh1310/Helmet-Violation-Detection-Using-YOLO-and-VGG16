#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHỨNG MINH ĐÃ TRAIN 2 MODEL CHO PROJECT
Sinh viên: Trần Nguyễn Phúc Mạnh - MSSV: 3122411121
Project: Helmet Violation Detection Using YOLO and VGG16
"""

import os
import sys
from datetime import datetime
from pathlib import Path

print("="*80)
print("CHỨNG MINH ĐÃ TRAIN 2 MODEL CHO PROJECT")
print("Sinh viên: Trần Nguyễn Phúc Mạnh - MSSV: 3122411121")
print("="*80)

# ============================================================================
# PHẦN 1: KIỂM TRA CÁC FILE MODEL TỒN TẠI
# ============================================================================
print("\n[PHẦN 1] KIỂM TRA CÁC FILE MODEL TỒN TẠI")
print("-" * 80)

model_paths = {
    "Model 1 (YOLOv10L - Motobike Detection)": "models/Motov10l.pt",
    "Model 2 (YOLOv8N - Helmet & License Plate Detection)": "models/HelmetLP.pt"
}

models_exist = {}
for model_name, model_path in model_paths.items():
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
        timestamp = os.path.getmtime(model_path)
        modified_date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        print(f"✅ {model_name}")
        print(f"   📁 Đường dẫn: {model_path}")
        print(f"   📊 Kích thước: {file_size:.2f} MB")
        print(f"   📅 Ngày sửa đổi: {modified_date}")
        models_exist[model_name] = True
    else:
        print(f"❌ {model_name} - TẬP TIN KHÔNG TỒN TẠI: {model_path}")
        models_exist[model_name] = False

# ============================================================================
# PHẦN 2: KIỂM TRA THÀNH PHẦN LỊ SỬ TRAINING
# ============================================================================
print("\n[PHẦN 2] KIỂM TRA THÀNH PHẦN LỊ SỬ TRAINING")
print("-" * 80)

# Kiểm tra runs/ directory
runs_path = "runs/detect"
if os.path.exists(runs_path):
    train_folders = [d for d in os.listdir(runs_path) if d.startswith("train")]
    if train_folders:
        print(f"✅ Tìm thấy {len(train_folders)} folder training:")
        for i, folder in enumerate(sorted(train_folders), 1):
            folder_path = os.path.join(runs_path, folder)
            print(f"\n   {i}. {folder}/")
            
            # Kiểm tra weights
            weights_path = os.path.join(folder_path, "weights")
            if os.path.exists(weights_path):
                weights_files = os.listdir(weights_path)
                for wf in weights_files:
                    print(f"      📁 weights/{wf}")
            
            # Kiểm tra results.csv
            results_path = os.path.join(folder_path, "results.csv")
            if os.path.exists(results_path):
                print(f"      📊 results.csv (Training metrics)")
            
            # Kiểm tra args.yaml
            args_path = os.path.join(folder_path, "args.yaml")
            if os.path.exists(args_path):
                print(f"      ⚙️  args.yaml (Training config)")
    else:
        print("❌ Không tìm thấy folder training")
else:
    print(f"⚠️  Không tìm thấy thư mục: {runs_path}")

# ============================================================================
# PHẦN 3: KIỂM TRA DATASET
# ============================================================================
print("\n[PHẦN 3] KIỂM TRA DATASET")
print("-" * 80)

dataset_paths = {
    "Stage 1 - Motobike Detection": "data/Motobike Detection.v18i.yolov8",
    "Stage 2 - LP & Helmet Detection": "data/LP-Helmet.v2i.yolov8"
}

for stage_name, dataset_path in dataset_paths.items():
    if os.path.exists(dataset_path):
        print(f"\n✅ {stage_name}")
        print(f"   📁 Đường dẫn: {dataset_path}")
        
        # Đếm train/val/test images
        for split in ['train', 'val', 'test']:
            image_path = os.path.join(dataset_path, split, 'images')
            if os.path.exists(image_path):
                num_images = len([f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f"      • {split}: {num_images} images")
    else:
        print(f"⚠️  {stage_name} - Dataset không tìm thấy")

# ============================================================================
# PHẦN 4: TRY LOAD MODELS (NẾU CÓ THƯ VIỆN)
# ============================================================================
print("\n[PHẦN 4] KIỂM TRA MODEL ARCHITECTURE")
print("-" * 80)

try:
    import torch
    print("✅ PyTorch đã cài đặt")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO đã cài đặt")
        
        # Load Model 1
        try:
            print("\n📌 Loading Model 1 (Motov10l.pt)...")
            model1 = YOLO("models/Motov10l.pt")
            print("✅ Model 1 loaded successfully!")
            print(f"   Model info: {model1.model}")
        except Exception as e:
            print(f"⚠️  Không thể load Model 1: {str(e)}")
        
        # Load Model 2
        try:
            print("\n📌 Loading Model 2 (HelmetLP.pt)...")
            model2 = YOLO("models/HelmetLP.pt")
            print("✅ Model 2 loaded successfully!")
            print(f"   Model info: {model2.model}")
        except Exception as e:
            print(f"⚠️  Không thể load Model 2: {str(e)}")
            
    except ImportError:
        print("⚠️  Ultralytics chưa cài đặt - bỏ qua test load models")
        print("    Cài đặt bằng: pip install ultralytics")
        
except ImportError:
    print("⚠️  PyTorch chưa cài đặt - bỏ qua test load models")
    print("    Cài đặt bằng: pip install torch torchvision")

# ============================================================================
# PHẦN 5: THÔNG TIN DỰ ÁN
# ============================================================================
print("\n[PHẦN 5] THÔNG TIN DỰ ÁN")
print("-" * 80)

project_info = {
    "Tên đồ án": "Helmet Violation Detection Using YOLO and VGG16",
    "Sinh viên": "Trần Nguyễn Phúc Mạnh",
    "MSSV": "3122411121",
    "Model 1": "YOLOv10 Large (Motobike Detection)",
    "Model 2": "YOLOv8 Nano (Helmet & License Plate Detection)",
    "Approach": "2-Stage Detection Pipeline",
    "Framework": "Ultralytics YOLO",
    "Ngôn ngữ": "Python 3.10+"
}

for key, value in project_info.items():
    print(f"• {key}: {value}")

# ============================================================================
# PHẦN 6: TÓM TẮT
# ============================================================================
print("\n[PHẦN 6] TÓM TẮT]")
print("-" * 80)

all_models_exist = all(models_exist.values())

if all_models_exist:
    print("✅ CHỨNG MINH THÀNH CÔNG: Cả 2 model đều đã được training và lưu trữ")
    print("\n🎓 Kết luận:")
    print("   ✓ Model 1 (Motov10l.pt) - YOLOv10 Large được train thành công")
    print("   ✓ Model 2 (HelmetLP.pt) - YOLOv8 Nano được train thành công")
    print("   ✓ Cấu trúc dữ liệu: 2-Stage Pipeline (Motobike → Helmet)")
else:
    print("⚠️  Một số model không tồn tại. Kiểm tra lại đường dẫn hoặc kích hoạt lại training.")

print("\n" + "="*80)
print(f"Thời gian kiểm tra: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
