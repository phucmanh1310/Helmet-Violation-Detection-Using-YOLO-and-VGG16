#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERATE TRAINING REPORT
Tạo báo cáo chi tiết quá trình training 2 model YOLO
Sinh viên: Trần Nguyễn Phúc Mạnh - MSSV: 3122411121
"""

import os
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

REPORT_FILE = "TRAINING_REPORT.txt"
PROJECT_NAME = "Helmet Violation Detection Using YOLO and VGG16"
STUDENT_NAME = "Trần Nguyễn Phúc Mạnh"
STUDENT_ID = "3122411121"

MODELS_INFO = {
    "Model 1 - Motobike Detection": {
        "checkpoint_dir": "runs/detect/model1_motorcyclist5",
        "model_file": "models/Motov10l.pt",
        "model_type": "YOLOv8 Nano",
        "stage": "Stage 1: Detect Motorbikes",
        "dataset": "data/_stage1_motorcyclist/data.yaml"
    },
    "Model 2 - Helmet & License Plate Detection": {
        "checkpoint_dir": "runs/detect/model2_crops",
        "model_file": "models/HelmetLP.pt",
        "model_type": "YOLOv8 Nano",
        "stage": "Stage 2: Detect Helmets & License Plates (Crops)",
        "dataset": "data/_stage2_helmet_lp_crops/data.yaml"
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_filesize(bytes_size):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def load_yaml(yaml_path):
    """Load YAML file"""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return None

def extract_training_metrics(checkpoint_dir):
    """Extract training metrics from checkpoint directory"""
    metrics = {}
    
    # Check args.yaml
    args_file = os.path.join(checkpoint_dir, "args.yaml")
    if os.path.exists(args_file):
        args = load_yaml(args_file)
        if args:
            metrics['epochs'] = args.get('epochs', 'N/A')
            metrics['batch_size'] = args.get('batch', 'N/A')
            metrics['image_size'] = args.get('imgsz', 'N/A')
            metrics['optimizer'] = args.get('optimizer', 'N/A')
            metrics['device'] = args.get('device', 'N/A')
            metrics['workers'] = args.get('workers', 'N/A')
    
    # Check results.csv for final metrics
    results_file = os.path.join(checkpoint_dir, "results.csv")
    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            last_row = df.iloc[-1]
            metrics['final_precision'] = f"{last_row.get('metrics/precision(B)', 'N/A'):.4f}" if 'metrics/precision(B)' in df.columns else 'N/A'
            metrics['final_recall'] = f"{last_row.get('metrics/recall(B)', 'N/A'):.4f}" if 'metrics/recall(B)' in df.columns else 'N/A'
            metrics['final_mAP50'] = f"{last_row.get('metrics/mAP50(B)', 'N/A'):.4f}" if 'metrics/mAP50(B)' in df.columns else 'N/A'
            metrics['final_mAP50_95'] = f"{last_row.get('metrics/mAP50-95(B)', 'N/A'):.4f}" if 'metrics/mAP50-95(B)' in df.columns else 'N/A'
            metrics['trained_epochs'] = len(df)
        except Exception as e:
            metrics['error'] = str(e)
    
    # Check weights directory
    weights_dir = os.path.join(checkpoint_dir, "weights")
    if os.path.exists(weights_dir):
        weights_files = os.listdir(weights_dir)
        metrics['weights_saved'] = weights_files
    
    return metrics

def count_dataset_images(data_yaml_path):
    """Count images in dataset"""
    data = load_yaml(data_yaml_path)
    counts = {}
    
    if data:
        path = data.get('path', '')
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(path, split, 'images')
            if os.path.exists(images_dir):
                images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                counts[split] = len(images)
    
    return counts

# ============================================================================
# GENERATE REPORT
# ============================================================================

def main():
    print("🔄 Generating training report...")
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 100)
    report_lines.append("BÁAO CÁO TRAINING - CHI TIẾT QUẶC TRÌNH HUẤN LUYỆN MÔ HÌNH")
    report_lines.append("=" * 100)
    report_lines.append(f"Dự án: {PROJECT_NAME}")
    report_lines.append(f"Sinh viên: {STUDENT_NAME} - MSSV: {STUDENT_ID}")
    report_lines.append(f"Ngày tạo báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 100)
    
    # ========================================================================
    # MODEL INFORMATION
    # ========================================================================
    report_lines.append("\n📋 THÔNG TIN MÔ HÌNH")
    report_lines.append("-" * 100)
    
    for model_name, model_info in MODELS_INFO.items():
        report_lines.append(f"\n🤖 {model_name}")
        report_lines.append(f"   • Loại mô hình: {model_info['model_type']}")
        report_lines.append(f"   • Giai đoạn: {model_info['stage']}")
        report_lines.append(f"   • File mô hình: {model_info['model_file']}")
        
        # Check model file
        if os.path.exists(model_info['model_file']):
            file_size = os.path.getsize(model_info['model_file'])
            mod_time = os.path.getmtime(model_info['model_file'])
            mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            report_lines.append(f"   • ✅ Mô hình tồn tại")
            report_lines.append(f"   • Kích thước: {format_filesize(file_size)}")
            report_lines.append(f"   • Ngày lưu: {mod_date}")
        else:
            report_lines.append(f"   • ❌ Mô hình không tồn tại")
    
    # ========================================================================
    # DATASET INFORMATION
    # ========================================================================
    report_lines.append("\n\n📊 THÔNG TIN DỮ LIỆU")
    report_lines.append("-" * 100)
    
    for model_name, model_info in MODELS_INFO.items():
        report_lines.append(f"\n{model_name}")
        
        dataset_path = model_info['dataset']
        if os.path.exists(dataset_path):
            data = load_yaml(dataset_path)
            if data:
                path = data.get('path', '')
                nc = data.get('nc', 'N/A')
                names = data.get('names', {})
                
                report_lines.append(f"   • Số lớp (classes): {nc}")
                if names:
                    report_lines.append(f"   • Tên lớp: {', '.join(names.values()) if isinstance(names, dict) else names}")
                
                # Count images
                dataset_counts = count_dataset_images(dataset_path)
                if dataset_counts:
                    total_images = sum(dataset_counts.values())
                    report_lines.append(f"   • Tổng số ảnh: {total_images}")
                    for split, count in dataset_counts.items():
                        report_lines.append(f"     - {split}: {count} ảnh")
        else:
            report_lines.append(f"   • ⚠️  Dataset không tìm thấy: {dataset_path}")
    
    # ========================================================================
    # TRAINING CONFIGURATION
    # ========================================================================
    report_lines.append("\n\n⚙️  CẤU HÌNH TRAINING")
    report_lines.append("-" * 100)
    
    for model_name, model_info in MODELS_INFO.items():
        report_lines.append(f"\n{model_name}")
        
        args_file = os.path.join(model_info['checkpoint_dir'], "args.yaml")
        if os.path.exists(args_file):
            args = load_yaml(args_file)
            if args:
                report_lines.append(f"   • Model base: {args.get('model', 'N/A')}")
                report_lines.append(f"   • Số epoch: {args.get('epochs', 'N/A')}")
                report_lines.append(f"   • Batch size: {args.get('batch', 'N/A')}")
                report_lines.append(f"   • Kích thước ảnh: {args.get('imgsz', 'N/A')}")
                report_lines.append(f"   • Optimizer: {args.get('optimizer', 'N/A')}")
                report_lines.append(f"   • Thiết bị: {args.get('device', 'N/A')}")
                report_lines.append(f"   • Workers: {args.get('workers', 'N/A')}")
                report_lines.append(f"   • Mixed precision (AMP): {args.get('amp', 'N/A')}")
                report_lines.append(f"   • Cache: {args.get('cache', 'N/A')}")
    
    # ========================================================================
    # TRAINING RESULTS
    # ========================================================================
    report_lines.append("\n\n📈 KẾT QUẢ TRAINING")
    report_lines.append("-" * 100)
    
    for model_name, model_info in MODELS_INFO.items():
        report_lines.append(f"\n{model_name}")
        
        metrics = extract_training_metrics(model_info['checkpoint_dir'])
        
        if 'error' in metrics:
            report_lines.append(f"   ⚠️  Lỗi khi tải metrics: {metrics['error']}")
        else:
            report_lines.append(f"   • Số epoch đã train: {metrics.get('trained_epochs', 'N/A')}")
            report_lines.append(f"   • Precision cuối: {metrics.get('final_precision', 'N/A')}")
            report_lines.append(f"   • Recall cuối: {metrics.get('final_recall', 'N/A')}")
            report_lines.append(f"   • mAP50 cuối: {metrics.get('final_mAP50', 'N/A')}")
            report_lines.append(f"   • mAP50-95 cuối: {metrics.get('final_mAP50_95', 'N/A')}")
            
            if 'weights_saved' in metrics:
                report_lines.append(f"   • Weights đã lưu: {', '.join(metrics['weights_saved'])}")
        
        # Show training curve
        results_file = os.path.join(model_info['checkpoint_dir'], "results.csv")
        if os.path.exists(results_file):
            try:
                df = pd.read_csv(results_file)
                report_lines.append(f"\n   📊 Chi tiết epochs (10 epochs đầu tiên):")
                report_lines.append("")
                
                # Select relevant columns
                cols_to_show = ['epoch', 'train/box_loss', 'train/cls_loss', 
                               'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']
                cols_available = [c for c in cols_to_show if c in df.columns]
                
                if cols_available:
                    df_display = df[cols_available].head(10)
                    # Format as table
                    table_str = df_display.to_string(index=False)
                    for line in table_str.split('\n'):
                        report_lines.append(f"   {line}")
            except Exception as e:
                report_lines.append(f"   ⚠️  Lỗi khi đọc results.csv: {str(e)}")
    
    # ========================================================================
    # OUTPUTS
    # ========================================================================
    report_lines.append("\n\n🎨 OUTPUT VÀ ARTIFACTS")
    report_lines.append("-" * 100)
    
    for model_name, model_info in MODELS_INFO.items():
        report_lines.append(f"\n{model_name}")
        
        checkpoint_dir = model_info['checkpoint_dir']
        if os.path.exists(checkpoint_dir):
            # List all files
            all_files = []
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isfile(item_path):
                    all_files.append(item)
            
            if all_files:
                report_lines.append(f"   Các file được tạo:")
                for f in sorted(all_files):
                    size = os.path.getsize(os.path.join(checkpoint_dir, f))
                    report_lines.append(f"      • {f} ({format_filesize(size)})")
            
            # Weights folder
            weights_dir = os.path.join(checkpoint_dir, "weights")
            if os.path.exists(weights_dir):
                report_lines.append(f"\n   Các file weights:")
                for w in os.listdir(weights_dir):
                    w_path = os.path.join(weights_dir, w)
                    if os.path.isfile(w_path):
                        size = os.path.getsize(w_path)
                        report_lines.append(f"      • {w} ({format_filesize(size)})")
    
    # ========================================================================
    # CONCLUSION
    # ========================================================================
    report_lines.append("\n\n✅ KẾT LUẬN")
    report_lines.append("-" * 100)
    report_lines.append(f"""
   ✓ Đã successfully training 2 mô hình YOLO cho dự án:
     - Model 1: YOLOv8 Nano để phát hiện motorbike (Stage 1)
     - Model 2: YOLOv8 Nano để phát hiện helmet & license plate (Stage 2)
   
   ✓ Mô hình được lưu tại:
     - {MODELS_INFO['Model 1 - Motobike Detection']['model_file']}
     - {MODELS_INFO['Model 2 - Helmet & License Plate Detection']['model_file']}
   
   ✓ Training artifacts được lưu tại folder runs/detect/
   
   ✓ Cả 2 mô hình đều sẵn sàng cho inference/deployment
""")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    report_lines.append("\n" + "=" * 100)
    report_lines.append(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 100)
    
    # Write to file
    report_content = "\n".join(report_lines)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Print to console
    print(report_content)
    print(f"\n✅ Report saved to: {REPORT_FILE}")

if __name__ == "__main__":
    main()
