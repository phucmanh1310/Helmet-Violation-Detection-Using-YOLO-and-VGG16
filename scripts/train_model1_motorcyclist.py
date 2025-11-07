"""
Script train Model 1 (Motorcyclist Detection) với PyTorch 2.6 compatibility fix.

Fix lỗi: weights_only=True trong PyTorch 2.6
"""

import sys

# Fix PyTorch 2.6 compatibility TRƯỚC KHI import YOLO
import torch.serialization
import ultralytics.nn.tasks as tasks

original_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    """Patched torch.load để force weights_only=False"""
    return original_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)

# Monkey patch
torch.load = patched_torch_load

# Bây giờ mới import YOLO
from ultralytics import YOLO


def main():
    print("=" * 70)
    print("TRAIN MODEL 1: MOTORCYCLIST DETECTION")
    print("=" * 70)
    print()

    # Configuration
    config = {
        'model': 'yolov8n.pt',
        'data': 'data/_stage1_motorcyclist/data.yaml',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'patience': 50,
        'project': 'runs/detect',
        'name': 'model1_motorcyclist',
        'device': 0,
        'verbose': True,
        'save': True,
        'save_period': 10,
        'val': True,
        # Tránh lỗi multiprocessing trên Windows khi không có main-guard
        'workers': 2,
    }

    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    # Load model
    print("🔄 Loading pretrained model...")
    model = YOLO(config['model'])
    print("✅ Model loaded successfully!")
    print()

    # Train
    print("🚀 Starting training...")
    print("=" * 70)
    results = model.train(**config)

    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    print()
    print(f"📁 Results saved to: runs/detect/model1_motorcyclist/")
    print(f"🏆 Best model: runs/detect/model1_motorcyclist/weights/best.pt")
    print(f"📊 Metrics: runs/detect/model1_motorcyclist/results.csv")
    print()
    print("📌 Next steps:")
    print("   1. Validate model:")
    print("      yolo detect val model=runs/detect/model1_motorcyclist/weights/best.pt data=data/_stage1_motorcyclist/data.yaml")
    print()
    print("   2. Copy to models folder:")
    print("      Copy-Item 'runs/detect/model1_motorcyclist/weights/best.pt' 'models/Motov10l.pt' -Force")
    print()
    print("   3. Train Model 2:")
    print("      py -3.13 scripts/train_model2_crops.py")
    print()


if __name__ == '__main__':
    # Windows/Spawn safety for multiprocessing dataloaders
    try:
        from multiprocessing import freeze_support, set_start_method
        set_start_method('spawn', force=True)
        freeze_support()
    except Exception:
        pass
    main()
