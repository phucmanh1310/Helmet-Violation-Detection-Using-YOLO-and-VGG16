"""
Script train Model 2 Option A (Full Scene) với PyTorch 2.6 compatibility fix.
"""

import sys

# Fix PyTorch 2.6 compatibility
import torch.serialization
import ultralytics.nn.tasks as tasks

original_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    """Patched torch.load để force weights_only=False"""
    return original_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)

torch.load = patched_torch_load

from ultralytics import YOLO


def main():
    print("=" * 70)
    print("TRAIN MODEL 2 OPTION A: HELMET/LP DETECTION - FULL SCENE")
    print("=" * 70)
    print()

    config = {
        'model': 'yolov8n.pt',
        'data': 'data/_stage2_helmet_lp_fullscene/data.yaml',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'patience': 50,
        'project': 'runs/detect',
        'name': 'model2_fullscene',
        'device': 0,
        'verbose': True,
        'save': True,
        'save_period': 10,
        'val': True,
        'workers': 2,
    }

    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    print("🔄 Loading pretrained model...")
    model = YOLO(config['model'])
    print("✅ Model loaded successfully!")
    print()

    print("🚀 Starting training...")
    print("=" * 70)
    results = model.train(**config)

    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    print()
    print(f"📁 Results: runs/detect/model2_fullscene/")
    print(f"🏆 Best model: runs/detect/model2_fullscene/weights/best.pt")
    print()


if __name__ == '__main__':
    try:
        from multiprocessing import freeze_support, set_start_method
        set_start_method('spawn', force=True)
        freeze_support()
    except Exception:
        pass
    main()
