"""
Script train Model 2 Option B (ROI Crops - RECOMMENDED) với PyTorch 2.6 compatibility fix.
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
    print("TRAIN MODEL 2 OPTION B: HELMET/LP DETECTION - ROI CROPS (RECOMMENDED)")
    print("=" * 70)
    print()

    config = {
        'model': 'yolov8n.pt',
        'data': 'data/_stage2_helmet_lp_crops/data.yaml',
        'epochs': 50,
        'imgsz': 768,  # Higher resolution for small objects
        'batch': 16,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'patience': 50,
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        # Other
        'project': 'runs/detect',
        'name': 'model2_crops',
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
    print("⚠️  This may take 4-6 hours on RTX 3060")
    print("=" * 70)
    results = model.train(**config)

    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    print()
    print(f"📁 Results: runs/detect/model2_crops/")
    print(f"🏆 Best model: runs/detect/model2_crops/weights/best.pt")
    print()
    print("📌 Next steps:")
    print("   1. Validate:")
    print("      yolo detect val model=runs/detect/model2_crops/weights/best.pt data=data/_stage2_helmet_lp_crops/data.yaml")
    print()
    print("   2. Compare with Full Scene version")
    print()
    print("   3. Copy best model:")
    print("      Copy-Item 'runs/detect/model2_crops/weights/best.pt' 'models/HelmetLP.pt' -Force")
    print()


if __name__ == '__main__':
    try:
        from multiprocessing import freeze_support, set_start_method
        set_start_method('spawn', force=True)
        freeze_support()
    except Exception:
        pass
    main()
