"""
Script resume training Model 1 từ checkpoint với PyTorch 2.6 compatibility fix.
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
    print("RESUME TRAINING MODEL 1: MOTORCYCLIST DETECTION")
    print("=" * 70)
    print()

    # Tìm checkpoint mới nhất
    import glob
    import os
    
    checkpoints = glob.glob('runs/detect/model1_motorcyclist*/weights/last.pt')
    if not checkpoints:
        print("❌ Không tìm thấy checkpoint nào!")
        print("   Hãy chạy train từ đầu: py -3.13 scripts/train_model1_motorcyclist.py")
        return
    
    # Lấy checkpoint mới nhất (theo thời gian modify)
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    
    print(f"📂 Tìm thấy checkpoint: {latest_checkpoint}")
    print()
    
    # Load model từ checkpoint
    print("🔄 Loading checkpoint...")
    model = YOLO(latest_checkpoint)
    print("✅ Checkpoint loaded successfully!")
    print()
    
    # Resume training với workers=2 cho Windows
    print("🚀 Resuming training...")
    print("=" * 70)
    
    results = model.train(
        resume=True,
        workers=2,  # Quan trọng cho Windows
    )
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    print()
    print(f"📁 Results saved to: {model.trainer.save_dir}")
    print(f"🏆 Best model: {model.trainer.best}")
    print(f"📊 Last model: {model.trainer.last}")
    print()
    print("📌 Next steps:")
    print("   1. Validate model:")
    print(f"      yolo detect val model={model.trainer.best} data=data/_stage1_motorcyclist/data.yaml")
    print()
    print("   2. Copy to models folder:")
    print(f"      Copy-Item '{model.trainer.best}' 'models/Motov10l.pt' -Force")
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
