"""
Script để merge 3 datasets và chuẩn hóa class names, sau đó tạo 2 views cho 2-stage training.

Datasets hiện tại:
1. Helmet_detect.v2i.yolov8: ['Helmet', 'Motorcyclist', 'Non_helmet', 'Plate']
2. Testing_AIP.v1i.yolov8: ['Helmet', 'LicensePlate', 'Motorcyclist', 'Nohelmet']
3. helmet-detection-and-license-plate-recognition.v6i.yolov8: ['helmet', 'licenseplate', 'motorcyclist', 'nohelmet']

Output chuẩn hóa:
- Class 0: helmet
- Class 1: nohelmet  
- Class 2: motorcyclist
- Class 3: licenseplate

Usage:
    py -3.13 scripts/merge_and_prepare_datasets.py
"""

import shutil
from pathlib import Path
from collections import defaultdict
import random

# Class mapping cho từng dataset
DATASET_MAPPINGS = {
    "Helmet_detect.v2i.yolov8": {
        # ['Helmet', 'Motorcyclist', 'Non_helmet', 'Plate']
        0: 0,  # Helmet -> helmet
        1: 2,  # Motorcyclist -> motorcyclist
        2: 1,  # Non_helmet -> nohelmet
        3: 3,  # Plate -> licenseplate
    },
    "Testing_AIP.v1i.yolov8": {
        # ['Helmet', 'LicensePlate', 'Motorcyclist', 'Nohelmet']
        0: 0,  # Helmet -> helmet
        1: 3,  # LicensePlate -> licenseplate
        2: 2,  # Motorcyclist -> motorcyclist
        3: 1,  # Nohelmet -> nohelmet
    },
    "helmet-detection-and-license-plate-recognition.v6i.yolov8": {
        # ['helmet', 'licenseplate', 'motorcyclist', 'nohelmet']
        0: 0,  # helmet -> helmet
        1: 3,  # licenseplate -> licenseplate
        2: 2,  # motorcyclist -> motorcyclist
        3: 1,  # nohelmet -> nohelmet
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def remap_label_line(line: str, mapping: dict) -> str:
    """Remap class ID trong 1 dòng label"""
    parts = line.strip().split()
    if len(parts) < 5:
        return ""
    try:
        old_cid = int(parts[0])
        new_cid = mapping.get(old_cid, old_cid)
        parts[0] = str(new_cid)
        return " ".join(parts[:5])
    except Exception:
        return ""


def merge_dataset(src_datasets: list, dst_root: Path):
    """Merge nhiều datasets thành 1, chuẩn hóa class IDs"""
    
    print("=" * 70)
    print("BƯỚC 1: MERGE VÀ CHUẨN HÓA DATASETS")
    print("=" * 70)
    
    # Counter để tránh trùng tên file
    file_counter = defaultdict(int)
    
    for dataset_name in src_datasets:
        dataset_path = Path("data") / dataset_name
        if not dataset_path.exists():
            print(f"⚠️  Dataset không tồn tại: {dataset_path}")
            continue
            
        print(f"\n📦 Đang xử lý: {dataset_name}")
        mapping = DATASET_MAPPINGS.get(dataset_name, {})
        
        for split in ["train", "valid", "test"]:
            src_images = dataset_path / split / "images"
            src_labels = dataset_path / split / "labels"
            
            if not src_images.exists():
                print(f"  ⚠️  Không tìm thấy: {split}/images")
                continue
                
            dst_images = dst_root / split / "images"
            dst_labels = dst_root / split / "labels"
            ensure_dir(dst_images)
            ensure_dir(dst_labels)
            
            img_count = 0
            for img_path in src_images.glob("*.*"):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                    
                stem = img_path.stem
                
                # Tạo tên file unique
                unique_name = f"{dataset_name.split('.')[0]}_{stem}_{file_counter[stem]:03d}"
                file_counter[stem] += 1
                
                # Copy image
                dst_img = dst_images / f"{unique_name}{img_path.suffix}"
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                
                # Remap và copy label
                src_label = src_labels / f"{stem}.txt"
                dst_label = dst_labels / f"{unique_name}.txt"
                
                if src_label.exists():
                    with src_label.open("r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    remapped_lines = []
                    for line in lines:
                        remapped = remap_label_line(line, mapping)
                        if remapped:
                            remapped_lines.append(remapped)
                    
                    dst_label.write_text("\n".join(remapped_lines), encoding="utf-8")
                else:
                    # Tạo file rỗng nếu không có label
                    dst_label.write_text("", encoding="utf-8")
                
                img_count += 1
            
            print(f"  ✅ {split}: {img_count} images")
    
    # Tạo data.yaml
    yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 4
names: ['helmet', 'nohelmet', 'motorcyclist', 'licenseplate']

# Merged from:
# - Helmet_detect.v2i.yolov8
# - Testing_AIP.v1i.yolov8
# - helmet-detection-and-license-plate-recognition.v6i.yolov8
"""
    (dst_root / "data.yaml").write_text(yaml_content, encoding="utf-8")
    
    print(f"\n✅ Merged dataset saved to: {dst_root}")
    print(f"✅ data.yaml created with standardized classes:")
    print(f"   0: helmet")
    print(f"   1: nohelmet")
    print(f"   2: motorcyclist")
    print(f"   3: licenseplate")


def create_stage1_view(merged_root: Path, stage1_root: Path):
    """Tạo view chỉ có motorcyclist class (id=2) cho Stage 1"""
    
    print("\n" + "=" * 70)
    print("BƯỚC 2: TẠO VIEW CHO STAGE 1 (MOTORCYCLIST ONLY)")
    print("=" * 70)
    
    for split in ["train", "valid", "test"]:
        src_images = merged_root / split / "images"
        src_labels = merged_root / split / "labels"
        dst_images = stage1_root / split / "images"
        dst_labels = stage1_root / split / "labels"
        
        ensure_dir(dst_images)
        ensure_dir(dst_labels)
        
        count = 0
        for img_path in src_images.glob("*.*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            
            stem = img_path.stem
            src_label = src_labels / f"{stem}.txt"
            
            # Lọc chỉ class 2 (motorcyclist) và remap về 0
            if src_label.exists():
                with src_label.open("r", encoding="utf-8") as f:
                    lines = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5 and int(parts[0]) == 2:
                            # Remap class 2 -> 0
                            parts[0] = "0"
                            lines.append(" ".join(parts[:5]))
                
                if lines:  # Chỉ copy nếu có motorcyclist
                    # Copy image (hardlink nếu được)
                    dst_img = dst_images / img_path.name
                    try:
                        dst_img.hardlink_to(img_path)
                    except Exception:
                        shutil.copy2(img_path, dst_img)
                    
                    # Write filtered label
                    (dst_labels / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                    count += 1
        
        print(f"  ✅ {split}: {count} images with motorcyclist")
    
    # Tạo data.yaml cho Stage 1
    yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['motorcyclist']
"""
    (stage1_root / "data.yaml").write_text(yaml_content, encoding="utf-8")
    
    print(f"\n✅ Stage 1 view saved to: {stage1_root}")


def main():
    # Paths
    data_dir = Path("data")
    merged_dir = data_dir / "_merged_all"
    stage1_dir = data_dir / "_stage1_motorcyclist"
    
    # Datasets cần merge
    datasets_to_merge = [
        "Helmet_detect.v2i.yolov8",
        "Testing_AIP.v1i.yolov8",
        "helmet-detection-and-license-plate-recognition.v6i.yolov8",
    ]
    
    # Step 1: Merge
    merge_dataset(datasets_to_merge, merged_dir)
    
    # Step 2: Create Stage 1 view
    create_stage1_view(merged_dir, stage1_dir)
    
    print("\n" + "=" * 70)
    print("HOÀN TẤT! TIẾP THEO:")
    print("=" * 70)
    print("\n1️⃣  Tạo Stage 2 ROI crops:")
    print("    py -3.13 scripts/make_roi_crops_from_class.py \\")
    print("      --src-root data/_merged_all \\")
    print("      --dst-root data/_stage2_helmet_lp_crops \\")
    print("      --parent-class 2 \\")
    print("      --keep-classes 0 1 3 \\")
    print("      --remap 0 1 2 \\")
    print("      --padding 0.18")
    
    print("\n2️⃣  Train Stage 1 (Motorcyclist):")
    print("    yolo detect train model=yolov8n.pt \\")
    print("      data=data/_stage1_motorcyclist/data.yaml \\")
    print("      epochs=100 imgsz=640 batch=16")
    
    print("\n3️⃣  Train Stage 2 (Helmet/NoHelmet/LP):")
    print("    yolo detect train model=yolov8n.pt \\")
    print("      data=data/_stage2_helmet_lp_crops/data.yaml \\")
    print("      epochs=150 imgsz=768 batch=16")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
