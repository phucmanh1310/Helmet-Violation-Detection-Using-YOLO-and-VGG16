"""
Filter YOLO labels to a subset of classes and optionally remap class IDs to 0..N-1.

Usage (PowerShell):

  py -3.13 scripts/filter_labels_by_classes.py `
    --src-root data/helmet-detection-and-license-plate-recognition.v6i.yolov8 `
    --dst-root data/_views/motorcyclist_only `
    --keep-classes 2 `
    --remap

This will copy images (optionally via hardlink if available) and write new labels containing
only class 2 (motorcyclist), remapped to 0.

Notes:
- Class IDs are taken from the source labels, not from data.yaml. Ensure mapping matches.
- Source structure expected:
  <src-root>/{train,valid,test}/images/*.jpg and labels/*.txt
- Destination structure created similarly, with filtered labels. Images are copied by default.
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", required=True, help="Source dataset root")
    p.add_argument("--dst-root", required=True, help="Destination root for filtered view")
    p.add_argument("--keep-classes", required=True, nargs="+", type=int, help="Class IDs to keep, e.g., 2 or 0 1 3")
    p.add_argument("--remap", action="store_true", help="Remap kept class IDs to 0..N-1 order of keep-classes")
    p.add_argument("--copy-images", action="store_true", help="Copy images to dst (default). If not, will create hardlinks when possible.")
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def filter_label_file(src_txt: Path, dst_txt: Path, keep: List[int], remap: bool):
    if not src_txt.exists():
        # No label file: write empty or skip; YOLO expects a file? Better to write empty for image with no objects
        dst_txt.write_text("")
        return
    lines_out = []
    mapping = {cid: i for i, cid in enumerate(keep)} if remap else None
    with src_txt.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid not in keep:
                continue
            if remap:
                parts[0] = str(mapping[cid])
            lines_out.append(" ".join(parts[:5]))
    dst_txt.write_text("\n".join(lines_out), encoding="utf-8")


def process_split(src_root: Path, dst_root: Path, split: str, keep: List[int], remap: bool, copy_images: bool):
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"
    ensure_dir(dst_images); ensure_dir(dst_labels)

    for img_path in src_images.glob("*.*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        rel_name = img_path.stem
        # copy image
        dst_img = dst_images / img_path.name
        if copy_images:
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
        else:
            # try hardlink
            try:
                if not dst_img.exists():
                    os.link(img_path, dst_img)  # type: ignore
            except Exception:
                shutil.copy2(img_path, dst_img)
        # filter label
        src_txt = src_labels / f"{rel_name}.txt"
        dst_txt = dst_labels / f"{rel_name}.txt"
        filter_label_file(src_txt, dst_txt, keep, remap)


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    keep = list(dict.fromkeys(args.keep_classes))

    for split in ("train", "valid", "test"):
        process_split(src_root, dst_root, split, keep, args.remap, args.copy_images)

    # write a small data.yaml for convenience
    yml = (dst_root / "data.yaml")
    names = ["class_" + str(i) for i in keep]
    if args.remap:
        # remapped to 0..N-1, provide generic names; user can edit
        names = [f"class{i}" for i in range(len(keep))]
    yml.write_text(
        "\n".join([
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            f"nc: {len(keep)}",
            f"names: {names}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Filtered view written to: {dst_root}")


if __name__ == "__main__":
    main()
