"""
Create ROI crops using a parent class (e.g., 'motorcyclist') and remap child labels inside the crop.

Typical use for your dataset (data/helmet-detection-and-license-plate-recognition.v6i.yolov8):
- Parent class id: 2 (motorcyclist)
- Child classes to keep inside ROI: [0 (helmet), 1 (licenseplate), 3 (nohelmet)]

Example (PowerShell):

  py -3.13 scripts/make_roi_crops_from_class.py `
    --src-root data/helmet-detection-and-license-plate-recognition.v6i.yolov8 `
    --dst-root data/_crops/helmet_lp_from_motorcyclist `
    --parent-class 2 `
    --keep-classes 0 1 3 `
    --remap 0 2 1

The --remap list reindexes kept classes to contiguous IDs. In example above:
  helmet(0)->0, licenseplate(1)->2, nohelmet(3)->1  (just an example; pick your mapping)
If you omit --remap, the original IDs are kept.

Output structure:
  <dst-root>/{train,valid,test}/{images,labels}

Each parent ROI becomes one cropped image containing any intersecting child boxes, whose
coordinates are converted to the crop frame.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", required=True)
    p.add_argument("--dst-root", required=True)
    p.add_argument("--parent-class", required=True, type=int, help="Class ID for ROI boxes, e.g., 2 for motorcyclist")
    p.add_argument("--keep-classes", required=True, nargs="+", type=int, help="Child classes to keep inside ROI")
    p.add_argument("--remap", nargs="*", type=int, help="Optional remap for kept classes to new IDs (same length as keep-classes)")
    p.add_argument("--padding", type=float, default=0.15, help="Padding ratio around parent box")
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    return [max(0, int(x1)), max(0, int(y1)), min(W-1, int(x2)), min(H-1, int(y2))]


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    xc = (x1 + x2) / (2 * W)
    yc = (y1 + y2) / (2 * H)
    return [max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc)), max(0.0, min(1.0, w)), max(0.0, min(1.0, h))]


def intersect(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def read_labels(txt_path: Path):
    items = []
    if not txt_path.exists():
        return items
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0]); xc, yc, w, h = map(float, parts[1:5])
            except Exception:
                continue
            items.append((cid, xc, yc, w, h))
    return items


def process_split(src_root: Path, dst_root: Path, split: str, parent_cid: int, keep_cids: list[int], remap: dict[int, int] | None, pad: float):
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"
    ensure_dir(dst_images); ensure_dir(dst_labels)

    for img_path in src_images.glob("*.*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        items = read_labels(src_labels / f"{stem}.txt")
        if not items:
            continue
        # parent boxes
        parents_xyxy = []
        for cid, xc, yc, w, h in items:
            if cid == parent_cid:
                x1,y1,x2,y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
                # padding
                pw, ph = int((x2-x1)*pad), int((y2-y1)*pad)
                x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
                x2 = min(W-1, x2 + pw); y2 = min(H-1, y2 + ph)
                parents_xyxy.append([x1,y1,x2,y2])
        if not parents_xyxy:
            continue
        # child boxes in xyxy
        children = []
        for cid, xc, yc, w, h in items:
            if cid in keep_cids:
                children.append((cid, *yolo_to_xyxy(xc, yc, w, h, W, H)))
        # crop for each parent
        for i, m in enumerate(parents_xyxy):
            mx1,my1,mx2,my2 = m
            crop = img[my1:my2, mx1:mx2]
            if crop.size == 0:
                continue
            cw, ch = (mx2-mx1), (my2-my1)
            # project children into crop
            lines = []
            for cid, x1,y1,x2,y2 in children:
                inter = intersect([x1,y1,x2,y2], m)
                if inter is None:
                    continue
                cx1, cy1, cx2, cy2 = inter
                cx1 -= mx1; cy1 -= my1; cx2 -= mx1; cy2 -= my1
                nx, ny, nw, nh = xyxy_to_yolo(cx1, cy1, cx2, cy2, cw, ch)
                if nw < 0.01 or nh < 0.01:
                    continue
                new_cid = remap[cid] if remap is not None else cid
                lines.append(f"{new_cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            if not lines:
                continue
            out_stem = f"{stem}_m{i:02d}"
            cv2.imwrite(str(dst_images / f"{out_stem}.jpg"), crop)
            (dst_labels / f"{out_stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    remap = None
    if args.remap is not None and len(args.remap) > 0:
        if len(args.remap) != len(args.keep_classes):
            raise SystemExit("--remap must have same length as --keep-classes")
        remap = {k: v for k, v in zip(args.keep_classes, args.remap)}
    for split in ("train", "valid", "test"):
        process_split(src_root, dst_root, split, args.parent_class, args.keep_classes, remap, args.padding)
    # write a data.yaml template
    names_line = "[\"helmet\", \"license_plate\", \"no_helmet\"]"
    (dst_root / "data.yaml").write_text("\n".join([
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "nc: 3",
        f"names: {names_line}",
    ]) + "\n", encoding="utf-8")
    print(f"Done. Crops saved at: {dst_root}")


if __name__ == "__main__":
    main()
