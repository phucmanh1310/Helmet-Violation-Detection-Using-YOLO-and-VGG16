# Consolidated Main Application for Helmet Violation Detection (Full Local)

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from pathlib import Path
import shutil
import argparse

# --- Configuration ---
# Paths to local YOLOv8 models
MOTO_MODEL_PATH = Path('models/Motov10l.pt') # Use the standard YOLOv8 model for better general detection
HELMET_LP_MODEL_PATH = Path('models/HelmetLP.pt')

# Confidence thresholds
MOTO_CONF = 0.4 # Lowered threshold to increase sensitivity
HELMET_LP_CONF = 0.4

# --- Helper Functions ---

def clean_temp_folders():
    """Cleans the temporary image folders."""
    print("🧹 Cleaning temporary folders...")
    for folder in ['img/Moto_Crops', 'img/LP_Crops']:
        p = Path(folder)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

def initialize_models():
    """Loads and initializes all required models."""
    print("🔄 Initializing models...")
    try:
        # The standard 'yolov8n.pt' model will be downloaded automatically, so no need to check for existence.
        # if not MOTO_MODEL_PATH.exists():
        #     raise FileNotFoundError(f"Motorcycle detection model not found at: {MOTO_MODEL_PATH}")
        if not HELMET_LP_MODEL_PATH.exists():
            raise FileNotFoundError(f"Helmet/LP detection model not found at: {HELMET_LP_MODEL_PATH}")
        
        moto_model = YOLO(MOTO_MODEL_PATH)
        helmet_lp_model = YOLO(HELMET_LP_MODEL_PATH)
        reader = easyocr.Reader(['en'], gpu=True) # gpu=False if you don't have a compatible GPU
        print("✅ Models initialized successfully.")
        return moto_model, helmet_lp_model, reader
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return None, None, None

# --- Core Processing Logic ---

def process_image(image_path, moto_model, helmet_lp_model, reader):
    """Processes a single image to detect helmet violations."""
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"\n🖼️  Processing image: {Path(image_path).name}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print("❌ Could not read the image.")
        return

    # 1. Detect Motorcycles
    moto_results = moto_model.predict(frame, conf=MOTO_CONF, verbose=False)
    
    if not moto_results or len(moto_results[0].boxes) == 0:
        print("-> No motorcycles detected.")
        return

    print(f"-> Found {len(moto_results[0].boxes)} potential motorcycle(s).")
    total_violations = 0

    # 2. Process each detected motorcycle
    for i, box in enumerate(moto_results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        moto_crop = frame[y1:y2, x1:x2]
        cv2.imwrite(f"img/Moto_Crops/moto_{i}.jpg", moto_crop)

        # 3. Detect Helmet/LP on the motorcycle crop
        helmet_lp_results = helmet_lp_model.predict(moto_crop, conf=HELMET_LP_CONF, verbose=False)[0]
        
        # Get class names from the model
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
                    # Crop the license plate
                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, det_box.xyxy[0])
                    lp_crop = moto_crop[lp_y1:lp_y2, lp_x1:lp_x2]
                    cv2.imwrite(f"img/LP_Crops/lp_{i}.jpg", lp_crop)
                    
                    # 4. Use EasyOCR to read the license plate
                    try:
                        ocr_result = reader.readtext(lp_crop, detail=0, paragraph=False)
                        if ocr_result:
                            lp_text = ' '.join(ocr_result).replace(" ", "").upper()
                    except Exception as e:
                        print(f"- OCR Error: {e}")

        # 5. Determine and log the result
        if has_no_helmet:
            print(f"  🚨 VIOLATION on motorcycle #{i}: No helmet detected. LP: {lp_text}")
            total_violations += 1
        elif has_helmet:
            print(f"  ✅ Compliance on motorcycle #{i}: Helmet detected. LP: {lp_text}")
        else:
            print(f"  ⚠️ Could not determine helmet status for motorcycle #{i}. LP: {lp_text}")

    print(f"\n🏁 Finished processing. Found {total_violations} violation(s) in the image.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet Violation Detection (Full Local)")
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--clean', action='store_true', help='Clean temporary folders before running.')
    args = parser.parse_args()

    if args.clean:
        clean_temp_folders()

    # Initialize models
    moto_model, helmet_lp_model, reader = initialize_models()

    if all([moto_model, helmet_lp_model, reader]):
        # Create temp folders if they don't exist
        Path('img/Moto_Crops').mkdir(parents=True, exist_ok=True)
        Path('img/LP_Crops').mkdir(parents=True, exist_ok=True)
        
        process_image(args.image, moto_model, helmet_lp_model, reader)

