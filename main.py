"""
Main script để chạy Helmet Violation Detection
Sử dụng: python main.py --video path/to/video.mp4
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Thêm thư mục Source vào Python path
current_dir = Path(__file__).parent
source_dir = current_dir / "Source"
sys.path.append(str(source_dir))

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import pandas as pd
    import easyocr

    # Import các module từ Source
    import _Motobike
    import _LP_Helmet
    import _ReadLP
    import _myFunc
except ImportError as e:
    print(f"Lỗi import thư viện: {e}")
    print("Vui lòng cài đặt các thư viện cần thiết bằng lệnh: pip install -r requirements.txt")
    sys.exit(1)

def setup_directories():
    """Tạo các thư mục cần thiết"""
    directories = [
        "models",
        "img/Traffic",
        "img/Moto",
        "config",
        "data"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✅ Đã tạo các thư mục cần thiết")

def check_models():
    """Kiểm tra xem các model cần thiết có tồn tại không"""
    required_models = [
        "models/Motov10l.pt"  # YOLO model để detect xe máy
    ]

    missing_models = []
    for model_path in required_models:
        if not Path(model_path).exists():
            missing_models.append(model_path)

    if missing_models:
        print("❌ Thiếu các model sau:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nVui lòng tải các model từ link trong README và đặt vào thư mục 'models/'")
        return False

    print("✅ Tất cả model đã sẵn sàng")
    return True

def check_config():
    """Kiểm tra file config cho Roboflow API"""
    config_path = Path("config/roboflow_config.json")

    if not config_path.exists():
        # Tạo file config mẫu
        sample_config = {
            "ROBOFLOW_API_KEY": "your_api_key_here",
            "ROBOFLOW_WORKSPACE_ID": "cdio-zmfmj",
            "ROBOFLOW_PROJECT_ID": "helmet-lincense-plate-detection-gevlq",
            "ROBOFLOW_VERSION_NUMBER": 7,
            "ROBOFLOW_SIZE": 640
        }

        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)

        print(f"❌ Đã tạo file config mẫu tại {config_path}")
        print("Vui lòng cập nhật API key của bạn trong file này")
        return False

    print("✅ File config đã tồn tại")
    return True

def process_video(video_path, start_frame=0, stop_frame=0):
    """Xử lý video để phát hiện vi phạm mũ bảo hiểm"""
def clean_outputs():
    # Xóa thư mục runs/detect và ảnh tạm img/
    try:
        import shutil
        if Path('runs/detect').exists():
            shutil.rmtree('runs/detect', ignore_errors=True)
        for sub in ['img/Traffic', 'img/Moto', 'img/LP', 'img/Thresh_result']:
            p = Path(sub)
            if p.exists():
                for f in p.glob('*'):
                    try:
                        f.unlink()
                    except Exception:
                        pass
        print('🧹 Đã dọn sạch các file tạm và thư mục runs/detect')
    except Exception as e:
        print(f'⚠️ Lỗi khi dọn dẹp: {e}')

def process_image(image_path: str):
    if not Path(image_path).exists():
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return 0

    print(f"🖼️ Xử lý ảnh: {image_path}")
    # Load models
    try:
        detect_moto_model = YOLO(str(Path('models/Motov10l.pt')))
        # Roboflow
        with open('config/roboflow_config.json') as f:
            config = json.load(f)
        from roboflow import Roboflow
        rf = Roboflow(api_key=config["ROBOFLOW_API_KEY"])
        version = rf.workspace(config["ROBOFLOW_WORKSPACE_ID"]).project(config["ROBOFLOW_PROJECT_ID"]).version(config["ROBOFLOW_VERSION_NUMBER"])
        detect_lp_model = version.model
        # EasyOCR
        import easyocr
        try:
            import torch
            reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        except Exception:
            reader = easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        print(f"❌ Lỗi khi load models: {e}")
        return 0

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không đọc được ảnh")
        return 0

    # Detect motobike and crop
    _myFunc.FilePreProcess('img/Moto')
    _myFunc.FilePreProcess('img/Traffic')
    _myFunc.FilePreProcess('img/LP')
    _myFunc.FilePreProcess('img/Thresh_result')

    _Motobike.image_detect(detect_moto_model, img, screen_threshold=0.1, index_img=0)

    violations_found = 0
    for filename in os.listdir('img/Moto'):
        if filename.startswith('image0_'):
            motobike_path = Path('img/Moto') / filename
            chartest, plate_img, helmet_img, nohelmet_img = _LP_Helmet.image_detect(
                detect_lp_model,
                str(motobike_path)
            )
            lp_info = 'UNKNOWN'
            if isinstance(plate_img, np.ndarray) and plate_img.size > 0:
                try:
                    texts = reader.readtext(plate_img, detail=False, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ')
                    lp_info = texts[0] if texts else 'UNKNOWN'
                except Exception:
                    lp_info = 'UNKNOWN'

            if isinstance(nohelmet_img, np.ndarray) and nohelmet_img.size > 0:
                violations_found += 1
                print(f"🚨 Vi phạm: Biển số {lp_info} - Không đội mũ bảo hiểm")
            elif isinstance(helmet_img, np.ndarray) and helmet_img.size > 0:
                print(f"✅ Tuân thủ: Biển số {lp_info} - Có đội mũ bảo hiểm")

    print(f"🏁 Ảnh {Path(image_path).name}: tìm thấy {violations_found} vi phạm")
    return violations_found


    if not Path(video_path).exists():
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    print(f"🎥 Bắt đầu xử lý video: {video_path}")

    # Khởi tạo models (cần được sửa đổi để sử dụng đường dẫn tương đối)
    try:
        # Load YOLO model
        moto_model_path = Path("models/Motov10l.pt")
        if moto_model_path.exists():
            detect_moto_model = YOLO(str(moto_model_path))
            print("✅ Đã load YOLO model")
        else:
            print("❌ Không tìm thấy YOLO model")
            return

        # Khởi tạo EasyOCR reader (không cần .h5)
        gpu = False
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        reader = easyocr.Reader(['en'], gpu=gpu)

        # Load Roboflow model (cần API key)
        config_path = Path("config/roboflow_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            if config["ROBOFLOW_API_KEY"] == "your_api_key_here":
                print("❌ Vui lòng cập nhật API key trong config/roboflow_config.json")
                return

            from roboflow import Roboflow
            rf = Roboflow(api_key=config["ROBOFLOW_API_KEY"])
            workspace = rf.workspace(config["ROBOFLOW_WORKSPACE_ID"])
            project = workspace.project(config["ROBOFLOW_PROJECT_ID"])
            version = project.version(config["ROBOFLOW_VERSION_NUMBER"])
            detect_lp_model = version.model
            print("✅ Đã kết nối Roboflow model")
        else:
            print("❌ Không tìm thấy config file")
            return

    except Exception as e:
        print(f"❌ Lỗi khi load models: {e}")
        return

    # Bắt đầu xử lý video
    cap = cv2.VideoCapture(video_path)
    seconds_interval = 0.7
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_extract = int(fps * seconds_interval)
    current_frame = 0
    frame_count = 0

    # Xóa các file cũ
    traffic_dir = Path("img/Traffic")
    moto_dir = Path("img/Moto")
    _myFunc.delete_files(str(traffic_dir))
    _myFunc.delete_files(str(moto_dir))

    violations_found = 0

    print("🎬 Bắt đầu phân tích video...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count == 0:
            if stop_frame != 0 and current_frame >= stop_frame:
                break

            index_img = current_frame // frames_to_extract

            # Lưu frame gốc
            image_traffic_path = traffic_dir / f"image{index_img:04d}.jpg"
            cv2.imwrite(str(image_traffic_path), frame)

            # Detect xe máy
            try:
                motobike_img = _Motobike.image_detect(
                    detect_moto_model,
                    frame,
                    screen_threshold=0.1,
                    index_img=index_img
                )

                # Xử lý từng xe máy được phát hiện
                for filename in os.listdir(moto_dir):
                    if filename.startswith(f"image{index_img}_"):
                        motobike_path = moto_dir / filename

                        # Detect mũ bảo hiểm và biển số
                        chartest, plate_img, helmet_img, nohelmet_img = _LP_Helmet.image_detect(
                            detect_lp_model,
                            str(motobike_path)
                        )

                        # Đọc biển số nếu có ảnh plate, nếu không gán UNKNOWN
                        lp_info = "UNKNOWN"
                        if isinstance(plate_img, np.ndarray) and plate_img.size > 0:
                            try:
                                lp_texts = reader.readtext(plate_img, detail=False, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ')
                                lp_info = lp_texts[0] if lp_texts else "UNKNOWN"
                            except Exception:
                                lp_info = "UNKNOWN"

                        # Kiểm tra vi phạm (ưu tiên no-helmet). Nếu chỉ có helmet => tuân thủ
                        if isinstance(nohelmet_img, np.ndarray) and nohelmet_img.size > 0:
                            violations_found += 1
                            print(f"🚨 Vi phạm #{violations_found}: Biển số {lp_info} - Không đội mũ bảo hiểm")
                        elif isinstance(helmet_img, np.ndarray) and helmet_img.size > 0:
                            print(f"✅ Tuân thủ: Biển số {lp_info} - Có đội mũ bảo hiểm")

            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý frame {current_frame}: {e}")

        current_frame += 1
        frame_count = (frame_count + 1) % frames_to_extract

        # Hiển thị tiến độ
        if current_frame % (fps * 5) == 0:  # Mỗi 5 giây
            print(f"📊 Đã xử lý {current_frame} frames, tìm thấy {violations_found} vi phạm")

    cap.release()

    print(f"🏁 Hoàn thành! Tổng cộng tìm thấy {violations_found} vi phạm trong video")
    return violations_found

def main():
    parser = argparse.ArgumentParser(description='Helmet Violation Detection System')
    parser.add_argument('--video', '-v', help='Đường dẫn đến file video')
    parser.add_argument('--image', '-i', help='Đường dẫn đến file ảnh để test nhanh')
    parser.add_argument('--clean', action='store_true', help='Dọn dẹp các file tạm (runs/detect, img/*)')
    parser.add_argument('--start', '-s', type=int, default=0, help='Frame bắt đầu (mặc định: 0)')
    parser.add_argument('--stop', '-e', type=int, default=0, help='Frame kết thúc (mặc định: 0 = đến cuối video)')
    parser.add_argument('--setup', action='store_true', help='Chỉ setup thư mục và kiểm tra')

    args = parser.parse_args()

    print("🚀 Helmet Violation Detection System")
    print("=" * 50)

    # Setup thư mục
    setup_directories()

    if args.setup:
        print("\n📋 Kiểm tra hệ thống:")
        check_models()
        check_config()
        print("\n✅ Setup hoàn tất!")
        return

    # Clean outputs nếu cần
    if args.clean:
        clean_outputs()
        return

    # Xử lý ảnh nếu có
    if args.image:
        # Kiểm tra trước khi chạy
        if not check_config():
            print("\n❌ Vui lòng hoàn thành setup trước khi chạy")
            print("Chạy lệnh: python main.py --setup")
            return
        process_image(args.image)
        return

    # Kiểm tra xem có video được cung cấp không
    if not args.video:
        print("\n❌ Vui lòng cung cấp đường dẫn ảnh (--image) hoặc video (--video)")
        print("Ví dụ: python main.py --image path/to/image.jpg")
        print("Hoặc: python main.py --video path/to/video.mp4")
        print("Hoặc chạy setup: python main.py --setup")
        return

    # Kiểm tra trước khi chạy
    if not check_models() or not check_config():
        print("\n❌ Vui lòng hoàn thành setup trước khi chạy")
        print("Chạy lệnh: python main.py --setup")
        return

    # Xử lý video
    process_video(args.video, args.start, args.stop)

if __name__ == "__main__":
    main()
