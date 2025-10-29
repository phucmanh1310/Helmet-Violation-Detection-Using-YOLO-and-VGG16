# Script khởi động nhanh UI

"""
Script này giúp khởi động giao diện web một cách nhanh chóng
Chạy: py -3.13 quick_start_ui.py
hoặc: python quick_start_ui.py
"""

import os
import sys
from pathlib import Path

# Đảm bảo đang ở thư mục gốc của project
project_root = Path(__file__).parent
os.chdir(project_root)

# Thêm thư mục Source vào path
sys.path.insert(0, str(project_root / 'Source'))

print("="*60)
print("🚀 KHỞI ĐỘNG HỆ THỐNG PHÁT HIỆN VI PHẠM MŨ BẢO HIỂM")
print("="*60)
print()
print("📋 Kiểm tra models...")

# Kiểm tra models
models_required = [
    'models/Motov10l.pt',
    'models/HelmetLP.pt'
]

all_models_exist = True
for model_path in models_required:
    if Path(model_path).exists():
        print(f"✅ {model_path}")
    else:
        print(f"❌ {model_path} - KHÔNG TÌM THẤY!")
        all_models_exist = False

if not all_models_exist:
    print()
    print("⚠️ Cảnh báo: Một số models không tồn tại.")
    print("   Vui lòng đảm bảo các file model đã được download.")
    print()
    response = input("Bạn có muốn tiếp tục? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

print()
print("="*60)
print("🌐 Đang khởi động giao diện web...")
print("="*60)
print()
print("📍 Sau khi khởi động, truy cập:")
print("   - Local: http://127.0.0.1:7860")
print("   - Network: http://0.0.0.0:7860")
print()
print("⚠️ Nhấn Ctrl+C để dừng server")
print("="*60)
print()

# Import và chạy UI
from Source.ui_app import demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True  # Tự động mở browser
    )
