import argparse
import sys
import inspect
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Validate or predict with Ultralytics YOLO using torch.safe_globals to bypass PyTorch 2.6 weights_only loader")
    parser.add_argument("task", choices=["val", "predict"], help="Run validation or prediction")
    parser.add_argument("--model", required=True, help="Path to weights .pt file")
    parser.add_argument("--data", help="Path to data.yaml (required for val)")
    parser.add_argument("--source", help="Image/dir/video path or glob (required for predict)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predict")
    parser.add_argument("--device", default="", help="CUDA device, i.e. 0 or 0,1,2,3 or cpu (default: auto)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"[ERROR] Model not found: {model_path}")

    # Lazy import to avoid hard dependency if user only wants the help text
    try:
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        import ultralytics.nn.modules as u_modules
        import ultralytics.utils as u_utils
        try:
            import ultralytics.utils.loss as u_loss
        except Exception:
            u_loss = None
    except Exception as e:
        sys.exit(f"[ERROR] Failed to import ultralytics. Install with: pip install ultralytics\nDetails: {e}")

    # Map device string
    device = None
    if args.device:
        device = args.device

    try:
        # Base PyTorch layer classes commonly seen
        from torch.nn.modules.container import Sequential, ModuleList
        from torch.nn.modules.activation import SiLU
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.batchnorm import BatchNorm2d
        from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
        from torch.nn.modules.upsampling import Upsample
        from torch.nn.modules.linear import Linear
        from torch.nn.modules.dropout import Dropout
        import torch.nn as nn

        allow_set = {
            DetectionModel,
            Sequential,
            ModuleList,
            SiLU,
            Conv2d,
            BatchNorm2d,
            MaxPool2d,
            AdaptiveAvgPool2d,
            Upsample,
            Linear,
            Dropout,
        }

        # Dynamically collect ALL class types from ultralytics.nn.modules and common submodules
        modules_to_scan = [u_modules, u_utils]
        if u_loss is not None:
            modules_to_scan.append(u_loss)
        # Also scan torch.nn top-level for common loss/activation layers
        modules_to_scan.append(nn)
        for sub in ("conv", "block", "head", "transformer", "attention"):
            try:
                mod = getattr(u_modules, sub)
                modules_to_scan.append(mod)
            except Exception:
                pass

        for mod in modules_to_scan:
            try:
                for name, obj in inspect.getmembers(mod):
                    if inspect.isclass(obj):
                        allow_set.add(obj)
            except Exception:
                continue

        with torch.serialization.safe_globals(list(allow_set)):
            model = YOLO(str(model_path))
    except Exception as e:
        sys.exit(
            "[ERROR] Failed to load model under safe_globals.\n"
            + str(e)
            + "\nKhuyến nghị nhanh: hạ torch về 2.5.x cho môi trường py -3.13 để dùng trực tiếp CLI yolo."
        )

    if args.task == "val":
        if not args.data:
            sys.exit("[ERROR] --data is required for val")
        data_path = Path(args.data)
        if not data_path.exists():
            sys.exit(f"[ERROR] data.yaml not found: {data_path}")
        results = model.val(data=str(data_path), imgsz=args.imgsz, device=device)
        # results is a Metrics object; print concise dict
        try:
            print(results.results_dict)
        except Exception:
            print(results)
    else:  # predict
        if not args.source:
            sys.exit("[ERROR] --source is required for predict")
        source_path = args.source
        out = model.predict(source=source_path, imgsz=args.imgsz, conf=args.conf, device=device)
        # Print where results are saved
        try:
            save_dir = out[0].save_dir if out and hasattr(out[0], 'save_dir') else None
        except Exception:
            save_dir = None
        if save_dir:
            print(f"[OK] Predictions saved to: {save_dir}")
        else:
            print("[OK] Prediction finished.")


if __name__ == "__main__":
    main()
