# Phần 1 thư viện , hàm , class

## 1️⃣ Imports & Libraries

Đây là những thư viện bạn sẽ gặp trong project

```bash
import cv2 # Computer Vision - xử lý ảnh
import numpy as np # Numerical computing - tính toán
import torch # Deep learning framework
from pathlib import Path # Xử lý đường dẫn file
```

## 2️⃣ Variables & Data Types

```bash
# String
name = "helmet"
path = "D:\\images\\photo.jpg"

# Integer & Float
confidence = 0.95
width = 640
height = 480

# List - danh sách
boxes = [100, 50, 200, 150]  # [x1, y1, x2, y2]
classes = ["helmet", "nohelmet", "licenseplate"]

# Dictionary - từ điển
result = {
    "class": "helmet",
    "confidence": 0.95,
    "box": [100, 50, 200, 150]
}

# Tuple - tuple (không thể thay đổi)
bbox = (100, 50, 200, 150)

# NumPy Array - mảng (rất quan trọng!)
image = np.zeros((640, 480, 3))  # Tạo ảnh đen 640x480 RGB
```

## 3️⃣ Functions - Hàm (CẬP NHẤT với ví dụ thực tế)

```bash
# Hàm đơn giản - không có return
def print_detection(class_name, confidence):
    print(f"Detected: {class_name} with confidence {confidence}")

# Gọi hàm
print_detection("helmet", 0.95)
# Output: Detected: helmet with confidence 0.95


# Hàm có return
def calculate_iou(box1, box2):
    """
    Tính toán IoU (Intersection over Union) giữa 2 bounding boxes
    Input: box1 = [x1, y1, x2, y2], box2 = [x1, y1, x2, y2]
    Output: iou = độ trùng lặp (0-1)
    """
    # Tính vùng giao nhau
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Tính vùng của từng box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU = giao / hợp
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

# Gọi hàm
iou = calculate_iou([0, 0, 100, 100], [50, 50, 150, 150])
print(f"IoU: {iou}")
# Output: IoU: 0.14285714...

# Hàm có Default Parameters:
def load_model(model_path, device="cpu"):
    """
    Load YOLO model
    Input:
      - model_path: đường dẫn tới model (str)
      - device: "cpu" hoặc "cuda" (default: "cpu")
    Output: model object
    """
    if device == "cuda":
        model = torch.load(model_path, map_location="cuda:0")
    else:
        model = torch.load(model_path, map_location="cpu")
    return model

# Gọi với default
model = load_model("models/HelmetLP.pt")

# Gọi với GPU
model = load_model("models/HelmetLP.pt", device="cuda")
```

## 4️⃣ Loops & Conditions

```bash
# For loop - lặp qua danh sách
boxes = [[100, 50, 200, 150], [300, 100, 400, 200]]
for box in boxes:
    x1, y1, x2, y2 = box
    print(f"Box: ({x1}, {y1}) to ({x2}, {y2})")

# If-else - điều kiện
confidence = 0.95
if confidence > 0.9:
    print("High confidence detection")
elif confidence > 0.7:
    print("Medium confidence")
else:
    print("Low confidence, skip")

# List comprehension - cách viết ngắn gọn
results = [0.95, 0.87, 0.92, 0.65]
high_conf = [r for r in results if r > 0.9]
print(high_conf)  # [0.95, 0.92]
```

## 5️⃣ Classes - Lớp (OOP)

```bash
# Định nghĩa class Detection
class Detection:
    def __init__(self, class_name, confidence, bbox):
        """
        Constructor - chạy khi tạo object
        Input:
          - class_name: "helmet", "nohelmet", hoặc "licenseplate"
          - confidence: 0-1
          - bbox: [x1, y1, x2, y2]
        """
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox

    def is_violation(self):
        """
        Method - kiểm tra có phải vi phạm không
        Output: True nếu là nohelmet, False nếu là helmet
        """
        return self.class_name == "nohelmet"

    def get_area(self):
        """
        Tính diện tích bounding box
        Output: diện tích (pixels²)
        """
        x1, y1, x2, y2 = self.bbox
        area = (x2 - x1) * (y2 - y1)
        return area

# Tạo object
det = Detection(class_name="helmet", confidence=0.95, bbox=[100, 50, 200, 150])
print(det.is_violation())  # False (đội mũ)
print(det.get_area())      # 5000 pixels²
```

# Phần B: OpenCV - Xử lý Ảnh

## 1️⃣ Đọc & Viết Ảnh

```bash
import cv2 ##Thư viện computer vision 2
import numpy as np

# Đọc ảnh
image = cv2.imread("photo.jpg") #.imread trả về một NumPy array dùng để lưu trữ ảnh
# image là NumPy array có shape (height, width, 3)
# 3 channels = BGR (Blue, Green, Red - không phải RGB!)

# Kiểm tra
print(image.shape)        # (480, 640, 3) - height, width, channels
print(image.dtype)        # uint8 - giá trị pixel từ 0-255

# Viết ảnh
cv2.imwrite("output.jpg", image) #.imwrite lưu NumPy array thành file ảnh

# Lỗi thường gặp
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#.cvtColor chuyển đổi không gian màu
# OpenCV dùng BGR, nên cần convert sang RGB để hiển thị đúng màu
-----------
im = cv2.imread(path)  # Đọc ảnh từ đường dẫn
# im là ảnh gốc (BGR format)
```

## 2️⃣ Xử lý Màu Sắc (Color Spaces)

```bash
image = cv2.imread("photo.jpg")  # BGR format

# Convert sang Grayscale (xám) - 1 channel
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)  # (480, 640) - chỉ 2 dimensions (dimension là height và width)

# Convert sang HSV (Hue, Saturation, Value) - tốt cho detection màu
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Sử dụng: tìm vật thể màu xanh
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
#.inRange tạo mask binary
# mask là ảnh binary: 255 (trắng) là xanh, 0 (đen) là không phải
# Từ _LP_Helmet.py: chuyển sang grayscale để OCR (OCR thường dùng ảnh xám là quá trình tiền xử lý phổ biến trước khi nhận dạng ký tự)
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(plate)
```

## 3️⃣ Thresholding - Ngưỡng

```bash
# Thresholding là kỹ thuật chuyển ảnh xám thành ảnh binary (đen trắng) dựa trên ngưỡng cố định hoặc tự động
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# ret = ngưỡng tìm được (127)
# binary = ảnh binary: 0 hoặc 255
# Pixel >= 127 → 255 (trắng), Pixel < 127 → 0 (đen)

# OTSU thresholding - tự động tìm ngưỡng tốt nhất
ret, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive thresholding - từng vùng ảnh khác nhau
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# Tốt hơn khi lighting không đồng nhất

#Trong project (_LP_Helmet.py, line 110-120):
_, imgGrayscaleplate = cv2.threshold(imgGrayscaleplate, 200, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Chuyển ảnh biển số thành binary để tìm contour
```

## 4️⃣ Contour Detection - Tìm Đường Viền

```bash
# Tìm contour (contours là đường viền của các vật thể trong ảnh binary)
# bằng cách sử dụng cv2.findContours đầu tiên chuyển ảnh sang binary bằng threshold
image = cv2.imread("photo.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = danh sách các đường viền tìm được
# hierarchy = mối quan hệ cha-con của contours

# Lặp qua contours
for cntr in contours:
    # Tính bounding box  từ contour
    x, y, w, h = cv2.boundingRect(cntr)
    # x, y = góc trái trên, w = rộng, h = cao

    # Tính diện tích
    area = cv2.contourArea(cntr)
    if area > 100:  # Chỉ quan tâm contour lớn
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite("output.jpg", image)
--------------------
# thực tế Trong project (_LP_Helmet.py, line 20-40):
cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Tìm contour của các ký tự biển số
for cntr in cntrs:
    intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
    if intWidth > lower_width and intWidth < upper_width:
        # Lọc contour có kích thước phù hợp
```

## 5️⃣ Morphological Operations - Xử lý Hình Dạng

```bash
# Tạo kernel (kernel - nhân) để thao tác hình dạng
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#.getStructuringElement tạo kernel hình chữ nhật 5x5

# Dilation - mở rộng
dilated = cv2.dilate(binary, kernel, iterations=1)
#.dilate làm các vùng trắng (255) to hơn , iterations = số lần lặp

# Erosion - co nhỏ
eroded = cv2.erode(binary, kernel, iterations=1)
#.erode làm các vùng trắng (255) nhỏ hơn

# Opening (erode + dilate) - xóa nhiễu nhỏ
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing (dilate + erode) - lấp khoảng trống
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

--------------------
# thực tế Trong project (_LP_Helmet.py, line 95-105):
kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
thre_mor = cv2.morphologyEx(img, cv2.MORPH_DILATE, kerel3)
# Dilation để nối các pixel ký tự lại với nhau
```

## 6️⃣ Drawing - Vẽ Hình

```bash
image = cv2.imread("photo.jpg")

# Vẽ hình chữ nhật
cv2.rectangle(image, (100, 50), (200, 150), (0, 255, 0), 2)
# (100, 50) = điểm trái trên, (200, 150) = điểm phải dưới
# (0, 255, 0) = màu xanh BGR, 2 = độ dày đường

# Vẽ hình tròn
cv2.circle(image, (150, 100), 50, (0, 0, 255), 2)
# (150, 100) = tâm, 50 = bán kính, (0, 0, 255) = màu đỏ

# Vẽ text
cv2.putText(image, "Helmet", (100, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2)
# (100, 30) = vị trí text, (255, 0, 0) = màu xanh dương, 1 = font size

cv2.imwrite("output.jpg", image)

--------------------
# thực tế Trong project (ui_app.py):
# Vẽ bounding boxes và labels
for detection in detections:
    if detection['violation']:
        color = (0, 0, 255)  # Đỏ - vi phạm
    else:
        color = (0, 255, 0)  # Xanh - OK
    cv2.rectangle(image, box, color, 2)
```

## 7️⃣ ROI (Region of Interest) - Cắt Vùng Ảnh

```bash
image = cv2.imread("photo.jpg")  # 640x480

# Cắt vùng từ (x1, y1) đến (x2, y2)
x1, y1, x2, y2 = 100, 50, 200, 150
roi = image[y1:y2, x1:x2]  # QUAN TRỌNG: [y, x] không phải [x, y]!
# roi là ảnh con (100x100)

print(roi.shape)  # (100, 100, 3)

# Lưu ROI
cv2.imwrite("roi.jpg", roi)

--------------------
# thực tế Trong project (_Motobike.py):
# Từ bounding box của xe máy, cắt ROI để detect mũ/biển số
x1, y1, x2, y2 = int(box.xyxy[0])
crop = image[y1:y2, x1:x2]  # Cắt ROI
# Sau đó dùng crop này để chạy Model 2
```
