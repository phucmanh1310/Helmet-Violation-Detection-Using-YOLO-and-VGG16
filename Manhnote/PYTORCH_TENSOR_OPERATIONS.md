# Phần A: PyTorch Basics & Tensors

## 1️⃣ PyTorch là gì?

PyTorch = Framework deep learning của Facebook/Meta

- Giống NumPy nhưng có GPU support
- Tự động tính gradient (đạo hàm) → training neural networks
- Dùng cho YOLOv8 trong project này

## 2️⃣ Tensor là gì?

Tensor thực chất là một mảng chứa các con số đa chiều (multidimensional array) nó hơn numpy array ở chỗ nó có thể được sử dụng trên GPU để tăng tốc tính toán. Và tự động tính gradient cho các phép toán.

#### Các loại Tensor phổ biến:

- Scalar (0D) - 1 số
  scalar = torch.tensor(42)
  print(scalar.shape) # torch.Size([])

- Vector (1D) - mảng 1 chiều

vector = torch.tensor([1, 2, 3])
print(vector.shape) # torch.Size([3])

- Matrix (2D) - ma trận

matrix = torch.tensor([[1, 2], [3, 4]])
print(matrix.shape) # torch.Size([2, 2])

- 3D Tensor - ảnh RGB

image = torch.randn(3, 224, 224) # (channels, height, width)
print(image.shape) # torch.Size([3, 224, 224])

- 4D Tensor - batch of images

batch = torch.randn(16, 3, 224, 224) # (batch_size, channels, height, width)
print(batch.shape) # torch.Size([16, 3, 224, 224])

## 3️⃣ Tạo Tensor trong PyTorch

```bash
# Từ list
tensor = torch.tensor([1, 2, 3, 4])
print(tensor)  # tensor([1, 2, 3, 4])

# Từ nested list (2D)
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(matrix.shape)  # torch.Size([3, 2])

# từ NumPy array
# NumPy → Tensor
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)
print(tensor)  # tensor([1, 2, 3])

# Tensor → NumPy
tensor = torch.tensor([4, 5, 6])
arr = tensor.numpy()
print(arr)  # [4 5 6]

#Tensor đặc biệt
# Tensor toàn 0
zeros = torch.zeros(3, 4)  # 3 hàng, 4 cột
print(zeros.shape)  # torch.Size([3, 4])

# Tensor toàn 1
ones = torch.ones(2, 3)

# Tensor random (uniform [0, 1))
rand = torch.rand(2, 3)
print(rand)
# tensor([[0.5414, 0.4792, 0.6927],
#         [0.2484, 0.7682, 0.0885]])

# Tensor random (normal distribution, mean=0, std=1)
randn = torch.randn(2, 3)

# Tensor với giá trị cụ thể
full = torch.full((2, 3), 7.0)  # Fill với 7.0
print(full)
# tensor([[7., 7., 7.],
#         [7., 7., 7.]])

# Tensor identity (ma trận đơn vị)
eye = torch.eye(3)
print(eye)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

#Tạo tensor giống shape khác:
x = torch.tensor([[1, 2], [3, 4]])

# Tạo tensor 0 cùng shape
zeros_like = torch.zeros_like(x)
print(zeros_like.shape)  # torch.Size([2, 2])

# Tạo tensor 1 cùng shape
ones_like = torch.ones_like(x)

```

## 4️⃣ Thuộc tính của Tensor

```bash
tensor = torch.randn(2, 3, 4)

# Shape là kích thước mỗi chiều
print(tensor.shape)  # torch.Size([2, 3, 4])
print(tensor.size())  # torch.Size([2, 3, 4]) - giống .shape

# Data type (các kiểu số)
print(tensor.dtype)  # torch.float32 (default)

# Device (CPU hay GPU)
print(tensor.device)  # cpu

# Số chiều (dimensions)
print(tensor.ndim)  # 3

# Tổng số phần tử
print(tensor.numel())  # 2 * 3 * 4 = 24

```

## 5️⃣ Kiểu dữ liệu (Data Types) trong PyTorch

```bash
# Float (default, dùng cho neural networks)
float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)

# Integer
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

# Boolean
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)

# Convert dtype
x = torch.tensor([1, 2, 3])  # int64
x_float = x.float()  # → float32
x_double = x.double()  # → float64
```

## 6️⃣ Các phép toán cơ bản với Tensor

- Đối với vector:

```bash
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Cộng
add = x + y  # tensor([5, 7, 9])
add = torch.add(x, y)  # giống nhau

# Trừ
sub = x - y  # tensor([-3, -3, -3])

# Nhân (element-wise)
mul = x * y  # tensor([4, 10, 18])

# Chia
div = x / y  # tensor([0.2500, 0.4000, 0.5000])

# Lũy thừa
pow = x ** 2  # tensor([1, 4, 9])

# Sqrt
sqrt = torch.sqrt(x.float())  # tensor([1.0000, 1.4142, 1.7321])
```

- Đối với ma trận:

```bash

#Matrix multiplication (ma trận nhân)
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Phép nhân ma trận (matrix multiplication)
C = torch.matmul(A, B)
# hoặc
C = A @ B
print(C)
# tensor([[19, 22],
#         [43, 50]])

# Transpose (chuyển vị)
A_T = A.T
# hoặc
A_T = A.transpose(0, 1)
print(A_T)
# tensor([[1, 3],
#         [2, 4]])
```

- Đối với Reduction operations (tổng, trung bình, max, min):

```bash
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Sum (tổng)
total = x.sum()  # tensor(21)
sum_dim0 = x.sum(dim=0)  # tensor([5, 7, 9]) - sum theo cột
sum_dim1 = x.sum(dim=1)  # tensor([6, 15]) - sum theo hàng

# Mean (trung bình)
mean = x.float().mean()  # tensor(3.5)

# Max, Min
max_val = x.max()  # tensor(6)
min_val = x.min()  # tensor(1)

# Argmax (index của giá trị max)
argmax = x.argmax()  # tensor(5) - vị trí của 6 (flatten)
```

- Đối với Reshaping tensors (ví dụ chuyển đổi shape):

```bash
x = torch.randn(2, 3, 4)  # shape: [2, 3, 4]

# View (reshape, chia sẻ memory)
y = x.view(6, 4)  # [6, 4]
y = x.view(-1, 4)  # [-1] = auto calculate (6 trong trường hợp này)

# Reshape (giống view nhưng có thể copy nếu cần)
z = x.reshape(2, 12)  # [2, 12]

# Flatten (chuyển thành 1D)
flat = x.flatten()  # [24]

# Squeeze (xóa dimensions = 1)
a = torch.randn(1, 3, 1, 4)
b = a.squeeze()  # [3, 4]

# Unsqueeze (thêm dimension = 1)
c = torch.randn(3, 4)
d = c.unsqueeze(0)  # [1, 3, 4]
e = c.unsqueeze(1)  # [3, 1, 4]
```

## 7️⃣ Device Management (CPU vs GPU)

## 8️⃣ Indexing & Slicing (Chỉ mục & Cắt Tensor)

```bash
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

# Lấy 1 phần tử
print(x[0, 1])  # tensor(2)

# Lấy 1 hàng
print(x[1])  # tensor([5, 6, 7, 8])

# Lấy 1 cột
print(x[:, 2])  # tensor([3, 7, 11])

# Slicing
print(x[0:2, 1:3])
# tensor([[2, 3],
#         [6, 7]])

# Advanced indexing
indices = torch.tensor([0, 2])
print(x[indices])  # Lấy hàng 0 và 2
# tensor([[1, 2, 3, 4],
#         [9, 10, 11, 12]])

# Boolean indexing
mask = x > 5
print(x[mask])  # tensor([6, 7, 8, 9, 10, 11, 12])
```

# Phần B : Autograd - Tự động Tính Gradient

## 1️⃣ Gradient là gì?

Gradient là đạo hàm của hàm số, biểu thị tốc độ thay đổi của hàm số theo biến số. Trong học sâu (deep learning), gradient được sử dụng để cập nhật trọng số của mô hình thông qua quá trình tối ưu hóa (optimization) nhằm giảm thiểu hàm mất mát (loss function).
Ví dụ : Hàm số y = x^2 có đạo hàm (gradient) là dy/dx = 2x. Tại điểm x = 3, gradient là 6, nghĩa là nếu ta tăng x lên một đơn vị nhỏ, y sẽ tăng khoảng 6 đơn vị.

## 2️⃣ Cách tự động tính gradient với PyTorch

Trong PyTorch, khi bạn tạo một tensor với thuộc tính `requires_grad=True`, PyTorch sẽ theo dõi tất cả các phép toán được thực hiện trên tensor đó để có thể tính toán gradient tự động khi cần thiết.

```bash
# Tensor thường (không track gradient)
x = torch.tensor([1.0, 2.0, 3.0])
print(x.requires_grad)  # False

# Tensor track gradient
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.requires_grad)  # True

# Hoặc
x = torch.tensor([1.0, 2.0, 3.0])
x.requires_grad_(True)  # In-place operation
```

## 3️⃣ Backward Pass - Tính Gradient của hàm mất mát

Khi bạn có một hàm số (thường là hàm mất mát trong học sâu), bạn có thể gọi phương thức `.backward()` trên tensor kết quả để tính toán gradient đối với tất cả các tensor có `requires_grad=True` tham gia vào phép toán.

```bash
# Define tensor với gradient tracking
x = torch.tensor(2.0, requires_grad=True)

# Forward pass: y = x²
y = x ** 2

# Backward pass: tính gradient dy/dx = 2x
y.backward()

# Xem gradient
print(x.grad)  # tensor(4.0) = 2 * 2
```

## 4️⃣ Gradient Accumulation (Tích lũy Gradient)

Mặc định, mỗi lần bạn gọi `.backward()`, gradient sẽ được cộng dồn vào thuộc tính `.grad` của tensor. Nếu bạn muốn reset gradient về 0 trước khi tính toán mới, bạn cần gọi `.zero_()`.

```bash
x = torch.tensor(2.0, requires_grad=True)

# First computation
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor(4.0)

# Second computation (gradient accumulates!)
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor(16.0) = 4.0 + 12.0

# Reset gradient
x.grad.zero_()
print(x.grad)  # tensor(0.)
```

Cụ thể trong Training loop của neural network, bạn thường sẽ gọi `optimizer.zero_grad()` trước mỗi lần tính toán backward để tránh việc gradient bị cộng dồn không mong muốn.

```bash
for epoch in range(100):
    optimizer.zero_grad()  # Reset gradient mỗi epoch

    output = model(input)
    loss = criterion(output, target)

    loss.backward()  # Tính gradient
    optimizer.step()  # Update weights
```

## 5️⃣ No Grad Context (Inference) - Vô hiệu hóa tính toán gradient

Khi predicting hoặc đánh giá mô hình (inference) sẽ kế thừa lại kết quả cũ nên bạn không cần tính toán gradient. Để tiết kiệm bộ nhớ và tăng tốc độ, bạn có thể sử dụng ngữ cảnh `torch.no_grad()` để vô hiệu hóa việc theo dõi gradient.

```bash
x = torch.randn(3, 4, requires_grad=True)

# Với gradient (training)
y = x ** 2
print(y.requires_grad)  # True

# Không gradient (inference)
with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)  # False

# Hoặc dùng decorator
@torch.no_grad()
def predict(model, input):
    return model(input)
```

#
