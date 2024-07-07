import torch
import intel_extension_for_pytorch as ipex
import time

# Check if XPU is available
if not torch.xpu.is_available():
    print("Intel XPU is not available. Please check your installation.")
    exit()

# Set the device to XPU
device = torch.device("xpu")

# Define matrix and vector dimensions
m, n = 256,256

# Create a random 512x512 matrix and a 512x1 vector on XPU
A = torch.randn(m, n, device=device, dtype=torch.bfloat16)
x = torch.randn(n, 1, device=device, dtype=torch.bfloat16)

# Perform matrix-vector multiplication
b = torch.matmul(A, x)

# Print shapes
print("Shape of A (matrix):", A.shape)
print("Shape of x (vector):", x.shape)
print("Shape of b (result):", b.shape)

# Print a small portion of the result
print("\nFirst 5 elements of the result vector:")
print(b[:5].cpu().float())  # Convert back to CPU and float for printing

# Measure execution time
num_iterations = 100
start_time = time.time()
for _ in range(num_iterations):
    b = torch.matmul(A, x)
torch.xpu.synchronize()  # Ensure all XPU computations are completed
end_time = time.time()

average_time = (end_time - start_time) / num_iterations
print(f"\nAverage execution time over {num_iterations} iterations: {average_time:.6f} seconds")
