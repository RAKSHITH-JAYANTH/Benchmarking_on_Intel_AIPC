import torch
import intel_extension_for_pytorch as ipex

# Check if XPU is available
if not torch.xpu.is_available():
    print("Intel XPU is not available. Please check your installation.")
    exit()

# Set the device to XPU
device = torch.device("xpu")

# Define matrix dimensions for 512x512 matrices
m, n, k = 256, 256, 256

# Create random 512x512 matrices on XPU
A = torch.randn(m, k, device=device, dtype=torch.bfloat16)
B = torch.randn(k, n, device=device, dtype=torch.bfloat16)

# Perform matrix multiplication
C = torch.matmul(A, B)

# Print shapes
print("Shape of A:", A.shape)
print("Shape of B:", B.shape)
print("Shape of C (result):", C.shape)

# Print a small portion of the result
print("\nFirst 4x4 sub-matrix of the result:")
print(C[:4, :4].cpu().float())  # Convert back to CPU and float for printing

# Optional: Measure execution time
import time

start_time = time.time()
for _ in range(10):  # Run 10 times to get an average
    C = torch.matmul(A, B)
torch.xpu.synchronize()  # Ensure all XPU computations are completed
end_time = time.time()

print(f"\nAverage execution time: {(end_time - start_time) / 10:.6f} seconds")
