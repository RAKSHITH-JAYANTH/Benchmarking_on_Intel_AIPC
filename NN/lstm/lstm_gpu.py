import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
import time

torch.manual_seed(42)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)

# Initialize model and data on CPU
input_size = 256
hidden_size = 512
model = SimpleLSTM(input_size, hidden_size)
input_data = torch.randn(1, 100, input_size)

# Perform initial inference on CPU
with torch.no_grad():
    predictions = model(input_data)

print("CPU Predictions shape:", predictions.shape)
print("CPU Predictions:", predictions)

# Save the model
torch.save(model.state_dict(), 'simple_lstm_model.pth')
print("Model saved as simple_lstm_model.pth")

# Load the model for XPU inference
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

loaded_model = SimpleLSTM(input_size, hidden_size)
loaded_model.load_state_dict(torch.load('simple_lstm_model.pth'))
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Optimize the model for XPU
loaded_model = ipex.optimize(loaded_model)

# Prepare input for XPU
test_input = torch.randn(1, 10, input_size, device=device)

# Warmup run
with torch.no_grad():
    _ = loaded_model(test_input)

# Perform inference on XPU with timing
num_iterations = 100
torch.xpu.synchronize()
start_time = time.time()

for _ in range(num_iterations):
    with torch.no_grad():
        output = loaded_model(test_input)

torch.xpu.synchronize()
end_time = time.time()

# Calculate average time
avg_time = (end_time - start_time) / num_iterations
print(f"\nAverage inference time over {num_iterations} iterations: {avg_time:.6f} seconds")

# Move output to CPU for printing
output_cpu = output.to("cpu")
print("\nLoaded model output shape:", output_cpu.shape)
print("Loaded model output:", output_cpu)
