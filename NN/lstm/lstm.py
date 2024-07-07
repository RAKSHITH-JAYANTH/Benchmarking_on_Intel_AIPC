import intel_npu_acceleration_library
import torch
import torch.nn as nn
import time

torch.manual_seed(42)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)

input_size = 256
hidden_size = 512
model = SimpleLSTM(input_size, hidden_size)
input_data = torch.randn(1, 100, input_size)

with torch.no_grad():
    predictions = model(input_data)

print("Predictions shape:", predictions.shape)
print("Predictions:", predictions)

torch.save(model.state_dict(), 'simple_lstm_model.pth')
print("Model saved as simple_lstm_model.pth")

loaded_model = SimpleLSTM(input_size, hidden_size)
loaded_model.load_state_dict(torch.load('simple_lstm_model.pth'))
loaded_model.eval()
loaded_model = torch.compile(loaded_model, backend="npu")

# Warmup run
test_input = torch.randn(1, 100, input_size)
with torch.no_grad():
    _ = loaded_model(test_input)

# Perform inference with timing
num_iterations = 100
total_time = 0

for _ in range(num_iterations):
    test_input = torch.randn(1, 100, input_size)
    
    start_time = time.time()
    with torch.no_grad():
        output = loaded_model(test_input)
    end_time = time.time()
    
    total_time += (end_time - start_time)

average_time = total_time / num_iterations
print(f"\nAverage inference time over {num_iterations} iterations: {average_time:.6f} seconds")

print("\nLoaded model output shape:", output.shape)
print("Loaded model output:", output)
