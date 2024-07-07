import intel_npu_acceleration_library
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(42)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)

input_size = 8
hidden_size = 4
model = SimpleLSTM(input_size, hidden_size)

input_data = torch.randn(1, 10, input_size)

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

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("loaded_model_inference"):
        test_input = torch.randn(1, 10, input_size)
        with torch.no_grad():
            output = loaded_model(test_input)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("\nLoaded model output shape:", output.shape)
print("Loaded model output:", output)
