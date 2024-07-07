import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = model.to(device)
model = ipex.optimize(model)
model.eval()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

query = "Tell me a short joke."
print(f"Query: {query}")

inputs = tokenizer(query, return_tensors="pt").to(device)

print("Running inference...")
start_time = time.time()

with torch.no_grad(), torch.xpu.amp.autocast():
    output = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=50, top_p=0.9)

end_time = time.time()

print(f"Inference time: {end_time - start_time:.2f} seconds")
print("Generated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
