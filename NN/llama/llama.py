from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM
import torch
import time

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = NPUModelForCausalLM.from_pretrained(model_id, use_cache=True, dtype=torch.int8).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

query = input("Ask something: ")
prefix = tokenizer(query, return_tensors="pt")["input_ids"]

generation_kwargs = dict(
    input_ids=prefix,
    streamer=streamer,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    max_new_tokens=512,
)

print("Run inference")
start_time = time.time()

output = model.generate(**generation_kwargs)

end_time = time.time()

total_inference_time = end_time - start_time
print(f"\nTotal inference time: {total_inference_time:.4f} seconds")

# If you want to print the generated text (this might already be done by the streamer)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated text:")
print(generated_text)
