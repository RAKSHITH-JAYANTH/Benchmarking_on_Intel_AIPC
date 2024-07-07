import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from custom_nn_npu_class import NPUModelForImageClassification
from torch.profiler import profile, record_function, ProfilerActivity

img = Image.open("dog.jpeg")
model = NPUModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("loaded_model_inference"):
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
predicted_label = logits.argmax(-1).item()
model.config.id2label[predicted_label]
print(predicted_label)
