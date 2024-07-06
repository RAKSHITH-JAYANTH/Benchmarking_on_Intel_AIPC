import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from custom_nn_npu_class import NPUModelForImageClassification

img = Image.open("dog.jpeg")
model = NPUModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
with torch.no_grad():
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

predicted_label = logits.argmax(-1).item()
model.config.id2label[predicted_label]
print(predicted_label)
