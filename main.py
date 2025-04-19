import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import json
from transformers import BlipProcessor, BlipForConditionalGeneration

resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with open("imagenet-simple-labels.json", "r") as f:
    imagenet_labels = json.load(f)

def detect_main_subject(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = resnet_model(image_tensor)
    
    _, predicted_class = outputs.max(1)
    class_name = imagenet_labels[predicted_class.item()]
    
    return class_name

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_scene(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    out = caption_model.generate(**inputs)
    scene_description = processor.decode(out[0], skip_special_tokens=True)
    return scene_description

def analyze_image(image_path):
    main_subject = detect_main_subject(image_path)
    subject_name = main_subject if main_subject is not None else "unknown subject"
    
    scene_description = describe_scene(image_path)

    output = f"The main subject is a {subject_name}. Background description: {scene_description}."
    return output

image_path = "uploads/pexels-photo-170811.jpeg"
result = analyze_image(image_path)
print(result)
