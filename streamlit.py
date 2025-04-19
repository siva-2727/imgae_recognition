import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import json
from transformers import BlipProcessor, BlipForConditionalGeneration

# Title
st.title("Image Analyzer - Main Subject & Scene Description")

# Load ResNet50 model and labels
@st.cache_resource
def load_resnet_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

@st.cache_data
def load_labels():
    with open("imagenet-simple-labels.json", "r") as f:
        return json.load(f)

resnet_model = load_resnet_model()
imagenet_labels = load_labels()

# Load BLIP model
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, caption_model = load_blip()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_main_subject(image):
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = resnet_model(image_tensor)
    _, predicted_class = outputs.max(1)
    class_name = imagenet_labels[predicted_class.item()]
    return class_name

def describe_scene(image):
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    scene_description = processor.decode(out[0], skip_special_tokens=True)
    return scene_description

def analyze_image(image):
    subject = detect_main_subject(image)
    scene = describe_scene(image)
    return subject, scene

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        subject, scene = analyze_image(image)

    st.markdown("### Analysis Result")
    st.write(f"**Main Subject:** {subject}")
    st.write(f"**Scene Description:** {scene}")
