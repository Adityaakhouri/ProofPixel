import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Load model
@st.cache_resource
def load_model():
    model = torch.load("model.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

# Preprocessing + Prediction
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Change to your model's input
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        return "AI-Generated" if pred.item() == 1 else "Real"

# Streamlit UI
st.title("AI vs Real Image Classifier")

uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    st.write("Analyzing...")
    model = load_model()
    result = predict(img, model)
    st.success(f"Prediction: {result}")
