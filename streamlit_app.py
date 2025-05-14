import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# ------------ Load model ------------
@st.cache_resource
def load_model():
    model = torch.load("model.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

# ------------ Predict Function ------------
def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Adjust based on your model input
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        return "AI-Generated" if pred.item() == 1 else "Real"

# ------------ Page Setup ------------
st.set_page_config(page_title="AI vs Real Classifier", layout="centered")

# ------------ Navigation Menu ------------
menu = st.columns([8, 1, 1])
with menu[1]:
    if st.button("Home"):
        st.experimental_rerun()
with menu[2]:
    if st.button("About Us"):
        st.info("This AI model classifies whether an image is AI-generated or real using a CNN-based model. Built by Aditya Akhouri.")

st.markdown("---")

# ------------ Image Display + Upload UI ------------
st.title("AI vs Real Image Classifier")
st.write("Upload an image to detect whether it is AI-generated or Real.")

layout = st.columns(2)

# Left Column: Placeholder or Uploaded Image
with layout[0]:
    st.subheader("Image Display")
    if 'uploaded_image' not in st.session_state:
        # Display placeholder box
        placeholder = Image.new("RGB", (256, 256), color=(230, 230, 230))
        st.image(placeholder, caption="No Image Uploaded", use_column_width=True)
    else:
        st.image(st.session_state['uploaded_image'], caption="Uploaded Image", use_column_width=True)

# Right Column: Upload & Prediction
with layout[1]:
    uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state['uploaded_image'] = image

        model = load_model()
        st.write("Analyzing image...")
        result = predict(image, model)
        st.success(f"Prediction: **{result}**")
