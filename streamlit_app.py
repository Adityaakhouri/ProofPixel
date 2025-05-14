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
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        return "AI-Generated" if pred.item() == 1 else "Real"

# ------------ Page Setup ------------
st.set_page_config(page_title="AI vs Real Classifier", layout="centered")

# ------------ Session Flags for Simulated Dialogs ------------
if "show_home_box" not in st.session_state:
    st.session_state.show_home_box = False
if "show_about_box" not in st.session_state:
    st.session_state.show_about_box = False

# ------------ Top Bar with Title + Buttons ------------
top_bar = st.columns([6, 1, 1])
with top_bar[0]:
    st.markdown(
        "<h1 style='text-align: left; color: #00BFFF; font-size: 48px;'>ProofPixel</h1>",
        unsafe_allow_html=True
    )
with top_bar[1]:
    if st.button("Home"):
        st.session_state.show_home_box = True
        st.session_state.show_about_box = False
with top_bar[2]:
    if st.button("About Us"):
        st.session_state.show_about_box = True
        st.session_state.show_home_box = False

st.markdown("---")

# ------------ Fake Dialog Box Area (Same size as image area) ------------
if st.session_state.show_home_box or st.session_state.show_about_box:
    with st.container():
        
        if st.session_state.show_home_box:
            st.markdown("### 🏠 You are at Home")
            st.markdown("This is the home section of ProofPixel.")
        if st.session_state.show_about_box:
            st.markdown("### 👥 About Us")
            st.markdown("This AI model classifies whether an image is AI-generated or real using a CNN-based model. Built by **Aditya Akhouri** and **Tanusree Saha**.")
        st.markdown("</div>", unsafe_allow_html=True)

    # If user clicks anywhere else (interacts with any other element), hide the box
    st.write("")  # Spacer to allow next interaction
    if st.button("Click here to dismiss", key="dismiss"):
        st.session_state.show_home_box = False
        st.session_state.show_about_box = False

# ------------ Main Classifier Section ------------
st.title("AI vs Real Image Classifier")
st.write("Upload an image to detect whether it is AI-generated or Real.")

layout = st.columns(1)

with layout[0]:
    uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state['uploaded_image'] = image

        model = load_model()
        st.write("Analyzing image...")
        result = predict(image, model)
        st.success(f"Prediction: **{result}**")
