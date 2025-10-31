import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("UAE Traffic Sign Detector")
st.write("Upload a photo of a UAE traffic sign")

model = YOLO("runs/detect/uae_signs/weights/best.pt")

file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])
if file:
    img = Image.open(file)
    results = model(img)
    annotated = results[0].plot()
    st.image(annotated, use_column_width=True)
