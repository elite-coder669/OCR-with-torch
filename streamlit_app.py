import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torchvision.transforms as transforms
from ocr import OCR

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      ("cuda" if torch.cuda.is_available() else "cpu"))

# Replace with your exact 62 class labels in order
classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# Preprocessing transform (same as test transforms)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@st.cache_resource
def load_model():
    model = OCR(input_channels=1, num_classes=62, p=0.25)
    model.load_state_dict(torch.load("./weights/bestmodel.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("OCR: Draw a Handwritten Character")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Black background so white drawing is obvious
    stroke_width=20,
    stroke_color="#FFFFFF",  # Drawing color white
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    import numpy as np

    # Convert the canvas image data (RGBA) to a grayscale PIL image of size 64x64
    img_gray = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L').resize((64, 64))

    # Invert colors if your training data was white background and black characters
    img_inverted = Image.eval(img_gray, lambda x: 255 - x)

    st.image(img_inverted, caption="Processed Drawing for Model", width=140)

    # Apply the normalization used in training (mean=0.5, std=0.5)
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform_to_tensor(img_inverted).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_class = classes[pred_idx]

    st.markdown(f"### Predicted Character: `{pred_class}`")
