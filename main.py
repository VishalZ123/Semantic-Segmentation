import torch
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
from models import Net
import matplotlib.pyplot as plt

# original model
model = Net(12)
model.load_state_dict(torch.load('./model.pt', map_location = 'cpu'))

# improved model
model_improved = Net(12)
model_improved.load_state_dict(torch.load('./improved_model.pt', map_location='cpu'))


h, w = 128, 128
transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
])

def predict(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        prediction = model(image).squeeze()
        # select maximum 
        prediction = torch.argmax(prediction, dim=0)

        # scale between 0 and 1
        prediction = (prediction-prediction.min()) / (prediction.max()-prediction.min())
        # convert to numpy
        prediction = prediction.numpy()
    return prediction

# UI
st.title("Semantic Segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Prediction using Original Model")
    if st.button("Predict", key=0):
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        with st.spinner("Predicting..."):
            predictions = predict(image, model)
            st.image(predictions, width=300, clamp=True)
            # apply cmap to predictions to convert to RGB
            predictions = plt.cm.hsv(predictions)
            st.image(predictions, width=300, clamp=True)
        
    st.subheader("Prediction using Improved Model")
    if st.button("Predict", key=1):
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        with st.spinner("Predicting..."):
            predictions = predict(image, model_improved)
            st.image(predictions, width=300, clamp=True)
            # apply cmap to predictions to convert to RGB
            predictions = plt.cm.hsv(predictions)
            st.image(predictions, width=300, clamp=True)
