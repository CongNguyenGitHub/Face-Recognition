from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

import streamlit as st
import torch.nn as nn
import torch
import os
import numpy as np
import pickle
import pathlib
from pathlib import Path

pathlib.PosixPath = pathlib.WindowsPath
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def face_detection(image, model):
    results = model(image)
    predictions = results.pandas().xyxy[0]
    if predictions.empty:
        return None
    top_prediction = predictions.loc[predictions['confidence'].idxmax()]
    x_min, y_min, x_max, y_max = map(int, [top_prediction['xmin'], top_prediction['ymin'], top_prediction['xmax'], top_prediction['ymax']])
    return image.crop((x_min, y_min, x_max, y_max))

class FaceDetectionTransform:
    def __init__(self, model):
        self.model = model

    def __call__(self, image):
        cropped_image = face_detection(image, self.model)
        return cropped_image if cropped_image is not None else image

@st.cache_resource
def load_detector(path="../models/yolo_model/weights/best.pt"):
    detector = torch.hub.load("ultralytics/yolov5", "custom", path=path)
    return detector

def transform_image(detector):
    return transforms.Compose([
        FaceDetectionTransform(detector),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseNetwork(nn.Module):
    def __init__(self, dropout_rate=0.75):
        super(SiameseNetwork, self).__init__()
        self.base_model = InceptionResnetV1(pretrained="vggface2")
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, 1)

    def forward(self, input1, input2):
        embedding1 = self.base_model(input1)
        embedding2 = self.base_model(input2)
        distance = torch.abs(embedding1 - embedding2)
        distance = self.dropout(distance)
        return self.fc(distance)

@st.cache_resource
def load_model(path_to_checkpoint, device, dropout_rate=0.75):
    model = SiameseNetwork(dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(path_to_checkpoint, map_location=device))
    return model

def extract_feature(model, path_to_img, transform, device):
    model.eval()
    model.to(device)
    img = Image.open(path_to_img).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return model.base_model(img_tensor)

def load_databases():
    feature_databases = np.load("./feature_databases.npy")
    with open("./name_databases.pkl", "rb") as file:
        name_databases = pickle.load(file)
    return torch.from_numpy(feature_databases), name_databases

def save_database(feature_databases, name_databases):
    np.save("./feature_databases.npy", feature_databases.numpy())
    with open("./name_databases.pkl", "wb") as file:
        pickle.dump(name_databases, file)

folder_images = "./images/"
os.makedirs(folder_images, exist_ok=True)

detector = load_detector()
transform_YOLO = transform_image(detector)
model = load_model("../models/siamese_model/weights/best_model.pt", device, dropout_rate=0.5)
features_databases, name_databases = load_databases()

st.title("Face Recognition & Database Management")

option = st.radio("Choose an option", ["Add to Database", "Recognize Face"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = os.path.join(folder_images, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    new_feature = extract_feature(model, file_path, transform_YOLO, device)
    
    if option == "Add to Database":
        label_input = st.text_input("Enter label:")
        if label_input:
            features_databases = torch.cat((features_databases, new_feature), dim=0)
            name_databases.append(label_input)
            save_database(features_databases, name_databases)
            st.success(f"Added {label_input} to database!")
    
    elif option == "Recognize Face":
        distance = torch.abs(new_feature - features_databases)
        output = model.fc(distance)
        results = torch.sigmoid(output)
        index_label = torch.argmax(results).item()
        pred_label = name_databases[index_label] if results[index_label] >= 0.62 else "Unknown"
        st.image(Image.open(uploaded_file))
        st.markdown(f"### Recognized: {pred_label}")
