from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

import streamlit as st
import torch.nn as nn
import torch
import os
import numpy as np
import pickle
import time

import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def face_detection(image,model):
    results = model(image)

    predictions = results.pandas().xyxy[0]
    if predictions.empty:
        return None

    top_prediction = predictions.loc[predictions['confidence'].idxmax()]
    x_min, y_min, x_max, y_max = int(top_prediction['xmin']), int(top_prediction['ymin']), int(top_prediction['xmax']), int(top_prediction['ymax'])
    return image.crop((x_min,y_min,x_max,y_max))

class FaceDetectionTransform:
    def __init__(self, model):
        self.model = model

    def __call__(self, image):
        cropped_image = face_detection(image, self.model)
        return cropped_image if cropped_image is not None else image

@st.cache_resource
def load_detector(path="../yolov5n/kaggle/working/model/weights/best.pt"):
    detector = torch.hub.load("ultralytics/yolov5", "custom", path=path)
    return detector

def transform_image(detector):
    image_size = (160, 160)
    transform_YOLO = transforms.Compose([
        FaceDetectionTransform(detector),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_YOLO

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
        output = self.fc(distance)
        return output

@st.cache_resource
def load_model(path_to_checkpoint, device, dropout_rate=0.75):
    model = SiameseNetwork(dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(path_to_checkpoint, map_location=device, weights_only=True))
    return model

def extract_feature(model, path_to_img, transform, device):
    model.eval()
    model.to(device)

    img = Image.open(path_to_img).convert("RGB")
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    feature = model.base_model(img_tensor)
    return feature

def load_databases(path_to_features="./feature_databases.npy", path_to_names="./name_databases.pkl"):
    feature_databases = np.load(path_to_features)

    name_databases = None
    with open(path_to_names, "rb") as file:
        name_databases = pickle.load(file)
    feature_databases = torch.from_numpy(feature_databases)
    return feature_databases, name_databases

folder_images = "./images/"
os.makedirs(folder_images, exist_ok=True)

uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png"])
theshold = 0.62

if uploaded_file is not None:
    features_databases, name_databases = load_databases()
    model = load_model(path_to_checkpoint="../Nguyen/5/kaggle/working/model_5/best_model.pt", device=device, dropout_rate=0.5)
    detector = load_detector()
    transform_YOLO = transform_image(detector)

    file_path = os.path.join(folder_images, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    start_time = time.time()
    new_feature = extract_feature(model, file_path, transform_YOLO, device)
    distance = torch.abs(new_feature - features_databases)
    output = model.fc(distance)
    results = torch.sigmoid(output)

    index_label = torch.argmax(results).item()

    pred_label = "Unknown"
    max_predict = results[index_label]
    if max_predict >= theshold:
        pred_label = name_databases[index_label]

    img = Image.open(uploaded_file)
    end_time = time.time()
    total_time = end_time-start_time
    column_1, column_2 = st.columns(2)
    with column_1:
        st.image(img)

    with column_2:
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 30px; margin-top: 150px">
                {pred_label} ({total_time:.4f} s)
            </div>
            """,
            unsafe_allow_html=True,
        )

