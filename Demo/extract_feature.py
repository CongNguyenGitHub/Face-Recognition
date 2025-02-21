import pathlib
import pickle
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

import torch.nn as nn
import torch
import os
import numpy as np

def face_detection(image,model):
    """
    Cắt khuôn mặt có độ tin cậy cao nhất từ ảnh đầu vào.

    Args:
        image (str hoặc PIL.Image): Đường dẫn tới ảnh hoặc đối tượng ảnh PIL.
        model: Mô hình YOLO đã huấn luyện để phát hiện khuôn mặt.

    Returns:
        PIL.Image: Ảnh đã cắt chứa khuôn mặt có độ tin cậy cao nhất.
    """

    # Dự đoán với YOLO
    results = model(image)

    # Lấy kết quả dự đoán
    predictions = results.pandas().xyxy[0]

    # Nếu không có khuôn mặt nào được phát hiện, trả về None
    if predictions.empty:
        return None

    # Lấy bounding box có confidence cao nhất
    top_prediction = predictions.loc[predictions['confidence'].idxmax()]
    x_min, y_min, x_max, y_max = int(top_prediction['xmin']), int(top_prediction['ymin']), int(top_prediction['xmax']), int(top_prediction['ymax'])

    # Ảnh đã được cắt
    return image.crop((x_min,y_min,x_max,y_max))

class FaceDetectionTransform:
    def __init__(self, model):
        self.model = model

    def __call__(self, image):
        # Sử dụng hàm face_detection để cắt khuôn mặt
        cropped_image = face_detection(image, self.model)
        # Nếu không phát hiện khuôn mặt nào, trả về ảnh gốc
        return cropped_image if cropped_image is not None else image


detector = torch.hub.load("ultralytics/yolov5", "custom", path="../yolov5n/kaggle/working/model/weights/best.pt")

image_size = (160, 160)
transform_YOLO = transforms.Compose([
    FaceDetectionTransform(detector),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Định nghĩa Siamese Network với lớp fully connected cuối và lớp dropout
class SiameseNetwork(nn.Module):
    def __init__(self, dropout_rate=0.75):  # Thêm tham số dropout_rate
        super(SiameseNetwork, self).__init__()
        self.base_model = InceptionResnetV1(pretrained="vggface2")
        # Freeze tất cả các tham số của base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)  # Khởi tạo lớp dropout
        self.fc = nn.Linear(512, 1)  # 512 là số chiều đầu ra mặc định từ InceptionResnetV1 với classify=False

    def forward(self, input1, input2):
        embedding1 = self.base_model(input1)
        embedding2 = self.base_model(input2)
        distance = torch.abs(embedding1 - embedding2)
        distance = self.dropout(distance)
        output = self.fc(distance)
        return output  # Đảm bảo trả về một Tensor, không phải danh sách

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

image_databases = [
    "Will Smith/Will Smith_041.jpg",
    "Nicole Kidman/Nicole Kidman_065.jpg",
    "Tom Hanks/Tom Hanks_037.jpg",
    "Natalie Portman/Natalie Portman_045.jpg",
    "Tom Cruise/Tom Cruise_004.jpg",
    "Megan Fox/Megan Fox_012.jpg",
    "Jennifer Lawrence/Jennifer Lawrence_091.jpg",
    "Hugh Jackman/Hugh Jackman_074.jpg",
    "Denzel Washington/Denzel Washington_079.jpg",
    "Scarlett Johansson/Scarlett Johansson_149.jpg",
    'Leonardo DiCaprio/Leonardo DiCaprio_007.jpg',
    "Brad Pitt/Brad Pitt_004.jpg",
    "Sandra Bullock/Sandra Bullock_055.jpg",
    "Kate Winslet/Kate Winslet_002.jpg",
    "Angelina Jolie/Angelina Jolie_015.jpg",
    "Robert Downey Jr/Robert Downey Jr_010.jpg",
    "Johnny Depp/Johnny Depp_092.jpg"
]

def get_databases(model, transform, image_databases, device):
    feature_databases = []
    name_databases = []

    for file in image_databases:
        path_to_img = os.path.join("../data/Celebrity Faces Dataset", file)
        feature = extract_feature(model=model, path_to_img=path_to_img, transform=transform, device=device)

        feature_databases.append(feature)
        name_databases.append(file[:file.find('/')])
    feature_databases = torch.vstack(feature_databases)
    return feature_databases, name_databases

model = load_model(path_to_checkpoint="../Nguyen/5/kaggle/working/model_5/best_model.pt", device=device, dropout_rate=0.5)
feature_databases, name_databases = get_databases(model, transform_YOLO, image_databases, device)

with open("./name_databases.pkl", "wb") as file:
    pickle.dump(name_databases, file)

np.save("./feature_databases.npy", feature_databases)