from flask import Flask, request, jsonify
import io
from PIL import Image
import torch
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b7
import torch.nn as nn
from pathlib import Path
import sys

app = Flask(__name__)

# Setup your model and variables here (adapted from your script)
yolov5_path = Path('D:/Vscode/NEPHLE3.0/yolov5')
if str(yolov5_path) not in sys.path:
    sys.path.insert(0, str(yolov5_path))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes

weights = yolov5_path / 'yolov5m.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = DetectMultiBackend(str(weights), device=device)
names = yolo_model.names
imgsz = 640
stride = yolo_model.stride

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

num_classes = 7
model = efficientnet_b7(pretrained=False)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model = model.to(device)
model.load_state_dict(torch.load("D:/Vscode/NEPHLE3.0/yolov5/efficientnet_b7_har_finetuned.pth", map_location=device))
model.eval()

class_labels = [
    'sitting', 'using_laptop', 'hugging', 'sleeping', 'drinking', 'clapping',
    'dancing', 'cycling', 'calling', 'laughing', 'eating', 'fighting',
    'listening_to_music', 'running', 'texting'
]

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

prediction_history = deque(maxlen=30)
confidence_threshold = 0.6

def letterbox(img, new_shape=(640, 640), stride=32, auto=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    scale = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*scale)), int(round(shape[0]*scale)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, scale, (dw, dh)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image'].read()
    image = Image.open(io.BytesIO(file)).convert('RGB')
    img0 = np.array(image)[:, :, ::-1].copy()  # RGB to BGR for OpenCV
    
    # Preprocess for YOLOv5
    img, scale, pad = letterbox(img0, imgsz, stride)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        pred = yolo_model(img_tensor, augment=False)
        pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)[0]
    
    actions = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], img0.shape)
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_name = names[int(cls)]
            if cls_name == 'person':
                crop = img0[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result = pose.process(crop_rgb)
                
                if result.pose_landmarks:
                    crop_pil = Image.fromarray(crop_rgb)
                    input_tensor = transform(crop_pil).unsqueeze(0).to(device)
                    
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, predicted = torch.max(probs, 1)
                    conf = conf.item()
                    pred_label = class_labels[predicted.item()]
                    
                    if conf >= confidence_threshold:
                        prediction_history.append(pred_label)
                    
                    if len(prediction_history) == 0:
                        prediction_history.append(pred_label)
                    
                    stable_label = Counter(prediction_history).most_common(1)[0][0]
                    actions.append({'label': stable_label, 'confidence': conf})
    
    if actions:
        # Return the first detected person's action (or adapt as needed)
        return jsonify(actions[0])
    else:
        return jsonify({'label': 'no_person_detected', 'confidence': 0.0})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
