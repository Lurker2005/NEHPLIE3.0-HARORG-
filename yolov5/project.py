
import torch
import cv2
import sys
from pathlib import Path
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b7
import torch.nn as nn

# Setup YOLOv5 path and import
yolov5_path = Path(r'D:/Vscode/NEPHLE3.0/yolov5')
if str(yolov5_path) not in sys.path:
    sys.path.insert(0, str(yolov5_path))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes

# Load YOLOv5 model for person detection
weights = yolov5_path / 'yolov5m.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = DetectMultiBackend(str(weights), device=device)
names = yolo_model.names
imgsz = 640
stride = yolo_model.stride

# Open webcam
cap = cv2.VideoCapture(0)

# Setup MediaPipe pose estimator
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Setup EfficientNet-B7 HAR model
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
model.load_state_dict(torch.load(r"D:\Vscode\NEPHLE3.0\yolov5\efficientnet_b7_har_finetuned.pth", map_location=device))
model.eval()

# Human activity class labels
class_labels = [
    'sitting', 'using_laptop', 'hugging', 'sleeping', 'drinking', 'clapping',
    'dancing', 'cycling', 'calling', 'laughing', 'eating', 'fighting',
    'listening_to_music', 'running', 'texting'
]

# Image preprocessing for EfficientNet
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Smoothing buffer for stable predictions
prediction_history = deque(maxlen=30)
confidence_threshold = 0.6  # Minimum confidence to consider prediction

# Letterbox function to resize + pad image maintaining aspect ratio
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

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img0 = frame.copy()
    img, scale, pad = letterbox(img0, imgsz, stride)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Run YOLOv5 person detection
    pred = yolo_model(img_tensor, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], img0.shape)

        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_idx = int(cls)
            cls_name = names[cls_idx]
            if cls_name == 'person':
                crop = img0[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result = pose.process(crop_rgb)

                # Draw pose landmarks on crop
                if result.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(crop, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    frame[y1:y2, x1:x2] = crop
                
                # Prepare crop input for HAR model
                crop_pil = Image.fromarray(crop_rgb)
                input_tensor = transform(crop_pil).unsqueeze(0).to(device)
                
                # Predict action
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, predicted = torch.max(probs, 1)
                    conf = conf.item()
                    pred_label = class_labels[predicted.item()]

                # Add prediction if confidence high enough
                if conf >= confidence_threshold:
                    prediction_history.append(pred_label)

                # Fallback if history is empty
                if len(prediction_history) == 0:
                    prediction_history.append(pred_label)

                stable_label = Counter(prediction_history).most_common(1)[0][0]

                print(f'Predicted action (smoothed): {stable_label}')
                cv2.putText(frame, f'Action: {stable_label}', (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{cls_name} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('YOLOv5 + MediaPipe + HAR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
