import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from utils import transform, LABEL_GROUPS
from gender_age import GenderAgeClassifier
from view import ViewClassifier
from clothing import ClothingClassifier
from accessories import AccessoriesClassifier
from ultralytics import YOLO
import cv2
from PIL import Image

class WrapperModel(nn.Module):
    def __init__(self, demographics, view, clothing, accessories):
        super(WrapperModel, self).__init__()
        self.demographics_model = demographics
        self.view_model = view
        self.clothing_model = clothing
        self.accessories_model = accessories

    def forward(self, image):
        demographics_preds = self.demographics_model(image)
        view_preds = self.view_model(image)
        clothing_preds = self.clothing_model(image)[:2]
        accessories_preds = self.accessories_model(image)
        return demographics_preds, view_preds, clothing_preds, accessories_preds

# Load models
view_model = ViewClassifier()
view_model.load_state_dict(torch.load('models/view_model.pth'))
view_model.to('cuda').eval()

demographics_model = GenderAgeClassifier()
demographics_model.load_state_dict(torch.load('models/age_gender_model.pth'))
demographics_model.to('cuda').eval()

clothing_model = ClothingClassifier()
clothing_model.load_state_dict(torch.load('models/best_clothing_model.pth'))
clothing_model.to('cuda').eval()

accessories_model = AccessoriesClassifier()
accessories_model.load_state_dict(torch.load('models/best_accessories_model.pth'))
accessories_model.to('cuda').eval()

model = WrapperModel(
    demographics=demographics_model,
    view=view_model,
    clothing=clothing_model,
    accessories=accessories_model
).eval().cuda()

yolo_model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("test.mp4")

# Get video writer setup
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use .avi extension then
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
print("FPS:", fps, "Width:", width, "Height:", height)


frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame, classes=[0], conf=0.4)

    if results and len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]

            try:
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                input_tensor = transform(cropped_pil).unsqueeze(0).cuda()

                with torch.no_grad():
                    demo_pred, view_pred, clothing_pred, accessories_pred = model(input_tensor)
                    gender_pred, age_pred = demo_pred
                    upper_clothing, lower_clothing = clothing_pred

                gender = 'female' if (gender_pred.item() > 0.6) else 'male'
                age = age_pred.argmax(dim=-1).item()
                view_cls = view_pred.argmax(dim=-1).item()
                upper_clothing = upper_clothing.argmax(dim=-1).item()
                lower_clothing = lower_clothing.argmax(dim=-1).item()
                accessories = (accessories_pred > 0.5).int().cpu().numpy()

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f'View: {LABEL_GROUPS["view"][view_cls]} age: {LABEL_GROUPS["age"][age]}, gender: {gender}'
                cv2.putText(frame, label_text, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            except Exception as e:
                print(f"Error processing crop: {e}")
                continue

    out.write(frame)
    cv2.imshow("Attributes Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
