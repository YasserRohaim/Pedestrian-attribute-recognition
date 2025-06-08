import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from gender_age import GenderAgeClassifier
from utils import ImageDatasetPT, transform

# === Load Test Data ===
test_csv = "PA-100K/test.csv"
image_root = "PA-100K/data"
label_columns = ["Female", "AgeOver60", "Age18-60", "AgeLess18"]

test_df = pd.read_csv(test_csv)
test_dataset = ImageDatasetPT(test_df, image_root, label_columns, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderAgeClassifier()
model.load_state_dict(torch.load("age_gender_model.pth", map_location=device))
model = model.to(device)
model.eval()

# === Evaluate ===
gender_correct = 0
age_correct = 0
gender_total = 0
age_total = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        gender_labels=labels[:,0]
        age_labels=labels[:,1:]
        inputs = inputs.to(device)
        gender_labels = gender_labels.to(device)
        age_labels = age_labels.to(device)

        gender_output, age_output = model(inputs)

        gender_preds = (gender_output > 0.5).float()
        gender_correct += (gender_preds == gender_labels).sum().item()
        gender_total += gender_labels.size(0)

        age_preds = torch.argmax(age_output, dim=1)
        age_true = torch.argmax(age_labels, dim=1)
        age_correct += (age_preds == age_true).sum().item()
        age_total += age_labels.size(0)

# === Print Accuracy ===
gender_acc = gender_correct / gender_total
age_acc = age_correct / age_total

print("\n--- Test Results ---")
print(f"Gender Accuracy: {gender_acc:.4f}")
print(f"Age Accuracy: {age_acc:.4f}")
