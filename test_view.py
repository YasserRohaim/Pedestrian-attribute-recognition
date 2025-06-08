import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from utils import LABEL_GROUPS, transform, ImageDatasetPT


# === Model Definition ===
class ViewClassifier(nn.Module):
    def __init__(self):
        super(ViewClassifier, self).__init__()
        self.backbone = models.resnet152(pretrained=False)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.shared_features = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
        )

        self.view_head = nn.Sequential(
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)
        output = self.view_head(shared).squeeze()
        return output


# === Load Test Data ===
test_csv = "PA-100K/test.csv"
image_root = "PA-100K/data"
view_columns = LABEL_GROUPS['view']  # Should be ["Front", "Side", "Back"]

test_df = pd.read_csv(test_csv)
test_dataset = ImageDatasetPT(test_df, image_root, view_columns, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViewClassifier()
model.load_state_dict(torch.load("view_model.pth", map_location=device))
model = model.to(device)
model.eval()

# === Evaluate ===
correct = 0
total = 0

with torch.no_grad():
    for inputs, view_labels in tqdm(test_loader, desc="Testing View Estimation"):
        inputs = inputs.to(device)
        view_labels = view_labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        true = torch.argmax(view_labels, dim=1)

        correct += (preds == true).sum().item()
        total += view_labels.size(0)

accuracy = correct / total

print("\n--- View Estimation Test Results ---")
print(f"View Accuracy: {accuracy:.4f}")
