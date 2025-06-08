import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from utils import ImageDatasetPT, LABEL_GROUPS, transform
from clothing import ClothingClassifier
import pandas as pd
from tqdm import tqdm

# Load test data
test = pd.read_csv("PA-100K/test.csv")
target_columns = LABEL_GROUPS['upper_clothing'] + LABEL_GROUPS['lower_clothing'] + LABEL_GROUPS['accessories']

test_dataset = ImageDatasetPT(test, "PA-100K/data", target_columns, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClothingClassifier().to(device)
model.load_state_dict(torch.load("best_clothing_model.pth"))
model.eval()

# Prepare lists for predictions and labels
upper_preds_list, lower_preds_list, accessories_preds_list = [], [], []
upper_labels_list, lower_labels_list, accessories_labels_list = [], [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        # Split labels
        upper_labels = torch.argmax(labels[:, :len(LABEL_GROUPS['upper_clothing'])], dim=1)
        lower_labels = torch.argmax(labels[:, len(LABEL_GROUPS['upper_clothing']):
                                           len(LABEL_GROUPS['upper_clothing']) + len(LABEL_GROUPS['lower_clothing'])], dim=1)
        accessories_labels = labels[:, -len(LABEL_GROUPS['accessories']):]

        # Forward pass
        upper_out, lower_out, accessories_out = model(images)

        # Predictions
        upper_preds = torch.argmax(upper_out, dim=1)
        lower_preds = torch.argmax(lower_out, dim=1)
        accessories_preds = (torch.sigmoid(accessories_out) > 0.5).float()

        # Store results
        upper_preds_list.append(upper_preds.cpu())
        lower_preds_list.append(lower_preds.cpu())
        accessories_preds_list.append(accessories_preds.cpu())

        upper_labels_list.append(upper_labels.cpu())
        lower_labels_list.append(lower_labels.cpu())
        accessories_labels_list.append(accessories_labels.cpu())

# Convert to arrays
upper_preds_all = torch.cat(upper_preds_list).numpy()
lower_preds_all = torch.cat(lower_preds_list).numpy()
accessories_preds_all = torch.cat(accessories_preds_list).numpy()

upper_labels_all = torch.cat(upper_labels_list).numpy()
lower_labels_all = torch.cat(lower_labels_list).numpy()
accessories_labels_all = torch.cat(accessories_labels_list).numpy()

# Compute accuracies
upper_acc = accuracy_score(upper_labels_all, upper_preds_all)
lower_acc = accuracy_score(lower_labels_all, lower_preds_all)
accessories_acc = accuracy_score(accessories_labels_all, accessories_preds_all)

# Also compute F1 for accessories since it's multi-label
accessories_f1 = f1_score(accessories_labels_all, accessories_preds_all, average='macro', zero_division=1)

# Output results
print(f"\nTest Accuracy:")
print(f"Upper Clothing Accuracy: {upper_acc:.4f}")
print(f"Lower Clothing Accuracy: {lower_acc:.4f}")
print(f"Accessories Accuracy (per-label mean): {accessories_acc:.4f}")
print(f"Accessories F1 Score: {accessories_f1:.4f}")
