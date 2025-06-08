import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from utils import LABEL_GROUPS, transform, ImageDatasetPT, EarlyStopping
import wandb

class AccessoriesClassifier(nn.Module):
    def __init__(self):
        super(AccessoriesClassifier, self).__init__()
        self.backbone = models.resnet152(pretrained=True)

        # Freeze early layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze later layers
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(LABEL_GROUPS['accessories']))  # Multi-label output
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize wandb
    run = wandb.init(project='accessories-only', config={
        "learning_rate": 1e-4,
        "architecture": "resnet152",
        "dataset": "PA-100K",
        "epochs": 20,
        "patience": 3,
        "batch_size": 32
    })

    # Read CSV files and extract only accessories targets
    accessories_columns = LABEL_GROUPS['accessories']
    train = pd.read_csv("PA-100K/train.csv")
    val = pd.read_csv("PA-100K/val.csv")

    # Create datasets
    train_dataset = ImageDatasetPT(train, "PA-100K/data", accessories_columns, transform=transform)
    val_dataset = ImageDatasetPT(val, "PA-100K/data", accessories_columns, transform=transform)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=run.config.batch_size, shuffle=False)

    # Model, loss, optimizer
    model = AccessoriesClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)
    early_stopping = EarlyStopping(patience=run.config.patience, path='best_accessories_model.pth')

    # Training loop
    for epoch in range(run.config.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{run.config.epochs}", leave=False)
        for inputs, targets in train_loader_tqdm:
            inputs, targets = inputs.to(device), targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Metrics
            preds = (torch.sigmoid(outputs) > 0.5).float()
            batch_correct = (preds == targets).sum().item()
            batch_total = targets.numel()
            acc = batch_correct / batch_total

            epoch_loss += loss.item() * inputs.size(0)
            correct += batch_correct
            total += batch_total

            # Log & display
            train_loader_tqdm.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
            run.log({
                'batch_loss': loss.item(),
                'batch_acc': acc
            })

        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"\n[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        run.log({
            'train_loss': epoch_loss,
            'train_acc': epoch_acc
        })

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == targets).sum().item()
                val_total += targets.numel()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        run.log({
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(torch.load('best_accessories_model.pth'))
    print("Loaded best model.")

    # Save final model
    torch.save(model.state_dict(), 'final_accessories_model.pth')
    print("Saved final model.")

    run.finish()
