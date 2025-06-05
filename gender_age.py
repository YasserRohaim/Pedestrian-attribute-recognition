import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from tqdm import tqdm
from utils import *
import wandb


# Initialize wandb
run = wandb.init(project='demographics', config={
    "learning_rate": 1e-3,
    "architecture": "resnet",
    "dataset": "age and gender",
    "epochs": 50,
    "patience": 5,  # Early stopping patience
})


# Read CSV files
train = pd.read_csv("PA-100K/train.csv")
validate = pd.read_csv("PA-100K/val.csv")
test = pd.read_csv("PA-100K/test.csv")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
train_dataset = ImageDatasetPT(train, "PA-100K/data", ["Female", "AgeOver60", "Age18-60", "AgeLess18"], transform=transform)
val_dataset = ImageDatasetPT(validate, "PA-100K/data", ["Female", "AgeOver60", "Age18-60", "AgeLess18"], transform=transform)
test_dataset = ImageDatasetPT(test, "PA-100K/data/", ["Female", "AgeOver60", "Age18-60", "AgeLess18"], transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create model
class GenderAgeClassifier(nn.Module):
    def __init__(self):
        super(GenderAgeClassifier, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        
        # Freeze early layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze later layers (adjust as needed)
        for param in self.backbone.layer4.parameters():  
            param.requires_grad = True
            
        num_ftrs = self.backbone.fc.in_features  
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
        )
        
        # Gender head (binary classification)
        self.gender_head = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
        # Age head (3-class classification)
        self.age_head = nn.Sequential(
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)
        
        gender_output = self.gender_head(shared).squeeze()
        age_output = self.age_head(shared)
        
        return gender_output, age_output

model = GenderAgeClassifier()
model = model.to('cuda' )

# Define losses
gender_criterion = nn.BCELoss()  # Binary cross-entropy for gender
age_criterion = nn.CrossEntropyLoss()  # Cross-entropy for age classification

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)

# Initialize early stopping
early_stopping = EarlyStopping(patience=run.config.patience, path='best_model.pth')

# Training loop
num_epochs = run.config.epochs
for epoch in range(num_epochs):
    model.train()
    running_gender_loss = 0.0
    running_age_loss = 0.0
    running_gender_corrects = 0
    running_age_corrects = 0
    total_samples = 0
    
    # Initialize tqdm progress bar
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
    
    for batch_idx, (inputs, (gender_labels, age_labels)) in enumerate(train_loader_tqdm):
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
        gender_labels = gender_labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        age_labels = age_labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        optimizer.zero_grad()
        
        gender_output, age_output = model(inputs)
        
        # Calculate losses
        gender_loss = gender_criterion(gender_output, gender_labels)
        age_loss = age_criterion(age_output, torch.argmax(age_labels, dim=1))
        total_loss = gender_loss + age_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Calculate batch statistics
        batch_size = inputs.size(0)
        total_samples += batch_size
        
        # Gender accuracy
        gender_preds = (gender_output > 0.5).float()
        gender_correct = torch.sum(gender_preds == gender_labels.data).item()
        gender_acc = gender_correct / batch_size
        
        # Age accuracy
        age_preds = torch.argmax(age_output, dim=1)
        age_true = torch.argmax(age_labels, dim=1)
        age_correct = torch.sum(age_preds == age_true).item()
        age_acc = age_correct / batch_size
        
        # Update running totals
        running_gender_loss += gender_loss.item() * batch_size
        running_age_loss += age_loss.item() * batch_size
        running_gender_corrects += gender_correct
        running_age_corrects += age_correct
        
        # Update progress bar with batch metrics
        train_loader_tqdm.set_postfix({
            'G_Loss': f'{gender_loss.item():.4f}',
            'G_Acc': f'{gender_acc:.4f}',
            'A_Loss': f'{age_loss.item():.4f}',
            'A_Acc': f'{age_acc:.4f}',
            'Total_Loss': f'{total_loss.item():.4f}'
        })
        run.log({'G_Loss': gender_loss.item(), 'G_Acc': gender_acc, 
                'A_Loss': age_loss.item(), 'A_acc': age_acc,
                'total_loss': total_loss.item()})
    
    # Calculate epoch statistics
    epoch_gender_loss = running_gender_loss / total_samples
    epoch_age_loss = running_age_loss / total_samples
    epoch_gender_acc = running_gender_corrects / total_samples
    epoch_age_acc = running_age_corrects / total_samples
    
    print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
    print(f'Gender - Loss: {epoch_gender_loss:.4f}, Acc: {epoch_gender_acc:.4f}')
    print(f'Age - Loss: {epoch_age_loss:.4f}, Acc: {epoch_age_acc:.4f}')
    
    # Validation
    model.eval()
    val_gender_loss = 0.0
    val_age_loss = 0.0
    val_gender_corrects = 0
    val_age_corrects = 0
    
    with torch.no_grad():
        for inputs, (gender_labels, age_labels) in val_loader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            gender_labels = gender_labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            age_labels = age_labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            gender_output, age_output = model(inputs)
            
            # Calculate losses
            val_gender_loss += gender_criterion(gender_output, gender_labels).item() * inputs.size(0)
            val_age_loss += age_criterion(age_output, torch.argmax(age_labels, dim=1)).item() * inputs.size(0)
            
            # Gender accuracy
            gender_preds = (gender_output > 0.5).float()
            val_gender_corrects += torch.sum(gender_preds == gender_labels.data)
            
            # Age accuracy
            age_preds = torch.argmax(age_output, dim=1)
            age_true = torch.argmax(age_labels, dim=1)
            val_age_corrects += torch.sum(age_preds == age_true)
    
    # Calculate validation statistics
    val_gender_loss = val_gender_loss / len(val_dataset)
    val_age_loss = val_age_loss / len(val_dataset)
    val_gender_acc = val_gender_corrects.float() / len(val_dataset)
    val_age_acc = val_age_corrects.float() / len(val_dataset)
    total_val_loss = val_gender_loss + val_age_loss
    
    print(f'\nValidation:')
    print(f'Gender - Loss: {val_gender_loss:.4f}, Acc: {val_gender_acc:.4f}')
    print(f'Age - Loss: {val_age_loss:.4f}, Acc: {val_age_acc:.4f}')
    print(f'Total Validation Loss: {total_val_loss:.4f}\n')
    
    # Log validation metrics
    run.log({
        'val_G_Loss': val_gender_loss,
        'val_G_Acc': val_gender_acc.item(),
        'val_A_Loss': val_age_loss,
        'val_A_Acc': val_age_acc.item(),
        'val_total_loss': total_val_loss
    })
    
    # Early stopping check
    early_stopping(total_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
print("Loaded best model weights")

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')
print("Saved final model")

# Close wandb run
run.finish()