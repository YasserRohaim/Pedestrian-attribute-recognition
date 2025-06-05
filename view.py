import pandas as pd
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
    "dataset": "views",
    "epochs": 10,
    "patience": 2,  # Early stopping patience
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
train_dataset = ImageDatasetPT(train, "PA-100K/data", LABEL_GROUPS['view'], transform=transform)
val_dataset = ImageDatasetPT(validate, "PA-100K/data", LABEL_GROUPS['view'], transform=transform)
test_dataset = ImageDatasetPT(test, "PA-100K/data/", LABEL_GROUPS['view'], transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create model
class ViewClassifier(nn.Module):
    def __init__(self):
        super(ViewClassifier, self).__init__()
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
        
      
        self.view_head = nn.Sequential(
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)
        )
        
       
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)
        
        output = self.view_head(shared).squeeze()
        
        
        return output

model = ViewClassifier()
model = model.to('cuda' )


view_loss = nn.CrossEntropyLoss()  # Cross-entropy for view classification

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)

# Initialize early stopping
early_stopping = EarlyStopping(patience=run.config.patience, path='best_model.pth')

# Training loop
num_epochs = run.config.epochs
for epoch in range(num_epochs):
    model.train()
    running_view_loss = 0.0
    running_view_corrects = 0
    total_samples = 0
    
    # Initialize tqdm progress bar
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
    
    for batch_idx, (inputs, view_labels) in enumerate(train_loader_tqdm):
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
        view_labels = view_labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        optimizer.zero_grad()
        
        view_output = model(inputs)
        
        # Calculate losses
        loss = view_loss(view_output, torch.argmax(view_labels, dim=1))
       
        loss.backward()
        optimizer.step()
        
        # Calculate batch statistics
        batch_size = inputs.size(0)
        total_samples += batch_size
        
        
        # view accuracy
        view_preds = torch.argmax(view_output, dim=1)
        view_true = torch.argmax(view_labels, dim=1)
        view_correct = torch.sum(view_preds == view_true).item()
        view_acc = view_correct / batch_size
        
        # Update running totals
        running_view_loss += loss * batch_size
        running_view_corrects += view_correct
        
        # Update progress bar with batch metrics
        train_loader_tqdm.set_postfix({
          
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{view_acc:.4f}',
        })
        run.log({
                'view_Loss': loss.item(), 'A_acc': view_acc,
               })
    
    # Calculate epoch statistics
    epoch_view_loss = running_view_loss / total_samples
    epoch_view_acc = running_view_corrects / total_samples
    
    print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
    print(f'View - Loss: {epoch_view_loss:.4f}, Acc: {epoch_view_acc:.4f}')
    
    # Validation
    model.eval()
    val_view_loss = 0.0
    val_view_corrects = 0
    
    with torch.no_grad():
        for inputs, view_labels in val_loader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            view_labels = view_labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            view_output = model(inputs)
            
            # Calculate losses
            val_view_loss += view_loss(view_output, torch.argmax(view_labels, dim=1)).item() * inputs.size(0)
         
            # view accuracy
            view_preds = torch.argmax(view_output, dim=1)
            view_true = torch.argmax(view_labels, dim=1)
            val_view_corrects += torch.sum(view_preds == view_true)
    
    # Calculate validation statistics
    val_view_loss = val_view_loss / len(val_dataset)
    val_view_acc = val_view_corrects.float() / len(val_dataset)
    
    print(f'\nValidation:')
    print(f'view - Loss: {val_view_loss:.4f}, Acc: {val_view_acc:.4f}')
    
    # Log validation metrics
    run.log({
       
        'val_view_Loss': val_view_loss,
        'val_view_Acc': val_view_acc.item(),
    })
    
    # Early stopping check
    early_stopping(val_view_loss, model)
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