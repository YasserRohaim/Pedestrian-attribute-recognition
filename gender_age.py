import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from tqdm import tqdm
import wandb

# Initialize wandb
run = wandb.init(project='demographics', config={
    "learning_rate": 1e-3,
    "architecture": "efficientnet",
    "dataset": "age and gender",
    "epochs": 50,
    "patience": 5,  # Early stopping patience
})

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_model.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved
            delta (float): Minimum change in the monitored quantity to qualify as an improvement
            path (str): Path for the checkpoint to be saved to
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Rest of your code remains the same until the training loop
LABEL_GROUPS = {
    "gender": ['Female'],  # Binary classification (0=Male, 1=Female)
    "age": ['AgeOver60', 'Age18-60', 'AgeLess18'],  # 3-class classification
    "view": ['Front', 'Side', 'Back'],
    "accessories": ['Hat', 'Glasses', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront'],
    "upper_clothing": ['ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LongCoat'],
    "lower_clothing": ['LowerStripe', 'LowerPattern', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots'],
}

class ImageDatasetPT(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_folder: str, label_group: list, image_size=(128, 256), transform=None):
        self.data = dataframe
        self.img_folder = img_folder
        self.image_size = image_size
        self.transform = transform
        
        self.label_columns = label_group
        self.data = self.data[["Image"] + self.label_columns]  # Keep only relevant columns
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image"]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        # Gender label (binary)
        gender_label = torch.tensor(self.data.iloc[idx]["Female"]).float()
        
        # Age label (one-hot encoded for 3 classes)
        age_labels = self.data.iloc[idx][["AgeOver60", "Age18-60", "AgeLess18"]].values.astype(np.float32)
        age_label = torch.tensor(age_labels)
        
        return image, (gender_label, age_label)

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
        for param in self.backbone.layer4.parameters():  # Changed for ResNet
            param.requires_grad = True
            
        num_ftrs = self.backbone.fc.in_features  # Changed for ResNet
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