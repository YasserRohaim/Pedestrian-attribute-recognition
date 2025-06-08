import pandas as pd
import torch
from torch.utils.data import  DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from utils import LABEL_GROUPS, transform, ImageDatasetPT, EarlyStopping
import wandb
class ClothingClassifier(nn.Module):
    def __init__(self):
        super(ClothingClassifier, self).__init__()
        self.backbone = models.resnet152(pretrained=True)
        
        # Freeze early layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze later layers
        for param in self.backbone.layer4.parameters():  
            param.requires_grad = True
            
        num_ftrs = self.backbone.fc.in_features  
        self.backbone.fc = nn.Identity()
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Task-specific heads
        self.upper_head = nn.Sequential(nn.Linear(1024, len(LABEL_GROUPS['upper_clothing']),nn.Softmax(dim=1)))
        self.lower_head = nn.Sequential(nn.Linear(1024, len(LABEL_GROUPS['lower_clothing']), nn.Softmax(dim=1)))
        self.accessories_head = nn.Linear(1024, len(LABEL_GROUPS['accessories']))
        
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)
        
        upper_out = self.upper_head(shared)
        lower_out = self.lower_head(shared)
        accessories_out = self.accessories_head(shared)
        
        return upper_out, lower_out, accessories_out
if __name__=="__main__":
    # Initialize wandb (focus on clothing/accessories)
    run = wandb.init(project='demographics', config={
        "learning_rate": 1e-4,
        "architecture": "resnet",
        "dataset": "PA-100K",
        "epochs": 20,
        "patience": 3,
    })

    # Read CSV files (using only clothing/accessory columns)
    target_columns = LABEL_GROUPS['upper_clothing'] + LABEL_GROUPS['lower_clothing'] + LABEL_GROUPS['accessories']
    train = pd.read_csv("PA-100K/train.csv")
    validate = pd.read_csv("PA-100K/val.csv")
    test = pd.read_csv("PA-100K/test.csv")



    # Create datasets
    train_dataset = ImageDatasetPT(train, "PA-100K/data", target_columns, transform=transform)
    val_dataset = ImageDatasetPT(validate, "PA-100K/data", target_columns, transform=transform)
    test_dataset = ImageDatasetPT(test, "PA-100K/data", target_columns, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model = ClothingClassifier().to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define losses
    upper_criterion = nn.CrossEntropyLoss()  # Single-label classification
    lower_criterion = nn.CrossEntropyLoss()  # Single-label classification
    accessories_criterion = nn.BCEWithLogitsLoss()  # Multi-label classification

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)

    # Early stopping
    early_stopping = EarlyStopping(patience=run.config.patience, path='best_clothing_model.pth')

    # Training loop
    for epoch in range(run.config.epochs):
        model.train()
        running_losses = {'upper': 0.0, 'lower': 0.0, 'accessories': 0.0}
        running_corrects = {'upper': 0, 'lower': 0, "accessories":0}
        running_accessories_positives = 0
        total_samples = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{run.config.epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(train_loader_tqdm):

            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Split targets
            upper_target = torch.argmax(targets[ :,:len(LABEL_GROUPS['upper_clothing'])], dim=1).to('cuda' if torch.cuda.is_available() else 'cpu')
        
            lower_target = torch.argmax(targets[ :, len(LABEL_GROUPS['upper_clothing']): 
                                        len(LABEL_GROUPS['upper_clothing'])+len(LABEL_GROUPS['lower_clothing'])], 
                                    dim=1).to('cuda' if torch.cuda.is_available() else 'cpu')
        
            accessories_target = targets[:, -len(LABEL_GROUPS['accessories']):].float().to('cuda' if torch.cuda.is_available() else 'cpu')
        
            
            optimizer.zero_grad()
            
            # Forward pass
            upper_out, lower_out, accessories_out = model(inputs)
            upper_out=upper_out.to('cuda' if torch.cuda.is_available() else 'cpu')
            lower_out= lower_out.to('cuda' if torch.cuda.is_available() else 'cpu')
            accessories_out=accessories_out.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Calculate losses
            losses = {
                'upper': upper_criterion(upper_out, upper_target),
                'lower': lower_criterion(lower_out, lower_target),
                'accessories': accessories_criterion(accessories_out, accessories_target)
            }
            total_loss = sum(losses.values())
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Calculate metrics
            upper_preds = torch.argmax(upper_out, dim=1)
            running_corrects['upper'] += (upper_preds == upper_target).sum().item()
            
            lower_preds = torch.argmax(lower_out, dim=1)
            running_corrects['lower'] += (lower_preds == lower_target).sum().item()
            
            accessories_preds = (torch.sigmoid(accessories_out) > 0.5).float()
            running_corrects['accessories'] += (accessories_preds == accessories_target).sum().item()
            
            # Update running losses
            for k in losses:
                running_losses[k] += losses[k].item() * batch_size
            
            # Update progress bar
            train_loader_tqdm.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Upper_Acc': f'{(upper_preds == upper_target).float().mean().item():.4f}',
                'Lower_Acc': f'{(lower_preds == lower_target).float().mean().item():.4f}',
                'accesory_acc': f'{(accessories_preds == accessories_target).float().mean().item():.4f}'
            })
            
            # Log batch metrics
            run.log({
                'batch_total_loss': total_loss.item(),
                'batch_upper_loss': losses['upper'].item(),
                'batch_lower_loss': losses['lower'].item(),
                'batch_accessories_loss': losses['accessories'].item()
            })
        
        # Calculate epoch metrics
        epoch_metrics = {
            'train_upper_loss': running_losses['upper'] / total_samples,
            'train_lower_loss': running_losses['lower'] / total_samples,
            'train_accessories_loss': running_losses['accessories'] / total_samples,
            'train_upper_acc': running_corrects['upper'] / total_samples,
            'train_lower_acc': running_corrects['lower'] / total_samples,
            'train_accesory_acc': running_corrects['accessories'] / total_samples,        }
        
        print(f"\nEpoch {epoch+1} Clothing Summary:")
        for k, v in epoch_metrics.items():
            print(f"{k}: {v:.4f}")
            run.log({k: v})
        
        # Validation
        model.eval()
        val_metrics = {
            'val_upper_loss': 0.0, 'val_lower_loss': 0.0, 'val_accessories_loss': 0.0,
            'val_upper_acc': 0, 'val_lower_acc': 0, 'val_accessories_positives': 0
        }
        val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                batch_size = inputs.size(0)
                val_samples += batch_size
                
                # Split targets
                upper_target = torch.argmax(targets[:, :len(LABEL_GROUPS['upper_clothing'])], dim=1).to('cuda' if torch.cuda.is_available() else 'cpu')
        
                lower_target = torch.argmax(targets[:, 
                                            len(LABEL_GROUPS['upper_clothing']): 
                                            len(LABEL_GROUPS['upper_clothing'])+len(LABEL_GROUPS['lower_clothing'])], 
                                        dim=1).to('cuda' if torch.cuda.is_available() else 'cpu')
        
                accessories_target = targets[:, -len(LABEL_GROUPS['accessories']):].float().to('cuda' if torch.cuda.is_available() else 'cpu')
        
                
                # Forward pass
                upper_out, lower_out, accessories_out = model(inputs)
                
                # Calculate losses
                val_metrics['val_upper_loss'] += upper_criterion(upper_out, upper_target).item() * batch_size
                val_metrics['val_lower_loss'] += lower_criterion(lower_out, lower_target).item() * batch_size
                val_metrics['val_accessories_loss'] += accessories_criterion(accessories_out, accessories_target).item() * batch_size
                
                # Calculate accuracies
                upper_preds = torch.argmax(upper_out, dim=1)
                val_metrics['val_upper_acc'] += (upper_preds == upper_target).sum().item()
                
                lower_preds = torch.argmax(lower_out, dim=1)
                val_metrics['val_lower_acc'] += (lower_preds == lower_target).sum().item()
                
                # Accessories
                accessories_preds = (torch.sigmoid(accessories_out) > 0.5).float()
                val_metrics['val_accessories_positives'] += accessories_preds.sum().item()
        
        # Normalize validation metrics
        for k in ['val_upper_loss', 'val_lower_loss', 'val_accessories_loss']:
            val_metrics[k] /= val_samples
            
        for k in ['val_upper_acc', 'val_lower_acc']:
            val_metrics[k] /= val_samples
            
        val_metrics['val_accessories_pos_rate'] = val_metrics['val_accessories_positives'] / (val_samples * len(LABEL_GROUPS['accessories']))
        
        print("\nValidation Summary:")
        for k, v in val_metrics.items():
            if k != 'val_accessories_positives':  # Don't print raw count
                print(f"{k}: {v:.4f}")
                run.log({k: v})
        
        # Early stopping check
        early_stopping(val_metrics['val_upper_loss'] + val_metrics['val_lower_loss'] + val_metrics['val_accessories_loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model
    model.load_state_dict(torch.load('best_clothing_model.pth'))
    print("Loaded best model weights")

    # Save the final model
    torch.save(model.state_dict(), 'final_clothing_model.pth')
    print("Saved final model")

    # Close wandb run
    run.finish()