import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

LABEL_GROUPS = {
    "gender": ['Female'],  # Binary classification (0=Male, 1=Female)
    "age": ['AgeOver60', 'Age18-60', 'AgeLess18'],  # 3-class classification
    "view": ['Front', 'Side', 'Back'],
    "accessories": ['Hat', 'Glasses', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront'],
    "upper_clothing": ['ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LongCoat'],
    "lower_clothing": ['LowerStripe', 'LowerPattern', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots'],
}
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
      


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
        
        if "Female" in self.label_columns and len(self.label_columns)==4:
            gender_label = self.data.iloc[idx]["Female"]
            age_labels = self.data.iloc[idx][["AgeOver60", "Age18-60", "AgeLess18"]].values.astype(np.float32)
            all_labels = np.concatenate([[gender_label], age_labels])
            labels = torch.tensor(all_labels, dtype=torch.float32)
            return image, labels

        else:

            labels = self.data.iloc[idx][self.label_columns].values.astype(np.float32)
            labels = torch.tensor(labels)
            return image, labels
        
