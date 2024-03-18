import os
import pandas as pd
import torch
import torch.nn as nn
import random
import pandas as pd
import torch.optim as optim

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed=42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Optional: Set for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


class PBCTorchDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.cell_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Define mapping for categorical attributes
        self.label_mapping = {
            'cell_size': {'small': 0, 'big': 1},
            'cell_shape': {'round': 0, 'irregular': 1},
            
            
            # Add other attributes here following the same structure
        }
        
    def __len__(self):
        return len(self.cell_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.cell_frame.iloc[idx]['path'])
        image = Image.open(img_path).convert('RGB')  # Load image as PIL Image
        labels = self._encode_attributes(idx)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels
    
    def _encode_attributes(self, idx):
        # Encode attributes into binary vector
        labels = []
        for attr, mapping in self.label_mapping.items():
            attr_value = self.cell_frame.iloc[idx][attr]
            encoded = [0] * len(mapping)  # Assuming binary attributes for simplicity
            if attr_value in mapping:
                encoded[mapping[attr_value]] = 1
            labels.extend(encoded)
        return torch.tensor(labels, dtype=torch.float32)
    
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_model(num_classes):
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Adjusting for our number of classes
    return model

def load_or_train_model(model_path, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10, device="cuda"):
    if os.path.exists(model_path):
        print("Loading saved model from:", model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training new model.")
        train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device)
        torch.save(model.state_dict(), model_path)
        print("Model saved to:", model_path)
    model.to(device)

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10, device="cuda"):
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()  # Deep copy the model state

    print("Training completed.")
    
    # After completing all epochs, load the best model and evaluate it on the test set.
    model.load_state_dict(best_model_state)
    test_loss, test_accuracy, test_f1 = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

    # Optionally save the best model
    torch.save(best_model_state, "best_model.pth")
    print("Best model saved to best_model.pth")
          
def evaluate_model(model, dataloader, criterion, device="cuda", threshold=0.5):
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Convert model outputs to binary labels
            predicted_labels = (torch.sigmoid(outputs) > threshold).int()
            all_predictions.append(predicted_labels.cpu())
            all_true_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_true_labels = torch.cat(all_true_labels, dim=0).numpy()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='macro')
    
    return avg_loss, accuracy, f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)  # Ensure seed is set before model initialization for reproducibility

    # Instantiate the model and move it to the appropriate device
    model = create_model(num_classes=4)  # Adjust num_classes based on your task
    model.to(device)

    # Define the loss function (criterion) and the optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare the data loaders
    train_dataset = PBCTorchDataset(csv_file='./pbc_attr_v1_train.csv', img_dir='./data/PBC/', transform=transform)
    val_dataset = PBCTorchDataset(csv_file='./pbc_attr_v1_val.csv', img_dir='./data/PBC/', transform=transform)
    test_dataset = PBCTorchDataset(csv_file='./pbc_attr_v1_test.csv', img_dir='./data/PBC/', transform=transform)

    num_workers = 8  # Adjust based on your system's capabilities
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    num_epochs = 10  # Adjust based on your preference
    model_path = "best_model.pth"  # Path where the best model is saved or to be saved

    # Decide to load or train the model based on the existence of the model file
    load_or_train_model(model_path, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device)