import torch
from torchvision import models, transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Custom imports
from attribute_predictor_SEB import AttributePredictor  
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
import numpy as np
import pandas as pd
import os

class TestDataset(Dataset):
    def __init__(self, csv_file, base_path, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.base_path = base_path
        self.transform = transform

        # Define a mapping of labels to indices
        self.label_mapping = {
            "irregular": 0,
            "segmented-bilobed": 1,
            "segmented-multilobed": 2,
            "unsegmented-band": 3,
            "unsegmented-indented": 4,
            "unsegmented-round": 5
        }

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.data_frame.iloc[idx, -1])
        label_name = self.data_frame.iloc[idx, 4]
        label = self.label_mapping[label_name]  # Convert label names to indices

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # outputs should be a list of tensors

            # If you're evaluating a specific attribute, select the correct tensor
            # For instance, if the first tensor corresponds to `nucleus_shape`
            outputs_for_attribute = outputs[0]  # Adjust index based on your model's output

            _, predicted = torch.max(outputs_for_attribute, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_file='pbc_attr_v1_val.csv'
base_path='./data/PBC/'


# Create the dataset and DataLoader
test_dataset = TestDataset(csv_file=csv_file, base_path=base_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Helper Functions and Classes
def get_image_encoder(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = torch.nn.Identity()
    return model, in_features



def denormalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    return tensor * std[:, None, None] + mean[:, None, None]

# Model Setup
image_encoder, image_encoder_output_dim = get_image_encoder(pretrained=True)
attribute_sizes = [6]  # Replace with actual sizes of your tasks

model = AttributePredictor(attribute_sizes, image_encoder_output_dim, image_encoder)
checkpoint = torch.load('./log/best_model_SEB.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

img_path = 'segmented_image.jpg'
original_img = Image.open(img_path)
img = transform(original_img).unsqueeze(0).to(device)
# img = transform(original_img).unsqueeze(0).to(device)

# Prediction and Probability Calculation
with torch.no_grad():
    predictions = model(img)

attribute_names = ["nucleus_shape"]
attribute_values = [
    ["irregular", "segmented-bilobed", "segmented-multilobed", "unsegmented-band", "unsegmented-indented", "unsegmented-round"]
]

# Calculate accuracy
accuracy = evaluate_accuracy(model, test_loader, device)
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')