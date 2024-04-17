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

class TestDataset(Dataset):
    def __init__(self, csv_file, base_path, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.data_frame.iloc[idx, -1])
        label = int(self.data_frame.iloc[idx, 4])  # Update this depending on label column
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = TestDataset(
    csv_file='pbc_attr_v1_val.csv',
    base_path='./data/PBC/',
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Helper Functions and Classes
def get_image_encoder(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = torch.nn.Identity()
    return model, in_features

class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model, output_index=0):
        super().__init__()
        self.model = model
        self.output_index = output_index

    def forward(self, x):
        return self.model(x)[self.output_index]

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

# Image Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = 'segmented_image.jpg'
original_img = Image.open(img_path)
img = transform(original_img).unsqueeze(0).to(device)
# img = transform(original_img).unsqueeze(0).to(device)

# Prediction and Probability Calculation
with torch.no_grad():
    predictions = model(img)

# Function to print the state dictionary
def print_model_state_dict(model):
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        # Uncomment the following line to print the actual values of the weights
        # print(model.state_dict()[param_tensor])

# Call the function to print the state dictionary
print_model_state_dict(model)

attribute_names = ["nucleus_shape"]
attribute_values = [
    ["irregular", "segmented-bilobed", "segmented-multilobed", "unsegmented-band", "unsegmented-indented", "unsegmented-round"]
]

for i, logits in enumerate(predictions):
    probabilities = F.softmax(logits, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1)
    predicted_label = attribute_values[i][predicted_index.item()]
    all_probabilities = probabilities.squeeze().tolist()
    
    print(f"Predictions for {attribute_names[i]}:")
    for class_index, class_probability in enumerate(all_probabilities):
        print(f"{attribute_values[i][class_index]}: {class_probability*100:.2f}%")
    print(f"Most likely: {predicted_label}, Probability: {all_probabilities[predicted_index.item()]*100:.2f}%\n")

# Grad-CAM Visualization
target_layer = model.image_encoder.layer4[-1]
gradcam_model_wrapper = GradCAMWrapper(model, output_index=0)
gradcam = GradCAM(gradcam_model_wrapper, target_layer)
gradcam_pp = GradCAMpp(gradcam_model_wrapper, target_layer)

mask, _ = gradcam(img)
heatmap, result = visualize_cam(mask, img)
if heatmap.ndim == 3 and heatmap.shape[0] == 3:
    heatmap = heatmap[0]

img_denorm = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_denorm = torch.clamp(img_denorm, 0, 1)

# img_pil = to_pil_image(img_denorm.squeeze())

# fig, axarr = plt.subplots(1, 3, figsize=(12, 4))
# axarr[0].imshow(img_pil)
# axarr[0].title.set_text('Original Image')
# axarr[1].imshow(heatmap.squeeze().cpu(), cmap='jet')
# axarr[1].title.set_text('Grad-CAM')
# axarr[2].imshow(img_pil)
# axarr[2].imshow(heatmap.squeeze().cpu(), cmap='jet', alpha=0.4)
# axarr[2].title.set_text('Result on SE blocks implementation')
# for ax in axarr:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# Convert the normalized tensor back to a PIL image for the original image
# Apply a colormap (e.g., 'jet') to the heatmap

heatmap_norm = (heatmap.squeeze().cpu() - heatmap.min()) / (heatmap.max() - heatmap.min())
colored_heatmap = plt.cm.jet(heatmap_norm.numpy())  # This applies the 'jet' colormap

# Convert colored heatmap to an image (discard the alpha channel)
heatmap_img = Image.fromarray((colored_heatmap[:, :, :3] * 255).astype(np.uint8))

# Combine the heatmap with the original image
img_pil = to_pil_image(img_denorm.squeeze()).convert("RGB")
heatmap_on_image = Image.blend(img_pil, heatmap_img, alpha=0.4)

# Display the images
img_pil.show(title="Original Image")
heatmap_img.show(title="Grad-CAM")
heatmap_on_image.show(title="Result on SE blocks implementation")

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
# Calculate accuracy
accuracy = evaluate_accuracy(model, test_loader, device)
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')