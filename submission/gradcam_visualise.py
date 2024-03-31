

import torch
from torchvision import models, transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
import matplotlib.pyplot as plt
import os
from PIL import Image

from attribute_predictor import AttributePredictor  # Make sure this is correctly imported
import cv2
def get_image_encoder(pretrained=True):
    # Initialize the pre-trained ResNet model
    model = models.resnet50(pretrained=pretrained)
    
    # Capture the in_features from the original fc layer before replacing it
    in_features = model.fc.in_features
    
    # Replace the final layer with Identity to use it as a feature extractor
    model.fc = torch.nn.Identity()
    
    return model, in_features

class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model, output_index=0):
        super().__init__()
        self.model = model
        self.output_index = output_index  # Index of the output tensor to use for Grad-CAM

    def forward(self, x):
        outputs = self.model(x)
        # Assuming outputs is a list of tensors, return the one specified by output_index
        return outputs[self.output_index]

# Function to denormalize image for visualization purposes
def denormalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    return tensor
    
# Instantiate your custom model just like you did before saving
image_encoder, image_encoder_output_dim = get_image_encoder(pretrained=True)
attribute_sizes = [6]  # Example sizes, replace with actual sizes of your tasks

model = AttributePredictor(attribute_sizes, image_encoder_output_dim, image_encoder)

# Load the entire checkpoint, not just the model state_dict
checkpoint = torch.load('./log/best_model_nucleus_crop.pth')

# Now, load the state_dict into your model
model.load_state_dict(checkpoint['model'])
model.eval()

# Assuming you're using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare your input image transformations without ToTensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Assuming your transform needs resizing, this would need to be adjusted
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
img_path = 'cropped_image.jpg'

original_img = read_image(img_path).float()


# print(original_img.min(), original_img.max())  # Should be in the range 0-255 for a typical image
original_img /= 255.0  # Already scaled to [0, 1] then normalized

# Resize the image
img = transform(original_img).unsqueeze(0).to(device)  # Normalize and add batch dimension

# Initialize Grad-CAM and Grad-CAM++
target_layer = model.image_encoder.layer4[-1]  # Directly pass the layer, not wrapped in a list
# Wrap your model specifying which output to use for Grad-CAM (e.g., index 0)
gradcam_model_wrapper = GradCAMWrapper(model, output_index=0)

# Then use this wrapper model with Grad-CAM
gradcam = GradCAM(gradcam_model_wrapper, target_layer)
gradcam_pp = GradCAMpp(gradcam_model_wrapper, target_layer)

# Generate the heatmap
mask, _ = gradcam(img)
heatmap, result = visualize_cam(mask, img)

# We first check if it's not a single channel, we take the first channel for visualization
if heatmap.ndim == 3 and heatmap.shape[0] == 3:
    heatmap = heatmap[0]  # Taking only the first channel

# Apply denormalization to the original image tensor
img_denorm = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_denorm = torch.clamp(img_denorm, 0, 1)  # Ensure values are within [0, 1]

# Convert to PIL Image for display
img_pil = to_pil_image(img_denorm.squeeze())  # Remove batch dimension and convert to PIL for visualization

# Visualization
fig, axarr = plt.subplots(1, 3, figsize=(12, 4))

# Original Image
axarr[0].imshow(img_pil)
axarr[0].title.set_text('Original Image')

# Grad-CAM
axarr[1].imshow(heatmap.squeeze().cpu(), cmap='jet')
axarr[1].title.set_text('Grad-CAM')

# Result on Image
axarr[2].imshow(img_pil)
axarr[2].imshow(heatmap.squeeze().cpu(), cmap='jet', alpha=0.4)
axarr[2].title.set_text('Result on cropped nucleus image')

for ax in axarr:
    ax.axis('off')
plt.tight_layout()
plt.show()

