import torch
from torchvision import models, transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Custom imports
from attribute_predictor import AttributePredictor  
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam

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
checkpoint = torch.load('./log/best_model_nucleus.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Image Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = './data/PBC/PBC_dataset_normal_DIB/neutrophil/SNE_70096.jpg'
original_img = read_image(img_path).float() / 255.0
img = transform(original_img).unsqueeze(0).to(device)

# Prediction and Probability Calculation
with torch.no_grad():
    predictions = model(img)

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
img_pil = to_pil_image(img_denorm.squeeze())

fig, axarr = plt.subplots(1, 3, figsize=(12, 4))
axarr[0].imshow(img_pil)
axarr[0].title.set_text('Original Image')
axarr[1].imshow(heatmap.squeeze().cpu(), cmap='jet')
axarr[1].title.set_text('Grad-CAM')
axarr[2].imshow(img_pil)
axarr[2].imshow(heatmap.squeeze().cpu(), cmap='jet', alpha=0.4)
axarr[2].title.set_text('Result on SE blocks implementation')
for ax in axarr:
    ax.axis('off')
plt.tight_layout()
plt.show()
