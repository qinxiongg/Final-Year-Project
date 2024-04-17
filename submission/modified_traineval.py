import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from att_dataset import AttDataset
from attribute_predictor import AttributePredictor
from torchvision import models, transforms


def load_model(model_path, device):
    # Load the model structure from attribute_predictor
    # Assumes the saved model includes both structure and weights
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def test_model(model, data_loader, device):
    all_preds, all_labels = [], []
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

def main(model_path, test_csv, image_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = AttDataset(test_csv, attribute_columns, image_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    accuracy = test_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    model_path = "path_to_saved_model.pth"
    test_csv = "path_to_test_csv.csv"
    image_dir = "path_to_images/"
    main(model_path, test_csv, image_dir)
