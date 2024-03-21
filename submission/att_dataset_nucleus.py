import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import numpy as np
import cv2
import PIL.Image as Image


class AttDataset(Dataset):
    def __init__(self, csv_file, attribute_columns, image_dir="", transform=None, multiply=1, attribute_encoders=None, loader=default_loader):
        if isinstance(csv_file, pd.DataFrame):
            self.df = csv_file
        else:
            self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.multiply = multiply
        self.attribute_columns = attribute_columns
        self.attribute_encoders = attribute_encoders or self.create_attribute_encoders()
        self.attribute_decoders = self.create_attribute_decoders(
            self.attribute_encoders)
        self.loader = loader

    def create_attribute_encoders(self):
        attribute_encoders = {}
        for col in self.attribute_columns:
            attribute_encoders[col] = {
                value: idx for idx, value in enumerate(sorted(self.df[col].unique()))}
        return attribute_encoders

    def create_attribute_decoders(self, attribute_encoders):
        # Create reverse mapping for each dictionary
        attribute_decoders = {}
        for col, encoding in attribute_encoders.items():
            attribute_decoders[col] = {v: k for k, v in encoding.items()}
        return attribute_decoders

    def __len__(self):
        return len(self.df) * self.multiply

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = os.path.join(self.image_dir, self.df.loc[idx, 'path'])
        image = self.loader(image_path)
        image = self.nucleus_segmentation(image)  # Apply nucleus segmentation
        attributes = [self.attribute_encoders[col][self.df.loc[idx, col]]
                      for col in self.attribute_columns]
        sample = {'image': image, 'attributes': torch.tensor(
            attributes, dtype=torch.long), 'img_path': image_path}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def nucleus_segmentation(self, image):
        # Assuming 'image' is a NumPy array. If it's a PIL Image, convert it to NumPy array first.
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_copy = image.copy()
        
        # Split original image into BGR channels (B, G, R not used afterwards)
        B, G, R = cv2.split(image)
        
        # Convert to image_copy to HSV and split channels
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(image_copy)

        # Subtract the S channel with the G channel
        processed_image = cv2.subtract(S, G)
        
        # convert to from greyscale to BGR format
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        
        # Convert processed_image back to PIL Image to maintain compatibility with torchvision transforms
        processed_image = Image.fromarray(processed_image)
        
        return processed_image