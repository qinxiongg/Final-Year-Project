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
        image = self.nucleus_crop(image)  # Apply nucleus preprocessing
        attributes = [self.attribute_encoders[col][self.df.loc[idx, col]]
                      for col in self.attribute_columns]
        sample = {'image': image, 'attributes': torch.tensor(
            attributes, dtype=torch.long), 'img_path': image_path}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def nucleus_grayscale(self, image):
        
        # convert PIL image to opencv format if not already
        # if not isinstance(image, np.ndarray):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_copy = image.copy()
        
        # Split original image into BGR channels
        B, G, R = cv2.split(image)
        
        # Convert to image_copy to HSV and split channels
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(image_copy)

        # Subtract the S channel with the G channel
        processed_image = cv2.subtract(S, G)
        
        # convert to from greyscale to BGR format
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        
        # Convert processed_image back to PIL Image
        processed_image = Image.fromarray(processed_image)
        
        return processed_image
    
    def nucleus_crop(self, image, min_size=(150, 150)):

        #convert PIL image to opencv format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
        # Convert to image to HSV color space and split the channels
        HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV_image)

        # Split BGR channels 
        B, G, R = cv2.split(image)

        # Subtract the S channel with the G channel
        subtracted_image = cv2.subtract(S, G)
        
        # Threshold the subtracted image
        ret, thresholded_image = cv2.threshold(subtracted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate the thresholded image improve contour detection
        kernel = np.ones((5,5),np.uint8)
        dilated_threshold_image = cv2.dilate(thresholded_image, kernel, iterations = 1)

        # Find contours
        contours, hierarchy = cv2.findContours(dilated_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming the largest contour is the nucleus, if not empty
        if contours:
    
            largest_contour = max(contours, key=cv2.contourArea)
            original_x, original_y, original_w, original_h = cv2.boundingRect(largest_contour)
            
            # Calculate the center of the original bounding box
            original_center_x = original_x + original_w // 2
            original_center_y = original_y + original_h // 2

            # Enforce minimum size, ensuring it's centered around the original bounding box
            w = max(original_w, min_size[0])
            h = max(original_h, min_size[1])

            # Adjust x and y to crop the image around the center of the bounding box
            new_x = max(original_center_x - w // 2, 0)
            new_y = max(original_center_y - h // 2, 0)

            # Adjust the end points, making sure we don't go out of the image boundaries
            new_x_end = min(new_x + w, image.shape[1])
            new_y_end = min(new_y + h, image.shape[0])

            # Correct the coordinates if they go out of bounds
            if new_x_end > image.shape[1]:
                new_x = image.shape[1] - w
            if new_y_end > image.shape[0]:
                new_y = image.shape[0] - h

            # Crop the image with the adjusted coordinates
            cropped_nucleus = image[new_y:new_y_end, new_x:new_x_end]
        
            # convert to back to RGB format for conversion back to PIL
            cropped_nucleus = cv2.cvtColor(cropped_nucleus, cv2.COLOR_BGR2RGB)
            # convert to PIL format
            processed_image = Image.fromarray(cropped_nucleus)
        
        return processed_image
    
    def nucleus_segmentation(self, image):

        #convert PIL image to opencv format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Convert to HSV color space and split channels
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # Split RGB channels
        b, g, r = cv2.split(image)

        # Subtract the S channel with the G channel
        subtracted_image = cv2.subtract(s, g)
        
        # Threshold the subtracted image
        _, thresh = cv2.threshold(subtracted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate the thresholded image 
        kernel = np.ones((5,5),np.uint8)
        dilated_thresh = cv2.dilate(thresh, kernel, iterations = 1)

        # Convert the binary threshold image to 3 channels
        thresh_3_channel = cv2.merge([dilated_thresh, dilated_thresh, dilated_thresh])

        # Element-wise multiplication of the binary threshold with the original image
        segmented_image = cv2.multiply(image, thresh_3_channel, scale=1/255)

        # convert to BGR format
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        # convert to PIL format
        segmented_image = Image.fromarray(segmented_image)

        return segmented_image