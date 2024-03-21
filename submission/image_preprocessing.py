from torchvision import transforms
import cv2
import numpy as np

class nucleus_segmentation():
    def __init__(self):
        super().__init__()
    
    def __call__(self, sample):
        
        sample = np.array(sample)
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        sample_copy = sample.copy()
        
        # split orginal image into BGR channels
        B, G, R = cv2.split(sample)
        
        # Convert to image_copy to HSV and split channels
        sample_copy = cv2.cvtColor(sample_copy, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(sample_copy)

        # Subtract the S channel with the G channel
        sample = cv2.subtract(S, G)
        
        return sample 