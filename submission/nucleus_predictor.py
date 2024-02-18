import os
import random
from PIL import Image
import cv2
import numpy as np

def image_preprocessing(image_path, target_size=(256, 256)):
    
    image = cv2.imread(image_path)
    cv2.imshow('image', image)
    image_2 = image.copy()
    image = cv2.resize(image, target_size) 
    image_2 = cv2.resize(image_2, target_size)
    
    # Get green channel from image
    _, image_G, _ = cv2.split(image)
    cv2.imshow('image_G', image_G)
    
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2HSV)
    _, _, image_S = cv2.split(image_2)
    cv2.imshow('image_S', image_S)
    
        # Perform Otsu's thresholding on the G channel
    # _, binary_G = cv2.adaptiveThreshold(image_G, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # _, binary_S = cv2.adaptiveThreshold(image_S, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    
    # cv2.imshow('binary_G', binary_G)
    # cv2.imshow('binary_S', binary_S)
    
    nucleus_segmented = cv2.subtract(image_S, image_G)
    cv2.imshow('nucleus_segmented', nucleus_segmented)
    # # Adjust gray levels
    # # Threshold the grayscale image
    _, binary_mask = cv2.threshold(nucleus_segmented, 90, 255, cv2.THRESH_BINARY)
    

    
    # Apply the binary mask to the original image
    adjusted_image = cv2.bitwise_and(nucleus_segmented, nucleus_segmented, mask=binary_mask)
    cv2.imshow('adjusted_image', adjusted_image)
    
    # # Apply morphological operations
    # kernel = np.ones((5,5), np.uint8)
    # adjusted_image2 = cv2.dilate(adjusted_image, kernel, iterations=1)
    # adjusted_image2 = cv2.erode(adjusted_image2, kernel, iterations=2)

    # cv2.imshow('morp image', adjusted_image2)
    
    # Crop out the nucleus area
    cropped_nucleus = crop_nucleus(adjusted_image, binary_mask)
    cv2.imshow('cropped_nucleus', cropped_nucleus)
    



    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return 0

def crop_nucleus(image, binary_mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the original image using the bounding box coordinates
    cropped_nucleus = image[y:y+h, x:x+w]
    
    return cropped_nucleus

def random_image_from_subfolder(root_folder):

    # List all subofolders in root folder
    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder,f))]
    if not subfolders:
        raise ValueError("No subfolders found in root folder")
        
    # Choose a random subfolder
    random_subfolder = random.choice(subfolders)
    subfolder_path = os.path.join(root_folder, random_subfolder)
    
    # List all iamges in the choosen subfolder
    images = os.listdir(subfolder_path)
    if not images:
        raise ValueError("No images found in subfolder")
    
    # Choose a random image from the subfolder
    selected_image = random.choice(images)
    # Construct full path to image
    image_path = os.path.join(subfolder_path, selected_image)
    
    return image_path

root_folder =  "C:\\Users\\tanqi\\Documents\\FYP\\Final Year Project\\submission\\data\\PBC\\PBC_dataset_normal_DIB"
selected_image_path = random_image_from_subfolder(root_folder)
image = image_preprocessing(selected_image_path)
image.show()


    



