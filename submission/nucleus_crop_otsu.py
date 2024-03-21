import os
import random
from PIL import Image
import cv2
import numpy as np

def image_preprocessing(image_path, target_size=(256, 256)):

    image = cv2.imread(image_path)
    assert image is not None, "Image not found"

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to create a binary mask
    # Otsu's method automatically finds the optimal threshold value
    _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Morphological operations to clean up the mask
    # This kernel size and iterations may need to be adjusted
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours on the cleaned mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and get the largest one, which should be the nucleus
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    nucleus_contour = sorted_contours[0]  # The largest contour
    x, y, w, h = cv2.boundingRect(nucleus_contour)

    # Crop the image to the bounding rectangle of the largest contour
    cropped_nucleus = image[y:y+h, x:x+w]

    # Save the cropped nucleus image

    cv2.imshow('Cropped Nucleus', cropped_nucleus)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
# selected_image_path = random_image_from_subfolder(root_folder)
# image = image_preprocessing(selected_image_path)

while True:  # Start of the loop
    selected_image_path = random_image_from_subfolder(root_folder)
    image_preprocessing(selected_image_path)

    # Wait for a key press
    key = cv2.waitKey(0)

    # If the 'n' key is pressed, continue to the next image
    if key == ord('n'):
        continue
    # If the 'q' key is pressed, exit the loop
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
