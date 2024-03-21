import os
import random
from PIL import Image
import cv2
import numpy as np

def image_preprocessing(image_path, target_size=(256, 256)):

    # image = cv2.imread(image_path)
    image = Image.open(image_path)
    assert image is not None, "Image not found"

    #convert image to opencv format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to HSV color space and split channels
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Split RGB channels
    b, g, r = cv2.split(image)

    # Subtract the S channel with the G channel
    subtracted_image = cv2.subtract(s, g)
    cv2.imshow('Subtracted Image', subtracted_image)
    
    # convert to BGR format
    subtracted_image = cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR)

    
    # convert to PIL format
    subtracted_image = Image.fromarray(subtracted_image)
    # show the subtracted image
    subtracted_image.show()

    # # Threshold the subtracted image
    # _, thresh = cv2.threshold(subtracted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Find contours
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Assuming the largest contour is the nucleus, if not empty
    # if contours:
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(largest_contour)
    #     # Crop the image to the bounding rectangle of the largest contour
    #     cropped_nucleus = image[y:y+h, x:x+w]
    #     cv2.imshow('Cropped Nucleus', cropped_nucleus)
    # else:
    #     raise ValueError("No contours found in the image.")
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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