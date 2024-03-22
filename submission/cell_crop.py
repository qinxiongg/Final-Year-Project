import os
import random
from PIL import Image
import cv2
import numpy as np

def HSV_preprocessing(image_path, target_size=(256, 256)):

    # image = cv2.imread(image_path)
    image = Image.open(image_path)
    assert image is not None, "Image not found"

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
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_preprocessing(image_path, crop_size=(150, 150)):

    # image = cv2.imread(image_path)
    image = Image.open(image_path)
    # image.show()

    #convert PIL image to opencv format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Original Image', image)
    
    # Convert to HSV color space and split channels
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Split RGB channels
    b, g, r = cv2.split(image)

    # Subtract the S channel with the G channel
    subtracted_image = cv2.subtract(s, g)
    # cv2.imshow('Subtracted Image', subtracted_image)
    
    # Threshold the subtracted image
    _, thresh = cv2.threshold(subtracted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('Threshold Image', thresh)   
    
    # Dilate the thresholded image 
    kernel = np.ones((5,5),np.uint8)
    dilated_thresh = cv2.dilate(thresh, kernel, iterations = 1)
    cv2.imshow('Dilated Image', dilated_thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the nucleus, if not empty
    if contours:
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # # Draw the bounding box on the original image
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('Image with Bounding Box', image)

        # # Enforce minimum size
        # w = max(w, min_size[0])
        # h = max(h, min_size[1])

        # # Crop with the adjusted size, centering the crop around the original bounding box as much as possible
        # center_x, center_y = x + w // 2, y + h // 2
        # x = max(center_x - w // 2, 0)
        # y = max(center_y - h // 2, 0)
        # x_end = min(x + w, image.shape[1])
        # y_end = min(y + h, image.shape[0])

        # cropped_nucleus = image[y:y_end, x:x_end]
        # cv2.imshow('Cropped Nucleus', cropped_nucleus)
        
        # Calculate the center of the bounding box
        center_x, center_y = x + w // 2, y + h // 2

        # Define the new crop dimensions
        new_w, new_h = crop_size

        # Calculate the top left corner of the new crop area
        new_x = max(center_x - new_w // 2, 0)
        new_y = max(center_y - new_h // 2, 0)

        # Make sure the crop area does not go out of the image
        new_x_end = min(new_x + new_w, image.shape[1])
        new_y_end = min(new_y + new_h, image.shape[0])

        # Adjust the starting point if necessary to fit the expected size
        if new_x_end - new_x < new_w:
            new_x = new_x_end - new_w
        if new_y_end - new_y < new_h:
            new_y = new_y_end - new_h

        # Crop the image
        cropped_nucleus = image[new_y:new_y_end, new_x:new_x_end]
        
        # Show the cropped nucleus
        cv2.imshow('Cropped Nucleus', cropped_nucleus)
       
        # convert to BGR format
        cropped_nucleus = cv2.cvtColor(cropped_nucleus, cv2.COLOR_BGR2RGB)
        # convert to PIL format
        cropped_nucleus = Image.fromarray(cropped_nucleus)
    
        # show the subtracted image
        cropped_nucleus.show()

    else:
        raise ValueError("No contours found in the image.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def nucleus_segmentation(image_path, crop_size=(150, 150)):

    # image = cv2.imread(image_path)
    image = Image.open(image_path)
    # image.show()

    #convert PIL image to opencv format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Original Image', image)
    
    # Convert to HSV color space and split channels
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Split RGB channels
    b, g, r = cv2.split(image)

    # Subtract the S channel with the G channel
    subtracted_image = cv2.subtract(s, g)
    # cv2.imshow('Subtracted Image', subtracted_image)
    
    # Threshold the subtracted image
    _, thresh = cv2.threshold(subtracted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('Threshold Image', thresh)   
    
    # Dilate the thresholded image 
    kernel = np.ones((5,5),np.uint8)
    dilated_thresh = cv2.dilate(thresh, kernel, iterations = 1)
    cv2.imshow('Dilated Image', dilated_thresh)

    # Convert the binary threshold image to 3 channels
    thresh_3_channel = cv2.merge([dilated_thresh, dilated_thresh, dilated_thresh])

    # Find contours
    contours, hierarchy = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Element-wise multiplication of the binary threshold with the original image
    segmented = cv2.multiply(image, thresh_3_channel, scale=1/255)
    cv2.imshow('Segmented Image', segmented)

    # convert to BGR format
    segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    # convert to PIL format
    segmented = Image.fromarray(segmented)

    # show the subtracted image
    segmented.show()

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


root_folder =  "./data/PBC/PBC_dataset_normal_DIB"
while True:
    selected_image_path = random_image_from_subfolder(root_folder)
    image = nucleus_segmentation(selected_image_path)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
       # Close the current image window before the next loop iteration