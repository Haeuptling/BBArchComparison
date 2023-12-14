import cv2
import numpy as np
import os

def get_image_size(image):
    height= image.shape[0]
    width = image.shape[1]
    return width, height

#TODO return image for prediction
def resize_image(input_path, output_path, height, width):
    image =  cv2.imread(input_path)
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite(output_path, resized_image)    

#TODO scale image for 
def resize_bounding_box_and_area(input_path,bounding_boxes, height, width):
    image =  cv2.imread(input_path)
    height_ratio = height / image.shape[0]
    width_ratio = width / image.shape[1]
    
    for box in bounding_boxes:
        resized_boxes = []
        resized_area = 0
        x = np.round(box[0]*width_ratio,2)
        y = np.round(box[1]*height_ratio,2)
        x_width = np.round(box[2]*width_ratio,2)
        y_height = np.round(box[3]*height_ratio,2)
        resized_boxes.append([x, y, x_width, y_height])
        resized_area = x_width * y_height

    return x, y, x_width, y_height, resized_area

def create_directories(output_path_images, output_path_annotations, new_width, new_height):
    output_path_images = output_path_images + '/' + str(new_height) + 'x' + str(new_width)
    output_path_annotations = output_path_annotations + '/' + str(new_height) + 'x' + str(new_width)
    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_annotations, exist_ok=True)
