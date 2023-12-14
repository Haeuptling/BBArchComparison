from model import load_weight_model,predict_image,get_image_as_array, show_image
from config import NUM_CLASSES_ALL,BBOX_PATH,MAIN_BBOX_DETECTOR_MODEL

bbox_model = load_weight_model(BBOX_PATH+MAIN_BBOX_DETECTOR_MODEL)
predicted_bounding_boxes,predicted_ids_names = predict_image('workspace/images/Bilder/Seite1.jpeg',bbox_model)
show_image(('workspace/images/Bilder/Seite1.jpeg'), predicted_bounding_boxes)


#predicted_bounding_boxes,predicted_ids_names = predict_image('workspace/images/640x640/000000000003.jpeg',bbox_model)
#show_image(('workspace/images/640x640/000000000003.jpeg'), predicted_bounding_boxes)