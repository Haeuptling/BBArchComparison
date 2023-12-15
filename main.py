from model import load_weight_model,predict_image,get_image_as_array, show_image
from config import NUM_CLASSES_ALL,BBOX_PATH,MAIN_BBOX_DETECTOR_MODEL,SUB_BBOX_DETECTOR_MODEL

bbox_model = load_weight_model(BBOX_PATH+SUB_BBOX_DETECTOR_MODEL)
#boxes,confidence,classes  = predict_image('workspace/images/Bilder/Seite1.jpeg',bbox_model)
#show_image(('workspace/images/Bilder/Seite1.jpeg'), boxes,confidence,classes )


boxes,confidence,classes = predict_image('workspace/images/640x640/000000000003.jpeg',bbox_model)
show_image(('workspace/images/640x640/000000000003.jpeg'), boxes,confidence,classes)

