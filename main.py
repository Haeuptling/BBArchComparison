from model import load_weight_model,predict_image,get_image_as_array, show_image
from config import NUM_CLASSES_ALL,BBOX_PATH,MAIN_BBOX_DETECTOR_MODEL,SUB_BBOX_DETECTOR_MODEL,sub_class_ids, main_class_ids

bbox_model = load_weight_model("workspace/models/sub_bbox_detector_model.h5", len(sub_class_ids))
#boxes,confidence,classes  = predict_image('workspace/images/Bilder/Seite1.jpeg',bbox_model)
#show_image(('workspace/images/Bilder/Seite1.jpeg'), boxes,confidence,classes )


boxes,confidence,classes = predict_image('workspace/images/640x640/000000000003.jpeg',bbox_model)
show_image(('workspace/images/640x640/000000000003.jpeg'), boxes,confidence,classes)

