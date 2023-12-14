from model import load_weight_model,predict_image,get_image_as_array
from config import NUM_CLASSES_ALL,BBOX_PATH,MAIN_BBOX_DETECTOR_MODEL

bbox_model = load_weight_model(BBOX_PATH+MAIN_BBOX_DETECTOR_MODEL)
predict_image('workspace/images/640x640/000000000001.jpeg',bbox_model)
