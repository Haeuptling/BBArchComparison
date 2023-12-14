import ressize
import keras_cv
import tensorflow as tf
import numpy as np
from keras.preprocessing import image as keras_image

from config import LEARNING_RATE,GLOBAL_CLIPNORM,NUM_CLASSES_ALL,SUB_BBOX_DETECTOR_MODEL,BBOX_PATH,MAIN_BBOX_DETECTOR_MODEL, class_ids, main_class_ids,sub_class_ids

def predict_image(image_path, model):
    image = get_image_as_array(image_path)
    predictions = model.predict(image)
    print(predictions)
    best_bboxes = extract_boxes(predictions)
    predicted_class_ids = list(best_bboxes.keys())
    predicted_bounding_boxes = list(best_bboxes.values())
    predicted_ids_names = []
    
    #TODO in universell aendern
    for id in predicted_class_ids:
        predicted_ids_names.append(get_class_mapping(MAIN_BBOX_DETECTOR_MODEL)[id])

    print("predicted_class_ids:", predicted_ids_names)
    print("predicted_bounding_boxes:", predicted_bounding_boxes)    


def define_model(num_classes):
    model = keras_cv.models.YOLOV8Detector(
    num_classes=num_classes, 
    bounding_box_format="xyxy",
    backbone=define_backbone("yolo_v8_xs_backbone_coco"),
    fpn_depth=1,
    )
    return model

def compile_model(model):
    model.compile(
    optimizer=define_optimizer(), 
    classification_loss="binary_crossentropy", 
    box_loss="ciou"
    )

def load_weight_model(model_path):
    base_model = define_model(2)#len(get_class_mapping(model_path)[0]
    compile_model(base_model)
    base_model.load_weights(model_path)
    return  base_model


def define_optimizer():
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM, 
    )
    return optimizer

def define_backbone(backbone):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone_coco",
         load_weights=True 
    )
    return backbone

def get_class_mapping(model_path):
    if MAIN_BBOX_DETECTOR_MODEL in model_path:
        main_class_mapping = dict(zip(range(len(main_class_ids)), main_class_ids))
        return main_class_mapping
    if SUB_BBOX_DETECTOR_MODEL in model_path:
        sub_class_mapping = dict(zip(range(len(sub_class_ids)), sub_class_ids))
        return sub_class_mapping
    else:
        class_mapping = dict(zip(range(len(class_ids)), class_ids))
        return class_mapping

def extract_boxes(predictions_on_image):
    best_bboxes = {}
    class_id = []
    bbox = []
    confidence = []

    for i in range(0, predictions_on_image['num_detections'][0]):
        class_id.append(predictions_on_image['classes'][0][i])
        bbox.append(predictions_on_image['boxes'][0][i])
        confidence.append(predictions_on_image['confidence'][0][i])

    for i in range(len(class_id)):
        current_class = class_id[i]
        current_confidence = confidence[i]
        current_box = bbox[i]

        if current_class in best_bboxes and np.all(current_box > best_bboxes[current_class]):
            best_bboxes[current_class] = current_box
        elif current_class not in best_bboxes:
            best_bboxes[current_class] = current_box
    return best_bboxes

#TODO get real height width 
def get_image_as_array(image_path):
    img = keras_image.load_img(image_path, target_size=(640, 640))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    return img_array