import tensorflow as tf
import numpy as np
import cv2
from batch_gen import Batch_Generator

from utils import SSDBoxEncoder
from layers import ssd300

image_height = 300
image_width = 300
image_channels = 3
n_classes = 21
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]

aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]

normalize_coords = True

n_boxes = [len(box) for box in aspect_ratios]

model = ssd300.ssd((image_height, image_width, image_channels), n_classes, n_boxes)
model.load_weights('ssd300_weights.h5')

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

predictor_sizes = [model.get_layer('conv4_3_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv11_2_mbox_conf').output_shape[1:3]]

encoder = SSDBoxEncoder(image_height, image_width, n_classes, predictor_sizes, scales_voc,
                        aspect_ratios_per_layer=aspect_ratios, normalize_coords = True)

VOC_annotations_dir = "VOC2012/Annotations"
VOC_images_dir = "VOC2012/JPEGImages"
VOC_image_sets_val = "VOC2012/ImageSets/Main/val.txt"


val = Batch_Generator()
val.parse_xml(VOC_annotations_dir, VOC_images_dir, VOC_image_sets_val, classes)
predict_generator = val.generate(batch_size = 1, ssd_box_encoder = encoder)

X = next(predict_generator)
X = np.array(X)
X = np.reshape(X, (1, image_height, image_width, image_channels))
y_pred = model.predict(X)

y_pred_decoded = encoder.decode_y(y_pred, confidence_thresh = 0.5, iou_threshold = 0.4, top_k = 200, normalize_coords = True,
                                 img_height = image_height, img_width = image_width)

X = np.reshape(X, (image_height, image_width, image_channels)) * 255
for boxes in y_pred_decoded[0]:
    xmin = boxes[-4]
    ymin = boxes[-3]
    xmax = boxes[-2]
    ymax = boxes[-1]
    label = classes[int (boxes[0])]
    print(label)
    cv2.rectangle(X, (xmin, ymin), (xmax, ymax),(20, 220, 50))
    cv2.putText(X, label, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

cv2.imwrite('result.jpg', X)

