import tensorflow as tf
import numpy as np

from layers import ssd300
from batch_gen import Batch_Generator
from utils import SSDBoxEncoder
from loss import compute_loss

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

n_boxes = [len(box) for box in aspect_ratios]

normalize_coords = True


train = Batch_Generator()

model = ssd300.ssd((image_height, image_width, image_channels), n_classes, n_boxes)

weights = 'vgg-16_ssd-fcn_ILSVRC-CLS-LOC.h5'
model.load_weights(weights, by_name = True)



adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

model.compile(optimizer = adam, loss = compute_loss)

VOC_annotations_dir = "VOC2012/Annotations"
VOC_images_dir = "VOC2012/JPEGImages"
VOC_image_sets_train = "VOC2012/ImageSets/Main/train.txt"
VOC_image_sets_val = "VOC2012/ImageSets/Main/val.txt"
VOC_image_sets_trainval = "VOC2012/ImageSets/Main/trainval.txt"

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

train.parse_xml(VOC_annotations_dir, VOC_images_dir, VOC_image_sets_train, classes)

predictor_sizes = [model.get_layer('conv4_3_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv11_2_mbox_conf').output_shape[1:3]]

encoder = SSDBoxEncoder(image_height, image_width, n_classes, predictor_sizes, scales_voc, aspect_ratios_per_layer=aspect_ratios, 
        normalize_coords = True)

n_train_samples = train.total_samples()

batch_size = 32

train_generator =  train.generate(ssd_box_encoder=encoder, train=True)


epochs = 10

history = model.fit_generator(generator = train_generator,
                             steps_per_epoch = np.ceil(n_train_samples/batch_size),
                              epochs = epochs)

model_name = 'ssd300'
model.save('{}.h5'.format(model_name))
model.save_weights('{}_weights.h5'.format(model_name))