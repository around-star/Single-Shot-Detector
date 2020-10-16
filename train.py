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



optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)



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

VOC_image_sets_dir = [VOC_image_sets_train, VOC_image_sets_val, VOC_image_sets_trainval]

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

batch_size = 10

train_generator =  train.generate(batch_size = batch_size, ssd_box_encoder=encoder,  train=True)
model_name = 'ssd300'


epochs = 1

for epoch in range(epochs):
    for steps in range(int (np.ceil(n_train_samples/batch_size))):
        data = next(train_generator)
        x_batch = data[0]
        y_true = data[1]
        
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = compute_loss(y_true, y_pred)
            print('Epoch : {} , Step : {} , Loss : {} '.format(epoch, steps, loss))

        grads = tape.gradient(loss, model.variables)

        optimizer.apply_gradients(zip(grads, model.variables))

        if steps % 25 == 0:

            model.save_weights('weights/{}_step_{}_weights.h5'.format(model_name, steps))
