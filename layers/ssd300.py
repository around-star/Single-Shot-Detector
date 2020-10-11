import tensorflow as tf
from layers.base_layer import vgg16

def ssd(img_shape, n_classes, n_boxes):
    inputs, conv4_3, conv7 = vgg16(img_shape)
    conv8_1 = tf.keras.layers.Conv2D(256, kernel_size=(1,1),  padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv7)
    conv8_1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(conv8_1)
    conv8_2 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), strides=(2, 2) , padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv8_1)

    conv9_1 = tf.keras.layers.Conv2D(128, kernel_size=(1,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv8_2)
    conv9_1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(conv9_1)
    conv9_2 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(2, 2) , padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv9_1)

    conv10_1 = tf.keras.layers.Conv2D(128, kernel_size=(1,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv9_2)
    conv10_2 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1, 1) , padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv10_1)

    conv11_1 = tf.keras.layers.Conv2D(128, kernel_size=(1,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv10_2)
    conv11_2 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1, 1) , padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv11_1)
    
    #conv4_3 = tf.keras.layers.LayerNormalization()

    # Predict n classes' confidence values for each cell and each box at each layer
    # Output Shape at each layer : (batch, height, width, n_boxes * n_classes)

    conv4_3_mbox_conf = tf.keras.layers.Conv2D(n_boxes[0] * n_classes, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv4_3_mbox_conf')(conv4_3)
    conv7_mbox_conf = tf.keras.layers.Conv2D(n_boxes[1] * n_classes, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv7_mbox_conf')(conv7)
    conv8_2_mbox_conf = tf.keras.layers.Conv2D(n_boxes[2] * n_classes, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = tf.keras.layers.Conv2D(n_boxes[3] * n_classes, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv9_2_mbox_conf')(conv9_2)
    conv10_2_mbox_conf = tf.keras.layers.Conv2D(n_boxes[4] * n_classes, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv10_2_mbox_conf')(conv10_2)
    conv11_2_mbox_conf = tf.keras.layers.Conv2D(n_boxes[5] * n_classes, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv11_2_mbox_conf')(conv11_2)


    # Predict 4 coordinate points for each cell and each box at each layer
    # Output shape at each layer : (batch, height, width, 4 * n_boxes)

    conv4_3_mbox_loc = tf.keras.layers.Conv2D(n_boxes[0] * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv4_3)
    conv7_mbox_loc = tf.keras.layers.Conv2D(n_boxes[1] * 4, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv7)
    conv8_2_mbox_loc = tf.keras.layers.Conv2D(n_boxes[2] * 4, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv8_2)
    conv9_2_mbox_loc = tf.keras.layers.Conv2D(n_boxes[3] * 4, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv9_2)
    conv10_2_mbox_loc = tf.keras.layers.Conv2D(n_boxes[4] * 4, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv10_2)
    conv11_2_mbox_loc = tf.keras.layers.Conv2D(n_boxes[5] * 4, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv11_2)



    # Reshape class predictions to shape (batch, height*width*n_boxes, n_classes)
    conv4_3_mbox_conf_reshape = tf.keras.layers.Reshape((-1, n_classes))(conv4_3_mbox_conf)
    conv7_mbox_conf_reshape = tf.keras.layers.Reshape((-1, n_classes))(conv7_mbox_conf)
    conv8_2_mbox_conf_reshape = tf.keras.layers.Reshape((-1, n_classes))(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = tf.keras.layers.Reshape((-1, n_classes))(conv9_2_mbox_conf)
    conv10_2mbox_conf_reshape = tf.keras.layers.Reshape((-1, n_classes))(conv10_2_mbox_conf)
    conv11_2_mbox_conf_reshape = tf.keras.layers.Reshape((-1, n_classes))(conv11_2_mbox_conf)

    # Reshape coordinate prediction to shape (batch, height*width*n_boxes, 4)
    conv4_3_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4))(conv4_3_mbox_loc)
    conv7_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4))(conv7_mbox_loc)
    conv8_2_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4))(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4))(conv9_2_mbox_loc)
    conv10_2mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4))(conv10_2_mbox_loc)
    conv11_2_mbox_loc_reshape = tf.keras.layers.Reshape((-1, 4))(conv11_2_mbox_loc)

    # Concatenate along the "total number of boxes axis".
    mbox_conf = tf.keras.layers.Concatenate(axis=1)([conv4_3_mbox_conf_reshape, 
                                                    conv7_mbox_conf_reshape, 
                                                    conv8_2_mbox_conf_reshape,
                                                    conv9_2_mbox_conf_reshape,
                                                    conv10_2mbox_conf_reshape,
                                                    conv11_2_mbox_conf_reshape])

    mbox_loc = tf.keras.layers.Concatenate(axis=1)([conv4_3_mbox_loc_reshape,
                                                    conv7_mbox_loc_reshape,
                                                    conv8_2_mbox_loc_reshape,
                                                    conv9_2_mbox_loc_reshape,
                                                    conv10_2mbox_loc_reshape,
                                                    conv11_2_mbox_loc_reshape])

    mbox_conf_softmax = tf.keras.layers.Softmax()(mbox_conf)

    predictions = tf.keras.layers.Concatenate(axis=2)([mbox_conf_softmax, mbox_loc])

    model = tf.keras.models.Model(inputs = inputs, outputs = predictions)

    return model