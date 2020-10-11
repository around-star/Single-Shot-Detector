import tensorflow as tf

def vgg16(img_shape) :
    inputs = tf.keras.layers.Input(shape=img_shape, name='input_1')

    conv1_1 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv1_1')(inputs)
    conv1_2 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv1_2')(conv1_1)

    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name = 'pool1')(conv1_2)

    conv2_1 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv2_1')(pool1)
    conv2_2 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv2_2')(conv2_1)

    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name = 'pool2')(conv2_2)

    conv3_1 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv3_1')(pool2)
    conv3_2 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv3_2')(conv3_1)
    conv3_3 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv3_3')(conv3_2)

    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name = 'pool3')(conv3_3)

    conv4_1 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv4_1')(pool3)
    conv4_2 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv4_2')(conv4_1)
    conv4_3 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv4_3')(conv4_2)

    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name = 'pool4')(conv4_3)

    conv5_1 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv5_1')(pool4)
    conv5_2 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv5_2')(conv5_1)
    conv5_3 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'conv5_3')(conv5_2)

    pool5 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same', name = 'pool5')(conv5_3)

    conv6 = tf.keras.layers.Conv2D(1024, kernel_size=(3,3), dilation_rate=(6,6), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'fc6')(pool5)

    conv7 = tf.keras.layers.Conv2D(1024, kernel_size=(1,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'fc7')(conv6)

    return inputs, conv4_3, conv7
    