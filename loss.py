import tensorflow as tf
import numpy as np

def l1_loss(y_true, y_pred):
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


def log_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 1e-15) # Prevent the value of y_pred from being zero

    log_loss = - tf.reduce_sum(tf.math.log(y_pred) * y_true, axis = -1)

    return log_loss


def compute_loss(y_true, y_pred, neg_pos_ratio = 3, alpha = 1):
    batch_size = tf.shape(y_pred)[0]
    n_boxes = tf.shape(y_pred)[1]

    neg_pos_ratio = tf.constant(neg_pos_ratio)
    classification_loss = tf.cast(log_loss(y_true[:, :, :-4], y_pred[:, :, :-4]), tf.float32)
    localization_loss = tf.cast(l1_loss(y_true[:, :, -4:], y_pred[:, :, -4:]), tf.float32)

    negatives = y_true[:, :, 0]
    positives = tf.cast(tf.math.reduce_max(y_true[:, :, 1:-4], axis=-1), tf.float32)

    n_positive = tf.reduce_sum(positives)

    pos_class_loss = tf.reduce_sum(classification_loss * positives, axis =-1)

    neg_class_loss_all= classification_loss * negatives
    n_neg_losses = tf.math.count_nonzero(neg_class_loss_all, tf.int32)

    n_negative_keep = tf.minimum(neg_pos_ratio * tf.cast(positives, tf.int32), n_neg_losses)

    neg_class_loss_1D = tf.reshape(neg_class_loss_all, [-1])
    _, indices = tf.nn.top_k(neg_class_loss_1D, n_negative_keep, sorted = False)

    negatives_keep = tf.scatter_nd(indices = tf.expand_dims(indices, axis=1), updates = tf.ones_like(indices, tf.int32), shape = tf.shape(neg_class_loss_1D))
    negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, n_boxes]), tf.float32)

    neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis = -1)

    class_loss= pos_class_loss + neg_class_loss

    loc_loss = tf.reduce_sum(localization_loss * positives, axis =-1)

    total_loss = tf.reduce_sum(class_loss + alpha * loc_loss) / tf.maximum(1.0, n_positive)

    total_loss *= tf.cast(batch_size, tf.float32)


    return total_loss
