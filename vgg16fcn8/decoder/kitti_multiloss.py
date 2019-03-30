"""
An implementation of FCN in tensorflow.
------------------------

The MIT License (MIT)

Copyright (c) 2016 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import decoder_utils as dutils
from utils import classes_utils as cutils

def _add_softmax(hypes, logits):
    num_classes = cutils.get_num_classes(hypes)
    with tf.name_scope('decoder'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=hypes['solver']['epsilon'])
        # logits = logits + epsilon

        softmax = tf.nn.softmax(logits)

    return softmax


def decoder(hypes, logits, train):
    """Apply decoder to the logits.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].

    Return:
      logits: the logits are already decoded.
    """
    decoded_logits = {}
    decoded_logits['logits'] = logits['fcn_logits']
    decoded_logits['softmax'] = _add_softmax(hypes, logits['fcn_logits'])
    return decoded_logits


def loss(hypes, decoded_logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    logits = decoded_logits['logits']
    with tf.name_scope('loss'):
        num_classes = cutils.get_num_classes(hypes)
        logits = tf.reshape(logits, (-1, num_classes))
        shape = [logits.get_shape()[0], num_classes]
        epsilon = tf.constant(value=hypes['solver']['epsilon'])
        # logits = logits + epsilon
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if hypes['loss'] == 'xentropy':
            cross_entropy_mean = _compute_cross_entropy_mean(hypes, labels,
                                                             softmax)
        elif hypes['loss'] == 'softF1':
            cross_entropy_mean = _compute_f1(hypes, labels, softmax, epsilon)

        elif hypes['loss'] == 'softIU':
            cross_entropy_mean = _compute_soft_ui(hypes, labels, softmax,
                                                  epsilon)

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

        weight_loss = tf.add_n(tf.get_collection(reg_loss_col),
                               name='reg_loss')

        total_loss = cross_entropy_mean + weight_loss

        losses = {}
        losses['total_loss'] = total_loss
        losses['xentropy'] = cross_entropy_mean
        losses['weight_loss'] = weight_loss

    return losses


def _compute_cross_entropy_mean(hypes, labels, softmax):
    head = hypes['arch']['weight']
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), head),
                                   reduction_indices=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='xentropy_mean')
    return cross_entropy_mean


def _compute_f1(hypes, labels, softmax, epsilon):
    labels = tf.to_float(tf.reshape(labels, (-1, cutils.get_num_classes(hypes))))[:, 1]
    logits = softmax[:, 1]
    true_positive = tf.reduce_sum(labels*logits)
    false_positive = tf.reduce_sum((1-labels)*logits)

    recall = true_positive / tf.reduce_sum(labels)
    precision = true_positive / (true_positive + false_positive + epsilon)

    score = 2*recall * precision / (precision + recall)
    f1_score = 1 - 2*recall * precision / (precision + recall)

    return f1_score


def _compute_soft_ui(hypes, labels, softmax, epsilon):
    intersection = tf.reduce_sum(labels*softmax, reduction_indices=0)
    union = tf.reduce_sum(labels+softmax, reduction_indices=0) \
        - intersection+epsilon

    mean_iou = 1-tf.reduce_mean(intersection/union, name='mean_iou')

    return mean_iou


def evaluation(hyp, images, labels, decoded_logits, losses, global_step):
    return dutils.evaluation_original(hyp, images, labels, decoded_logits, losses, global_step)
