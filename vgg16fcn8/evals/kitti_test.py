#!/usr/bin/env python
# pylint: disable=missing-docstring
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import json
import logging
import numpy as np
import os.path
import sys

import scipy as scp
import scipy.misc


sys.path.insert(1, '../../incl')

import tensorflow as tf

import tensorvision.utils as utils
import tensorvision.core as core
import time as time
from evals import kitti_eval
from seg_utils import seg_utils as seg
import utils.classes_utils as cutils

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

flags = tf.app.flags
FLAGS = flags.FLAGS

# test_file = 'data_road/testing.txt'


def create_test_output(hypes, sess, image_pl, softmax, data_file):
    # data_dir = hypes['dirs']['data_dir']
    # data_file = os.path.join(data_dir, test_file)

    image_dir = os.path.dirname(data_file)

    logdir_prediction = "test_images_prediction/"

    logging.info("Images will be written to {}/test_images_{{prediction, rg}}"
                 .format(logdir_prediction))

    logdir_prediction = os.path.join(hypes['dirs']['output_dir'], logdir_prediction)

    if not os.path.exists(logdir_prediction):
        os.mkdir(logdir_prediction)

    num_classes = cutils.get_num_classes(hypes)
    color_dict = cutils.get_output_color_dict(hypes)
    total_confusion_matrix = np.zeros([num_classes, num_classes], int)
    name_classes = cutils.get_name_classes(hypes)

    with open(data_file) as file:
        for i, datum in enumerate(file):

            t = time.time()

            image_file = datum.rstrip()
            if len(image_file.split(" ")) > 1:
                image_file, gt_file = image_file.split(" ")
                gt_file = os.path.join(image_dir, gt_file)
                gt_image = scp.misc.imread(gt_file, mode='RGB')
            image_file = os.path.join(image_dir, image_file)
            image = scp.misc.imread(image_file)
            shape = image.shape

            feed_dict = {image_pl: image}

            output = sess.run([softmax['softmax']], feed_dict=feed_dict)
            output_im = output[0].argmax(axis=1).reshape(shape[0], shape[1])

            # Saving RB Plot
            name = os.path.basename(image_file)

            new_name = name.split('.')[0] + '_prediction.png'

            prediction_image = utils.overlay_segmentation(image, output_im, color_dict)

            save_file = os.path.join(logdir_prediction, new_name)
            scp.misc.imsave(save_file, prediction_image)

            elapsed = time.time() - t
            print("elapsed time: " + str(elapsed))
            if 'gt_image' in locals():
                confusion_matrix = kitti_eval.eval_image(hypes, cutils.get_gt_image_index(gt_image, hypes), output_im)
                total_confusion_matrix += confusion_matrix
                for j in range(num_classes):
                    gray_scale_file_name = name.split('.')[0] + '_' + name_classes[j] + '_grayscale.png'
                    save_file = os.path.join(logdir_prediction, gray_scale_file_name)
                    output_prob_class = np.around(output[0][:, j].reshape(shape[0], shape[1]) * 255)
                    scp.misc.imsave(save_file, output_prob_class)

        if 'gt_image' in locals():
            normalized_total_confusion_matrix = total_confusion_matrix.astype('float') / total_confusion_matrix.sum(axis=1)[:, np.newaxis]
            normalized_total_confusion_matrix[np.isnan(normalized_total_confusion_matrix )] = 0
            classes_result = {
                "confusion_matrix": total_confusion_matrix,
                "normalized_confusion_matrix": normalized_total_confusion_matrix
            }
            for i in range(num_classes):
                classes_result[name_classes[i]] = kitti_eval.obtain_class_result(total_confusion_matrix, i)
            eval_result = {"classes_result": classes_result}
            print(eval_result)


def _create_input_placeholder():
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.float32)
    return image_pl, label_pl


def do_inference(logdir, data_file):
    """
    Analyze a trained model.

    This will load model files and weights found in logdir and run a basic
    analysis.

    Parameters
    ----------
    logdir : string
        Directory with logs.
    """
    hypes = utils.load_hypes_from_logdir(logdir)
    modules = utils.load_modules_from_logdir(logdir)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # prepare the tv session

        with tf.name_scope('Validation'):
            image_pl, label_pl = _create_input_placeholder()
            image = tf.expand_dims(image_pl, 0)
            softmax = core.build_inference_graph(hypes, modules, image=image)

        sess = tf.Session()
        saver = tf.train.Saver()

        core.load_weights(logdir, sess, saver)
        create_test_output(hypes, sess, image_pl, softmax, data_file)

    return


def main(_):

    """Run main function."""
    if FLAGS.logdir is None:
        logging.error("No logdir are given.")
        logging.error("Usage: tv-analyze --logdir dir")
        exit(1)

    if FLAGS.gpus is None:
        if 'TV_USE_GPUS' in os.environ:
            if os.environ['TV_USE_GPUS'] == 'force':
                logging.error('Please specify a GPU.')
                logging.error('Usage tv-train --gpus <ids>')
                exit(1)
            else:
                gpus = os.environ['TV_USE_GPUS']
                logging.info("GPUs are set to: %s", gpus)
                os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        logging.info("GPUs are set to: %s", FLAGS.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    utils.load_plugins()

    logdir = os.path.realpath(FLAGS.logdir)

    logging.info("Starting to analyze Model in: %s", logdir)
    do_inference(logdir)



if __name__ == '__main__':
    tf.app.run()

