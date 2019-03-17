#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
from seg_utils import seg_utils as seg
from sklearn import metrics

import time
import tensorvision.utils as utils
import utils.classes_utils as cutils


def eval_image(hypes, gt_image, cnn_image):

    flat_gt_image = gt_image.flatten()
    flat_cnn_image = cnn_image.flatten()

    confusion_matrix = metrics.confusion_matrix(flat_gt_image, flat_cnn_image, range(cutils.get_num_classes(hypes)))

    return confusion_matrix


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def evaluate(hypes, sess, image_pl, inf_out):

    softmax = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']

    color_dict = cutils.get_output_color_dict(hypes)

    num_classes = cutils.get_num_classes(hypes)

    eval_dict = {}
    for phase in ['train', 'val']:
        total_confusion_matrix = np.zeros([num_classes, num_classes], int)
        data_file = hypes['data']['{}_file'.format(phase)]
        data_file = os.path.join(data_dir, data_file)
        image_dir = os.path.dirname(data_file)

        image_list = []

        with open(data_file) as file:
            for i, datum in enumerate(file):
                datum = datum.rstrip()
                image_file, gt_file = datum.split(" ")
                image_file = os.path.join(image_dir, image_file)
                gt_file = os.path.join(image_dir, gt_file)

                image = scp.misc.imread(image_file, mode='RGB')
                gt_image = scp.misc.imread(gt_file, mode='RGB')

                if hypes['jitter']['fix_shape']:
                    shape = image.shape
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    assert(image_height >= shape[0])
                    assert(image_width >= shape[1])

                    offset_x = (image_height - shape[0])//2
                    offset_y = (image_width - shape[1])//2
                    new_image = np.zeros([image_height, image_width, 3])
                    new_image[offset_x:offset_x+shape[0],
                              offset_y:offset_y+shape[1]] = image
                    input_image = new_image
                elif hypes['jitter']['reseize_image']:
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    image, gt_image = resize_label_image(image, gt_image,
                                                         image_height,
                                                         image_width)
                    input_image = image
                else:
                    input_image = image

                shape = input_image.shape

                feed_dict = {image_pl: input_image}

                output = sess.run([softmax], feed_dict=feed_dict)
                output_im = output[0].argmax(axis=1).reshape(shape[0], shape[1])
                output_prob = output[0].max(axis=1).reshape(shape[0], shape[1])

                if hypes['jitter']['fix_shape']:
                    gt_shape = gt_image.shape
                    output_im = output_im[offset_x:offset_x+gt_shape[0],
                                          offset_y:offset_y+gt_shape[1]]
                    output_prob = output_prob[offset_x:offset_x + gt_shape[0],
                                              offset_y:offset_y + gt_shape[1]]

                if phase == 'val':
                    # Saving RB Plot
                    ov_image = seg.make_overlay(image, output_prob)
                    name = os.path.basename(image_file)
                    image_list.append((name, ov_image))

                    name2 = name.split('.')[0] + '_prediction.png'

                    prediction_image = utils.overlay_segmentation(image, output_im, color_dict)
                    image_list.append((name2, prediction_image))

                confusion_matrix = eval_image(hypes, cutils.get_gt_image_index(gt_image, hypes), output_im)

                total_confusion_matrix += confusion_matrix

        classes_result = {"confusion_matrix": total_confusion_matrix,
                          "normalized_confusion_matrix": normalize_confusion_matrix(total_confusion_matrix)}
        name_classes = cutils.get_name_classes(hypes)
        for i in range(num_classes):
            classes_result[name_classes[i]] = obtain_class_result(total_confusion_matrix, i)

        eval_dict[phase] = {"classes_result": classes_result}

        if phase == 'val':
            start_time = time.time()
            for i in xrange(10):
                sess.run([softmax], feed_dict=feed_dict)
            dt = (time.time() - start_time)/10

    eval_list = []

    for phase in ['train', 'val']:
        print(phase)
        print(eval_dict[phase]["classes_result"]["confusion_matrix"])
        print(eval_dict[phase]["classes_result"]["normalized_confusion_matrix"])
        for class_name in name_classes:
            eval_list.append(('[{} {}] Recall'.format(phase, class_name), 100 * eval_dict[phase]["classes_result"][class_name]["recall"]))
            eval_list.append(('[{} {}] Precision'.format(phase, class_name), 100 * eval_dict[phase]["classes_result"][class_name]["precision"]))
            eval_list.append(('[{} {}] TNR'.format(phase, class_name), 100 * eval_dict[phase]["classes_result"][class_name]["TNR"]))
            eval_list.append(('[{} {}] Accuracy'.format(phase, class_name), 100 * eval_dict[phase]["classes_result"][class_name]["accuracy"]))
            eval_list.append(('[{} {}] F1'.format(phase, class_name), 100 * eval_dict[phase]["classes_result"][class_name]["F1"]))

    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))
    return eval_list, image_list


def obtain_class_result(confusion_matrix, index_class):
    all_result = np.sum(confusion_matrix)
    class_result = {}
    true_positive = confusion_matrix[index_class, index_class]
    false_positive = np.sum(confusion_matrix[:, index_class]) - true_positive
    false_negative = np.sum(confusion_matrix[index_class, :]) - true_positive
    true_negative = all_result - true_positive - false_positive - false_negative
    recall = np.nan_to_num(true_positive / (false_negative + true_positive))
    precision = np.nan_to_num(true_positive / (true_positive + false_positive))
    accuracy = np.nan_to_num((true_positive + true_negative) / all_result)
    F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    TNR = np.nan_to_num(true_negative / (true_negative + false_positive))
    class_result["TP"] = true_positive
    class_result["TN"] = true_negative
    class_result["FP"] = false_positive
    class_result["FN"] = false_negative
    class_result["recall"] = recall
    class_result["precision"] = precision
    class_result["TNR"] = TNR
    class_result["accuracy"] = accuracy
    class_result["F1"] = F1
    return class_result


def normalize_confusion_matrix(confusion_matrix):
    normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_confusion_matrix[np.isnan(normalized_confusion_matrix)] = 0
    return normalized_confusion_matrix