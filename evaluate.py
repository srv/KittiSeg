#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the KittiSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')


from evaluation import kitti_test

flags.DEFINE_string('RUN', 'KittiSeg_pretrained', 'Modifier for model parameters.')
flags.DEFINE_string('hypes', 'hypes/KittiSeg.json', 'File storing model parameters.')
flags.DEFINE_string('data_file', '', 'File storing test images location.')

def main(_):

    runs_dir = 'RUNS'

    logdir = os.path.join(runs_dir, FLAGS.RUN)

    logging.info("Creating output on test data.")

    kitti_test.do_inference(logdir, FLAGS.data_file)

    logging.info("Analysis for pretrained model complete.")

if __name__ == '__main__':
    tf.app.run()
