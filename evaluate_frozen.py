
import logging
import os.path
import scipy as scp
import tensorflow as tf
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('RUN', '', 'Modifier for model parameters.')
flags.DEFINE_string('data_file', '', 'Modifier for model parameters.')

MODEL_NAME = FLAGS.RUN
data_file = FLAGS.data_file

image_dir = os.path.dirname(data_file)
PATH_TO_FROZEN = 'RUNS/' + MODEL_NAME + '/frozen_model.pb'
output_dir = "RUNS/" + MODEL_NAME + "/test_images/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            l_input = detection_graph.get_tensor_by_name('Inputs/fifo_queue_Dequeue:0')  # Input Tensor
            l_output = detection_graph.get_tensor_by_name('upscore32/conv2d_transpose:0')  # Output Tensor

            with open(data_file) as file:
                for i, image_file in enumerate(file):
                    image_file = image_file.rstrip()
                    image_file = os.path.join(image_dir, image_file)
                    image = scp.misc.imread(image_file)
                    shape = image.shape

                    output = sess.run(l_output, feed_dict={l_input: image})

                    a = output[0]

                    output_im1 = a[..., 0]
                    output_im2 = a[..., 1]

                    name = os.path.basename(image_file)
                    body, ext = name.split(".")

                    new_name = body + "_bw." + ext
                    save_file = os.path.join(output_dir, new_name)
                    logging.info("Writing file: %s", save_file)
                    scp.misc.imsave(save_file, output_im1)

                    new_name = body + "_wb." + ext
                    save_file = os.path.join(output_dir, new_name)
                    logging.info("Writing file: %s", save_file)
                    scp.misc.imsave(save_file, output_im2)
