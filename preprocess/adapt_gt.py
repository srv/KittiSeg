
import tensorflow as tf
import numpy as np
from scipy import ndimage
import scipy.misc
import os
import re

"""
set b-w label maps to rgb
"""

flags = tf.app.flags
flags.DEFINE_string('gt_path_in', '', 'Directory where the gt is stored.')
flags.DEFINE_string('gt_path_out', '', 'Directory where the gt will be stored.')
flags.DEFINE_string('read_mode', 'I', 'mode used to read gt images')
FLAGS = flags.FLAGS

for root, dirs, files in os.walk(FLAGS.gt_path_in):  # for each folder

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image
            image = ndimage.imread(filepath, mode=FLAGS.read_mode)  # read image

            img = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)  # auxiliary image

            img[..., 0] = 255  # set red layer

            y = np.where(image != 255) # get black pixels
            c = img[..., 2]
            c[y[0], y[1]] = 255
            img[..., 2] = c  # set blue layer

            

            scipy.misc.imsave(FLAGS.gt_path_out + "/" + file[1], img)  # generate image file

