import tensorflow as tf
from scipy import ndimage
import scipy
import os
import re
import numpy as np

"""
bianrize an image given a threshold
"""

flags = tf.app.flags
flags.DEFINE_string('im_path_in', '', 'Directory where the input images are stored.')
flags.DEFINE_string('im_path_out', '', 'Directory where the output images will be stored.')
flags.DEFINE_string('threshold', '', 'grey level threshold')
FLAGS = flags.FLAGS

thr = int(FLAGS.threshold)

for root, dirs, files in os.walk(FLAGS.im_path_in):  # for each folder

    for file in enumerate(files):                    # for each file in the folder

        filepath = os.path.join(root, file[1])       # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image

            image = ndimage.imread(filepath, mode='RGB')  # read image

            n1, n2, trash = file[1].split("_")  # keep image name
            trash, ext = file[1].split(".")     # keep image extension

            img = np.full([image.shape[0], image.shape[1]], 0, dtype=np.uint8)  # auxiliary image
            y = np.where(image > thr)   # get pixels above threshold
            img[y[0], y[1]] = 255       # set pixels above threshold to white

            scipy.misc.imsave(FLAGS.im_path_out + "/" + n1 + "_" + n2 +  "_bw." + ext, img)  # generate image file

