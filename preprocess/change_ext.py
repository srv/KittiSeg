import tensorflow as tf
from scipy import ndimage
import scipy
import os
import re

"""
change a file extension
"""

flags = tf.app.flags
flags.DEFINE_string('im_path_in', '', 'Directory where the input images are stored.')
flags.DEFINE_string('im_path_out', '', 'Directory where the output images will be stored.')
flags.DEFINE_string('read_mode', 'RGB', 'mode used to read images')
flags.DEFINE_string('extension', 'png', 'desired extension')
FLAGS = flags.FLAGS

for root, dirs, files in os.walk(FLAGS.im_path_in):  # for each folder

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image
            image = ndimage.imread(filepath, mode=FLAGS.read_mode)  # read image

            name, ext = file[1].split(".")  # get name

            scipy.misc.imsave(FLAGS.im_path_out + "/" + name + "." + FLAGS.extension, image)  # generate image file with specificated extension
