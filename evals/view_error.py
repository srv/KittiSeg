import tensorflow as tf
from scipy import ndimage
import numpy as np
import os
import re
from scipy import ndimage
import scipy.misc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

"""
compare the output classification vs its corresponding ground truth, generating a new image, marking the FP and FN areas
"""

flags = tf.app.flags
flags.DEFINE_string('im_path_gt', '', 'Directory where the gt images are stored.')
flags.DEFINE_string('im_path_class', '', 'Directory where the output classification images of the model are stored.')
flags.DEFINE_string('im_path_out', '', 'Directory where the err images will be stored.')
flags.DEFINE_string('read_mode', 'P', 'mode used to read images')

FLAGS = flags.FLAGS

images_gt = []
images_class = []
names = []

for root, dirs, files in os.walk(FLAGS.im_path_gt):  # for each folder

    files.sort()  # sort files by name

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image

            image = ndimage.imread(filepath, mode=FLAGS.read_mode)  # read image
            images_gt.append(image)  # store image in a list
            names.append(file[1])    # save names

for root, dirs, files in os.walk(FLAGS.im_path_class):  # for each folder

    files.sort()  # sort files by name

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image

            image = ndimage.imread(filepath, mode=FLAGS.read_mode)  # read image
            images_class.append(image)  # store images in a list

cnf_matrix = np.zeros((2, 2))   # auxiliary image

for i in enumerate(images_gt):  # for each image

    gt = images_gt[i[0]]
    clas = images_class[i[0]]
    name = names[i[0]]

    sub = gt-clas  # compute difference

    img = np.full([sub.shape[0], sub.shape[1], 3], 255, dtype=np.uint8)  # auxiliary image

    FN = np.where(sub == 255)  # false negative error

    img[FN[0], FN[1], 0] = 0  # set red layer
    img[FN[0], FN[1], 1] = 0  # set green layer

    FP = np.where(sub == 1)  # false positive error

    img[FP[0], FP[1], 0] = 0  # set red layer
    img[FP[0], FP[1], 2] = 0  # set blue layer

    name, ext = name.split(".")

    scipy.misc.imsave( FLAGS.im_path_out + "/" + name + "_err." + ext, img)  # generate image file







