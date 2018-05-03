import tensorflow as tf
from scipy import ndimage
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize


"""
evaluate the output classification vs its corresponding ground truth, generating a confusion matrix
"""

flags = tf.app.flags
flags.DEFINE_string('im_path_gt', '', 'Directory where the gt images are stored.')
flags.DEFINE_string('im_path_test', '', 'Directory where the output classification images of the model are stored.')
flags.DEFINE_string('read_mode', 'P', 'mode used to read images')

FLAGS = flags.FLAGS

images_gt = []
images_output = []

for root, dirs, files in os.walk(FLAGS.im_path_gt):  # for each folder

    files.sort()  # sort files by name

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image

            image = ndimage.imread(filepath, mode=FLAGS.read_mode)  # read image
            im_flat = image.flatten()
            images_gt.append(im_flat)  # store image in a list

for root, dirs, files in os.walk(FLAGS.im_path_test):  # for each folder

    files.sort()  # sort files by name

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image

            image = ndimage.imread(filepath, mode=FLAGS.read_mode)  # read image
            im_flat = image.flatten()
            images_output.append(im_flat)  # store images in a list

cnf_matrix = np.zeros((2, 2))  # auxiliary image

for i in enumerate(images_gt):

    a = images_gt[i[0]]
    b = images_output[i[0]]
    cnf = confusion_matrix(a, b)
    s = cnf.shape

    if s == (1, 1):
        n = a.size
        valor = a[0]

        if valor == 255:
            cnf_matrix[1, 1] = cnf_matrix[1, 1] + n
        elif valor == 0:
            cnf_matrix[0, 0] = cnf_matrix[0, 0] + n

    else:
        cnf_matrix = cnf + cnf_matrix

cnf_norm = normalize(cnf_matrix, norm='l1', axis=1)
s = np.array_str(cnf_norm)
f = open(FLAGS.im_path_test + '/evaluation.txt', 'w')
f.write(s)
f.close()

z = 1






