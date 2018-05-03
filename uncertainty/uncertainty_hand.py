
import tensorflow as tf
import numpy as np
from scipy import ndimage
import scipy.misc
import os
import re

"""
calculate uncertainty area of hand labeled ground truths
"""

img = np.full([360, 480, 3], 255, dtype=np.uint8)  # auxiliary image
aux = np.full([360, 480], 0)
n_im = 0

for root, dirs, files in os.walk("hand/"):  # for each folder

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image

            n_im = n_im + 1
            image = ndimage.imread(filepath, mode='I')  # read image
            aux = aux + image

a = np.where(aux == 255*n_im)   # get max value pixels (all coincided)
aux[a[0], a[1]] = 0             # set them to 0
aux = aux/(n_im)                # compute mean

b = np.where(aux > (255/2))             # split in half
aux[b[0], b[1]] = 255-aux[b[0], b[1]]   # mirror values

aux = 2*aux     # resize to 0-255
aux = 255-aux

c = np.where(aux != 255)  # find uncertainty pixels

img[c[0], c[1], 0] = 255
img[c[0], c[1], 1] = aux[c[0], c[1]]  # 0 #
img[c[0], c[1], 2] = aux[c[0], c[1]]  # 0 #

scipy.misc.imsave("incertidumbre_hand.png", img)  # generate image file

