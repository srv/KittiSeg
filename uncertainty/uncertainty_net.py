
import tensorflow as tf
import numpy as np
from scipy import ndimage
import scipy.misc
import os
import re

"""
calculate uncertainty area of network output
"""

img = np.full([360, 480, 3], 255, dtype=np.uint8)  # auxiliary image

path = "net/"

image_low = ndimage.imread(path + "bw_low.png", mode='I')  # read image

image_high = ndimage.imread(path + "bw_high.png", mode='I')  # read image

aux = image_low - image_high  # compute difference

a = np.where(aux != 0)  # get uncertainty pixels

img[a[0], a[1], 1] = 0
img[a[0], a[1], 2] = 0


scipy.misc.imsave("incertidumbre_net.png", img)  # generate image file

