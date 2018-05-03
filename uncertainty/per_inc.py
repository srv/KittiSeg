
import tensorflow as tf
import numpy as np
from scipy import ndimage
import scipy.misc
import os
import re

"""
calculate the error percentage of an output classification
"""


path = ""

name = "err128"

image = ndimage.imread(path + name + ".png", mode='RGB')  # read image

pixels = image.size/3  # get number of total pixels

error = np.where(image != [255, 255, 255])  # get error pixels

n_err = len(error[0])  # get number of error pixels

per = (n_err/pixels)*100  # calculate percentage of error pixels

print("Percentaje: " + str(per))



