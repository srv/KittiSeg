
import tensorflow as tf
import numpy as np
from scipy import ndimage
import scipy.misc
import os
import re

"""
calculate percentage of intersection between the error network and the uncertainty area of hand labeled ground truths
"""


path = ""

err_file = "err128"
unc_file = "incertidumbre_hand_levels"

aux = np.full([360, 480], 0)

err = ndimage.imread(path + err_file + ".png", mode='RGB')  # read image
inc = ndimage.imread(path + unc_file + ".png", mode='RGB')  # read image

err_pixels = np.where(err[..., 0] == 0)  # get error pixels
inc_pixels = np.where(inc[..., 1] != 255)  # get uncertainty area pixels

aux[err_pixels[0], err_pixels[1]] = aux[err_pixels[0], err_pixels[1]] + 100  # stack
aux[inc_pixels[0], inc_pixels[1]] = aux[inc_pixels[0], inc_pixels[1]] + 150  # stack


no_inter = np.where(aux == 100)  # get no intersection pixels
inter = np.where(aux == 250)     # get intersection pixels

n_nointer = len(no_inter[0])     # number of no intersection pixels
n_inter = len(inter[0])          # number of intersection pixels

per = (n_inter/(n_nointer+n_inter))*100  # percentage of intersection pixels

print("Percentaje: " + str(per))



