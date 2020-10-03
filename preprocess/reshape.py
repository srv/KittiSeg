import argparse
from scipy import ndimage
import scipy
import sys
import os
import re

"""
reshape an image
"""

parser = argparse.ArgumentParser()
parser.add_argument('--path_in', help ='Directory where the input images are stored.')
parser.add_argument('--path_out', help ='Directory where the output images will be stored.')
parser.add_argument('--read_mode', default='RGB', help ='mode used to read images')
parser.add_argument('--h', default='0', help ='desired height.')
parser.add_argument('--w', default='0', help ='desired width.')
parsed_args = parser.parse_args(sys.argv[1:])

path_in = parsed_args.path_in
path_out = parsed_args.path_out
read_mode = parsed_args.read_mode
h = int(parsed_args.h)
w = int(parsed_args.w)

for root, dirs, files in os.walk(path_in):  # for each folder

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image
            image = ndimage.imread(filepath, mode=read_mode)  # read image

            img = scipy.misc.imresize(image, (h, w))  # resized image

            name, ext = file[1].split(".")

            scipy.misc.imsave(path_out + "/" + file[1], img)  # generate image file
