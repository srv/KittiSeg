from scipy import ndimage
import scipy
import os
import re
import sys
import argparse

"""
crop an image
"""


parser = argparse.ArgumentParser()
parser.add_argument('--path_in', help ='Directory where the input images are stored.')
parser.add_argument('--path_out', help ='Directory where the output images will be stored.')
parser.add_argument('--read_mode', default='RGB', help ='mode used to read images')
parser.add_argument('--l', default='0', help ='desired left crop.')
parser.add_argument('--r', default='0', help ='desired right crop.')
parser.add_argument('--t', default='0', help ='desired top crop.')
parser.add_argument('--b', default='0', help ='desired bottom crop.')
parsed_args = parser.parse_args(sys.argv[1:])

path_in = parsed_args.path_in
path_out = parsed_args.path_out
read_mode = parsed_args.read_mode
l = int(parsed_args.l)
r = int(parsed_args.r)
t = int(parsed_args.t)
b = int(parsed_args.b)


for root, dirs, files in os.walk(path_in):  # for each folder

    for file in enumerate(files):  # for each file in the folder

        filepath = os.path.join(root, file[1])  # file path

        if re.search("\.(png|jpg|jpeg)$", file[1]):  # if the file is an image

            name,ext = os.path.splitext(file[1])
            image = ndimage.imread(filepath, mode=read_mode)  # read image
            image_crop = image[t:-b, l:-r]   # crop

            scipy.misc.imsave(path_out + "/" + name + '_crop' + ext , image_crop)  # generate image file
