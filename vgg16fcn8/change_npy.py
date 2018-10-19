import tensorflow as tf
import scipy
import numpy as np
from scipy import interpolate

"""
change fc6 layer of the pretrained weights
"""

data_dict = np.load('/home/miguel/Escritorio/semantic/KittiSeg/pesos/77/vgg16.npy', encoding='latin1').item()

name = "fc6"
shape = [7, 7, 512, 4096]

weights1 = data_dict[name][0]

weights2 = weights1.reshape(shape)

weights2updated = np.zeros([12, 15, 512, 4096], dtype=np.float32)
w2updated = np.zeros([12, 15], dtype=np.float32)

for i in range(0, weights2.shape[3]):

    for j in range(0, weights2.shape[2]):

        w2 = weights2[..., j, i]

        x = np.array(range(w2.shape[0]))
        y = np.array(range(w2.shape[1]))

        xx, yy = np.meshgrid(x, y)
        f = interpolate.interp2d(x, y, w2, kind='linear')

        xnew = np.linspace(0, 6, 15)
        ynew = np.linspace(0, 6, 12)

        w2updated = f(xnew, ynew)

        weights2updated[..., j, i] = w2updated

d = weights2updated.shape

data_dict[name][0] = weights2updated

np.save('vgg16.npy', data_dict)

z = 1
