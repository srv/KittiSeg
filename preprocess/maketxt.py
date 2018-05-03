
import tensorflow as tf
import os
import random
import math

"""
generate testing and training file lists
"""

flags = tf.app.flags
flags.DEFINE_string('data_path', '', 'Directory of the dataset where the data is stored.')
FLAGS = flags.FLAGS

folders = ["testing", "training"]

for folder in folders:

    full_path = FLAGS.data_path + "/" + folder + "/images"

    for root, dirs, files in os.walk(full_path):

        files.sort()

        if folder == "testing":
            f = open(FLAGS.data_path + "/testing.txt", 'w')
            for i in files:
                f.write(folder + "/images/" + i + '\n')  # write test images
            f.close()

        if folder == "training":

            length = len(files)

            nVal = math.floor(length*0.05)  # select validation files
            iVal = random.sample(range(0, length), nVal)
            iVal.sort()
            filesVal = [files[i] for i in iVal]

            filesTrain = files  # select train files
            for i in sorted(iVal, reverse=True):
                del filesTrain[i]

            f = open(FLAGS.data_path + "/train3.txt", 'w')
            for i in filesTrain:  # write train images
                name, ext = i.split(".")
                f.write(folder + "/images/" + i + " " + folder + "/gt_images/" + name + "_gt." + ext + '\n')
            f.close()

            f = open(FLAGS.data_path + "/val3.txt", 'w')
            for i in filesVal:  # write validation images
                name, ext = i.split(".")
                f.write(folder + "/images/" + i + " " + folder + "/gt_images/" + name + "_gt." + ext + '\n')
            f.close()



