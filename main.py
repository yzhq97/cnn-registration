from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image

from VGG16mo import VGG16mo

height = 224
width = 224

datadir = '../data/RemoteSense/ANGLE/68/'
IX_name = '2.jpg'
IY_name = '7.jpg'

IX_image = Image.open(datadir+IX_name)
IY_image = Image.open(datadir+IX_name)

IX_image = IX_image.resize((width, height))
IY_image = IY_image.resize((width, height))

IX = np.asarray(IX_image, dtype='float32')
IY = np.asarray(IY_image, dtype='float32')
IX = np.expand_dims(IX, axis=0)
IY = np.expand_dims(IY, axis=0)
input = np.concatenate((IX, IY), axis=0)

with tf.Session() as sess:
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: input}
    vgg = VGG16mo()
    vgg.build(images)
    pool1, idx1, pool2, idx2, pool3, idx3,\
    pool4, idx4, pool5, idx5 = sess.run([
        vgg.pool1, vgg.idx1, vgg.pool2, vgg.idx2,
        vgg.pool3, vgg.idx3, vgg.pool4, vgg.idx4,
        vgg.pool5, vgg.idx5
    ], feed_dict=feed_dict)

Xpool1 = pool1[0]
iX1 = np.unravel_index(idx1[0], [2, 224, 224, 64])[1:]
Xmap1 = np.zeros([224, 224, 64])
Xmap1[iX1[0], iX1[1], iX1[2]] = Xpool1

Xpool2 = pool2[0]
iX2 = np.unravel_index(idx2[0], [2, 112, 112, 128])[1:]
Xmap2_1 = np.zeros([112, 112, 128])
Xmap2_1[iX2[0], iX2[1], iX2[2]] = Xpool2
Xmap2 = np.zeros([224, 224, 128])
